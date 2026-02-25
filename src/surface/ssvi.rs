//! SSVI (Surface SVI) global parameterization.
//!
//! SSVI parameterizes the entire volatility surface (not just a single tenor)
//! as a function of total variance at-the-money and log-moneyness. The core
//! formula for total implied variance is:
//!
//! ```text
//! w(k, θ) = (θ/2) · [1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²))]
//! ```
//!
//! where `φ(θ) = η / θ^γ` is the power-law mixing function.
//!
//! The three global parameters `(ρ, η, γ)` control:
//! - `ρ ∈ (−1, 1)`: skew direction and magnitude
//! - `η > 0`: smile amplitude
//! - `γ ∈ [0, 1]`: term structure decay of the smile
//!
//! Per-tenor ATM total variances `θ_i = σ²_ATM,i · T_i` anchor the surface
//! to market-observed ATM levels.
//!
//! # No-Arbitrage Conditions
//!
//! The power-law φ satisfies the sufficient condition for no butterfly arbitrage
//! (Theorem 4.1) when `η·(1 + |ρ|) ≤ 2`.
//!
//! # References
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
use crate::surface::VolSurface;
use crate::surface::arbitrage::{CalendarViolation, SurfaceDiagnostics};
use crate::types::{Variance, Vol};
use crate::validate::validate_positive;

/// SSVI volatility surface parameterized by Gatheral-Jacquier (2014).
///
/// Stores three global parameters `(ρ, η, γ)` plus per-tenor data
/// `(T_i, F_i, θ_i)`. The surface is fully determined by these values
/// and can evaluate the SSVI total variance formula at any `(T, K)`.
///
/// # Construction
///
/// ```
/// use volsurf::surface::SsviSurface;
///
/// let surface = SsviSurface::new(
///     -0.3,                              // rho: skew
///     0.5,                               // eta: smile amplitude
///     0.5,                               // gamma: term decay
///     vec![0.25, 0.5, 1.0],             // tenors (years)
///     vec![100.0, 100.0, 100.0],        // forward prices
///     vec![0.04, 0.08, 0.16],           // ATM total variances (theta)
/// ).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SsviSurfaceRaw", into = "SsviSurfaceRaw")]
pub struct SsviSurface {
    /// Global skew parameter ρ ∈ (−1, 1).
    rho: f64,
    /// Smile amplitude parameter η > 0.
    eta: f64,
    /// Term structure decay parameter γ ∈ [0, 1].
    gamma: f64,
    /// Per-tenor expiries, strictly increasing and positive.
    tenors: Vec<f64>,
    /// Forward prices at each tenor, all positive.
    forwards: Vec<f64>,
    /// ATM total variance θ_i = σ²_ATM,i · T_i at each tenor, strictly increasing.
    thetas: Vec<f64>,
    /// Precomputed 1 − ρ² for the hot path.
    one_minus_rho_sq: f64,
}

#[derive(Serialize, Deserialize)]
struct SsviSurfaceRaw {
    rho: f64,
    eta: f64,
    gamma: f64,
    tenors: Vec<f64>,
    forwards: Vec<f64>,
    thetas: Vec<f64>,
}

impl TryFrom<SsviSurfaceRaw> for SsviSurface {
    type Error = VolSurfError;
    fn try_from(raw: SsviSurfaceRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.rho,
            raw.eta,
            raw.gamma,
            raw.tenors,
            raw.forwards,
            raw.thetas,
        )
    }
}

impl From<SsviSurface> for SsviSurfaceRaw {
    fn from(s: SsviSurface) -> Self {
        Self {
            rho: s.rho,
            eta: s.eta,
            gamma: s.gamma,
            tenors: s.tenors,
            forwards: s.forwards,
            thetas: s.thetas,
        }
    }
}

impl SsviSurface {
    /// Create an SSVI surface from global parameters and per-tenor data.
    ///
    /// # Arguments
    /// * `rho` — Global skew, must be in (−1, 1)
    /// * `eta` — Smile amplitude, must be positive
    /// * `gamma` — Term structure decay, must be in \[0, 1\]
    /// * `tenors` — Expiries in years, strictly increasing, all positive
    /// * `forwards` — Forward prices at each tenor, all positive
    /// * `thetas` — ATM total variances θ_i, strictly increasing, all positive
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is invalid.
    pub fn new(
        rho: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> error::Result<Self> {
        // Scalar validation
        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho| must be less than 1, got {rho}"),
            });
        }
        validate_positive(eta, "eta")?;
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(VolSurfError::InvalidInput {
                message: format!("gamma must be in [0, 1], got {gamma}"),
            });
        }

        // Vector length checks
        if tenors.is_empty() {
            return Err(VolSurfError::InvalidInput {
                message: "at least one tenor is required".into(),
            });
        }
        if tenors.len() != forwards.len() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors and forwards must have the same length, got {} and {}",
                    tenors.len(),
                    forwards.len()
                ),
            });
        }
        if tenors.len() != thetas.len() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors and thetas must have the same length, got {} and {}",
                    tenors.len(),
                    thetas.len()
                ),
            });
        }

        // Element-wise validation
        for (i, &t) in tenors.iter().enumerate() {
            if !t.is_finite() || t <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("tenors must be positive and finite, got tenors[{i}]={t}"),
                });
            }
        }
        for (i, &f) in forwards.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("forwards must be positive and finite, got forwards[{i}]={f}"),
                });
            }
        }
        for (i, &th) in thetas.iter().enumerate() {
            if !th.is_finite() || th <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("thetas must be positive and finite, got thetas[{i}]={th}"),
                });
            }
        }

        // Monotonicity checks
        for w in tenors.windows(2) {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "tenors must be strictly increasing, but {} >= {}",
                        w[0], w[1]
                    ),
                });
            }
        }
        for w in thetas.windows(2) {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "thetas must be strictly increasing, but {} >= {}",
                        w[0], w[1]
                    ),
                });
            }
        }

        Ok(Self {
            one_minus_rho_sq: 1.0 - rho * rho,
            rho,
            eta,
            gamma,
            tenors,
            forwards,
            thetas,
        })
    }

    /// Global skew parameter ρ.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Smile amplitude parameter η.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Term structure decay parameter γ.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Per-tenor expiries.
    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    /// Forward prices at each tenor.
    pub fn forwards(&self) -> &[f64] {
        &self.forwards
    }

    /// ATM total variances θ_i at each tenor.
    pub fn thetas(&self) -> &[f64] {
        &self.thetas
    }

    /// Evaluate the SSVI total variance formula at `(θ, k)`.
    ///
    /// ```text
    /// w(k, θ) = (θ/2) · [1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²))]
    /// ```
    ///
    /// where `φ(θ) = η / θ^γ`.
    ///
    /// This is the hot path — no allocations, no branching.
    pub(crate) fn total_variance_at(&self, theta: f64, k: f64) -> f64 {
        let phi = self.eta / theta.powf(self.gamma);
        let phi_k = phi * k;
        (theta / 2.0)
            * (1.0 + self.rho * phi_k + ((phi_k + self.rho).powi(2) + self.one_minus_rho_sq).sqrt())
    }

    /// Evaluate the power-law mixing function φ(θ) = η / θ^γ.
    #[cfg(test)]
    pub(crate) fn phi(&self, theta: f64) -> f64 {
        self.eta / theta.powf(self.gamma)
    }

    /// Partial derivative ∂w/∂θ of SSVI total variance with respect to
    /// ATM total variance θ, at a given `(θ, k)`.
    ///
    /// ```text
    /// ∂w/∂θ = (1/2) · [1 + ρ·φk·(1−γ) + R − γ·φk·(φk+ρ)/R]
    /// ```
    ///
    /// where `R = √((φk + ρ)² + (1 − ρ²))` and `φ = η/θ^γ`.
    ///
    /// This derivative is provably non-negative for all `(θ, k)` when
    /// `|ρ| < 1` and `γ ∈ [0, 1]`. The expression is linear in γ: at
    /// γ = 0 it equals `w/θ > 0`; at γ = 1 it equals `(1 + (1+ρφk)/R)/2 ≥ 0`
    /// because `(1 + uρ)² ≤ R²` follows from `|ρ| < 1`. Linearity in γ
    /// then guarantees non-negativity for all `γ ∈ [0, 1]`.
    ///
    /// # References
    /// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014), §4
    pub(crate) fn dw_dtheta(&self, theta: f64, k: f64) -> f64 {
        let phi = self.eta / theta.powf(self.gamma);
        let u = phi * k;
        let r = ((u + self.rho).powi(2) + self.one_minus_rho_sq).sqrt();
        0.5 * (1.0 + self.rho * u * (1.0 - self.gamma) + r - self.gamma * u * (u + self.rho) / r)
    }

    /// Analytical calendar arbitrage check for this SSVI surface.
    ///
    /// Scans `∂w/∂θ` (see `dw_dtheta`) on a grid of
    /// `(θ, k)` points at each consecutive tenor pair. Returns calendar
    /// violations where the derivative is negative, indicating total
    /// variance would decrease with increasing ATM variance.
    ///
    /// For valid SSVI surfaces with power-law `φ(θ) = η/θ^γ`, `γ ∈ [0, 1]`,
    /// and `|ρ| < 1`, this always returns an empty vector because `∂w/∂θ ≥ 0`
    /// is mathematically guaranteed. The scan serves as empirical confirmation
    /// of the analytical bound.
    ///
    /// # References
    /// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014), Theorem 4.2
    pub fn calendar_arb_analytical(&self) -> Vec<CalendarViolation> {
        const K_GRID: usize = 41;
        const K_MIN: f64 = -2.0;
        const K_MAX: f64 = 2.0;
        const TOL: f64 = -1e-10;

        let mut violations = Vec::new();

        for i in 0..self.tenors.len().saturating_sub(1) {
            let theta = self.thetas[i];
            let f_avg = 0.5 * (self.forwards[i] + self.forwards[i + 1]);

            for j in 0..K_GRID {
                let k = K_MIN + (K_MAX - K_MIN) * (j as f64) / ((K_GRID - 1) as f64);
                if self.dw_dtheta(theta, k) < TOL {
                    let strike = f_avg * k.exp();
                    let k_short = (strike / self.forwards[i]).ln();
                    let k_long = (strike / self.forwards[i + 1]).ln();
                    violations.push(CalendarViolation {
                        strike,
                        tenor_short: self.tenors[i],
                        tenor_long: self.tenors[i + 1],
                        variance_short: self.total_variance_at(self.thetas[i], k_short),
                        variance_long: self.total_variance_at(self.thetas[i + 1], k_long),
                    });
                }
            }
        }

        violations
    }

    /// Interpolate `(θ, F)` at an arbitrary expiry.
    ///
    /// - **Exact match** (within 1e-10): uses stored values directly.
    /// - **Between tenors**: linear interpolation in θ (total variance) space,
    ///   linear interpolation for forward price.
    /// - **Before first tenor**: flat vol extrapolation `θ(T) = θ₀ · T/T₀`.
    /// - **After last tenor**: flat vol extrapolation `θ(T) = θₙ · T/Tₙ`.
    pub(crate) fn theta_and_forward_at(&self, expiry: f64) -> (f64, f64) {
        super::interp::interpolate_theta_forward(&self.tenors, &self.thetas, &self.forwards, expiry)
    }

    /// Calibrate an SSVI surface from multi-tenor market data.
    ///
    /// Uses a two-stage approach:
    /// 1. Per-tenor SVI calibration to extract ATM total variances (θ) and skew (ρ)
    /// 2. Global optimization of (η, γ) with ρ fixed from the SVI average
    ///
    /// # Arguments
    /// * `market_data` — Per-tenor slices of (strike, implied_vol) pairs (min 5 per tenor)
    /// * `tenors` — Expiry times in years (min 2, all positive)
    /// * `forwards` — Forward prices at each tenor (all positive)
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for insufficient or invalid data,
    /// [`VolSurfError::CalibrationError`] if the optimizer fails.
    ///
    /// # References
    /// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::surface::{SsviSurface, VolSurface};
    ///
    /// let data_3m = vec![
    ///     (80.0, 0.30), (90.0, 0.25), (95.0, 0.23),
    ///     (100.0, 0.21), (105.0, 0.23), (110.0, 0.25), (120.0, 0.30),
    /// ];
    /// let data_1y = vec![
    ///     (80.0, 0.28), (90.0, 0.24), (95.0, 0.22),
    ///     (100.0, 0.20), (105.0, 0.22), (110.0, 0.24), (120.0, 0.28),
    /// ];
    /// let surface = SsviSurface::calibrate(
    ///     &[data_3m, data_1y],
    ///     &[0.25, 1.0],
    ///     &[100.0, 100.0],
    /// )?;
    /// let vol = surface.black_vol(0.5, 100.0)?;
    /// assert!(vol.0 > 0.0);
    /// # Ok::<(), volsurf::VolSurfError>(())
    /// ```
    pub fn calibrate(
        market_data: &[Vec<(f64, f64)>],
        tenors: &[f64],
        forwards: &[f64],
    ) -> error::Result<Self> {
        #[cfg(feature = "logging")]
        tracing::debug!(n_tenors = tenors.len(), "SSVI calibration started");

        const MIN_TENORS: usize = 2;
        const GRID_N: usize = 15;
        const NM_MAX_ITER: usize = 300;
        const NM_DIAMETER_TOL: f64 = 1e-8;
        const NM_FVALUE_TOL: f64 = 1e-12;

        // Input validation
        if tenors.len() < MIN_TENORS {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "at least {MIN_TENORS} tenors required for SSVI calibration, got {}",
                    tenors.len()
                ),
            });
        }
        if tenors.len() != forwards.len() || tenors.len() != market_data.len() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors, forwards, and market_data must have the same length: {}, {}, {}",
                    tenors.len(),
                    forwards.len(),
                    market_data.len()
                ),
            });
        }
        for (i, &t) in tenors.iter().enumerate() {
            if !t.is_finite() || t <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("tenors[{i}] must be positive and finite, got {t}"),
                });
            }
        }
        for (i, &f) in forwards.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("forwards[{i}] must be positive and finite, got {f}"),
                });
            }
        }

        // Stage 1: Per-tenor SVI calibration
        let n_tenors = tenors.len();
        let mut thetas = Vec::with_capacity(n_tenors);
        let mut rho_sum = 0.0;

        for (i, market_vols) in market_data.iter().enumerate() {
            let svi = crate::smile::SviSmile::calibrate(forwards[i], tenors[i], market_vols)
                .map_err(|e| VolSurfError::CalibrationError {
                    message: format!(
                        "per-tenor SVI calibration failed for tenor[{i}]={}: {e}",
                        tenors[i]
                    ),
                    model: "SSVI",
                    rms_error: None,
                })?;
            // Extract ATM total variance (theta) from calibrated SVI
            let theta = svi.variance(forwards[i])?.0;
            thetas.push(theta);
            rho_sum += svi.rho();
        }

        for (i, w) in thetas.windows(2).enumerate() {
            if w[1] <= w[0] {
                return Err(VolSurfError::CalibrationError {
                    message: format!(
                        "per-tenor SVI calibration produced non-monotone ATM variances: \
                         theta[{i}]={:.6} >= theta[{}]={:.6} (tenors {}, {})",
                        w[0],
                        i + 1,
                        w[1],
                        tenors[i],
                        tenors[i + 1]
                    ),
                    model: "SSVI",
                    rms_error: None,
                });
            }
        }

        // Average rho from per-tenor SVI fits, clamped to valid range
        let rho_global = (rho_sum / n_tenors as f64).clamp(-0.999, 0.999);
        let one_minus_rho_sq = 1.0 - rho_global * rho_global;

        // Prepare observation triples: (theta, log_moneyness, total_variance)
        let mut all_points: Vec<(f64, f64, f64)> = Vec::new();
        for (i, market_vols) in market_data.iter().enumerate() {
            for &(strike, vol) in market_vols {
                let k = (strike / forwards[i]).ln();
                let w_obs = vol * vol * tenors[i];
                all_points.push((thetas[i], k, w_obs));
            }
        }

        // Stage 2: Optimize (eta, gamma)
        let objective = |eta: f64, gamma: f64| -> f64 {
            if eta <= 0.0 || !eta.is_finite() || !gamma.is_finite() || !(0.0..=1.0).contains(&gamma)
            {
                return f64::MAX;
            }
            let mut rss = 0.0;
            for &(theta, k, w_obs) in &all_points {
                let theta_c = theta.max(1e-10);
                let phi = eta / theta_c.powf(gamma);
                let phi_k = phi * k;
                let w_pred = (theta / 2.0)
                    * (1.0
                        + rho_global * phi_k
                        + ((phi_k + rho_global).powi(2) + one_minus_rho_sq).sqrt());
                if !w_pred.is_finite() {
                    return f64::MAX;
                }
                rss += (w_pred - w_obs).powi(2);
            }
            rss
        };

        // Grid search over (eta, gamma)
        let eta_lo = 0.01_f64;
        let eta_hi = 3.0_f64;
        let gamma_lo = 0.0_f64;
        let gamma_hi = 1.0_f64;

        let mut best_eta = 0.5;
        let mut best_gamma = 0.5;
        let mut best_rss = f64::MAX;

        for ie in 0..GRID_N {
            let eta = eta_lo + (eta_hi - eta_lo) * (ie as f64) / ((GRID_N - 1) as f64);
            for ig in 0..GRID_N {
                let gamma = gamma_lo + (gamma_hi - gamma_lo) * (ig as f64) / ((GRID_N - 1) as f64);
                let rss = objective(eta, gamma);
                if rss < best_rss {
                    best_rss = rss;
                    best_eta = eta;
                    best_gamma = gamma;
                }
            }
        }

        if best_rss >= f64::MAX {
            return Err(VolSurfError::CalibrationError {
                message: "grid search found no valid starting point".into(),
                model: "SSVI",
                rms_error: None,
            });
        }

        // Nelder-Mead refinement
        let step_eta = (eta_hi - eta_lo) / (GRID_N as f64) * 0.5;
        let step_gamma = (gamma_hi - gamma_lo) / (GRID_N as f64) * 0.5;

        let nm_config = crate::optim::NelderMeadConfig {
            max_iter: NM_MAX_ITER,
            diameter_tol: NM_DIAMETER_TOL,
            fvalue_tol: NM_FVALUE_TOL,
        };
        let nm_result = crate::optim::nelder_mead_2d(
            objective, best_eta, best_gamma, step_eta, step_gamma, &nm_config,
        );

        let opt_eta = nm_result.x.max(1e-6);
        let opt_gamma = nm_result.y.clamp(0.0, 1.0);

        // Compute RMS for diagnostics
        let n_points = all_points.len();
        let rms = if n_points > 0 {
            (nm_result.fval / n_points as f64).sqrt()
        } else {
            0.0
        };

        #[cfg(feature = "logging")]
        tracing::debug!(
            rho = rho_global,
            eta = opt_eta,
            gamma = opt_gamma,
            rms_total_variance = rms,
            "SSVI calibration complete"
        );

        Self::new(
            rho_global,
            opt_eta,
            opt_gamma,
            tenors.to_vec(),
            forwards.to_vec(),
            thetas,
        )
        .map_err(|e| VolSurfError::CalibrationError {
            message: format!("calibrated SSVI params invalid: {e}"),
            model: "SSVI",
            rms_error: Some(rms),
        })
    }
}

/// Number of strikes for calendar arbitrage checks.
pub(crate) const CALENDAR_CHECK_GRID_SIZE: usize = 41;

impl VolSurface for SsviSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        validate_positive(expiry, "expiry")?;
        let var = self.black_variance(expiry, strike)?;
        Ok(Vol((var.0 / expiry).sqrt()))
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> error::Result<Variance> {
        validate_positive(expiry, "expiry")?;
        validate_positive(strike, "strike")?;
        let (theta, forward) = self.theta_and_forward_at(expiry);
        let k = (strike / forward).ln();
        let w = self.total_variance_at(theta, k);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SSVI total variance is negative: w({k}) = {w}"),
            });
        }
        Ok(Variance(w))
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        validate_positive(expiry, "expiry")?;
        let (theta, forward) = self.theta_and_forward_at(expiry);
        let slice = SsviSlice::new(forward, expiry, self.rho, self.eta, self.gamma, theta)?;
        Ok(Box::new(slice))
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        // Per-tenor butterfly reports
        let mut smile_reports = Vec::with_capacity(self.tenors.len());
        for (i, &tenor) in self.tenors.iter().enumerate() {
            let slice = SsviSlice::new(
                self.forwards[i],
                tenor,
                self.rho,
                self.eta,
                self.gamma,
                self.thetas[i],
            )?;
            smile_reports.push(slice.is_arbitrage_free()?);
        }

        // Calendar spread checks between consecutive tenors
        let mut calendar_violations = Vec::new();
        for i in 0..self.tenors.len().saturating_sub(1) {
            let f_avg = 0.5 * (self.forwards[i] + self.forwards[i + 1]);
            let grid = strike_grid(f_avg, CALENDAR_CHECK_GRID_SIZE);

            for &strike in &grid {
                let k_short = (strike / self.forwards[i]).ln();
                let k_long = (strike / self.forwards[i + 1]).ln();
                let w_short = self.total_variance_at(self.thetas[i], k_short);
                let w_long = self.total_variance_at(self.thetas[i + 1], k_long);
                if w_long < w_short - 1e-10 {
                    calendar_violations.push(CalendarViolation {
                        strike,
                        tenor_short: self.tenors[i],
                        tenor_long: self.tenors[i + 1],
                        variance_short: w_short,
                        variance_long: w_long,
                    });
                }
            }
        }

        let is_free = smile_reports.iter().all(|r| r.is_free) && calendar_violations.is_empty();

        Ok(SurfaceDiagnostics {
            smile_reports,
            calendar_violations,
            is_free,
        })
    }
}

pub(crate) fn strike_grid(forward: f64, n: usize) -> Vec<f64> {
    let ln_lo = (0.5_f64).ln();
    let ln_hi = (2.0_f64).ln();
    let step = (ln_hi - ln_lo) / (n - 1) as f64;
    (0..n)
        .map(|i| forward * (ln_lo + step * i as f64).exp())
        .collect()
}

/// A single-tenor slice through an SSVI surface.
///
/// Evaluates the SSVI total variance formula at a fixed ATM total variance θ,
/// providing a [`SmileSection`] interface for use in surface queries and
/// downstream consumers expecting a per-tenor smile.
///
/// This is a lightweight value type (6 `f64` fields) with no heap allocation.
/// Constructed by [`SsviSurface::smile_at()`] or directly via [`SsviSlice::new()`].
///
/// # Formula
///
/// At log-moneyness `k = ln(K/F)`:
///
/// ```text
/// w(k) = (θ/2) · [1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²))]
/// ```
///
/// where `φ = η / θ^γ`.
///
/// # References
/// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SsviSliceRaw", into = "SsviSliceRaw")]
pub struct SsviSlice {
    /// Forward price at this tenor.
    forward: f64,
    /// Time to expiry in years.
    expiry: f64,
    /// Global skew parameter ρ ∈ (−1, 1).
    rho: f64,
    /// Smile amplitude η > 0.
    eta: f64,
    /// Term structure decay γ ∈ [0, 1].
    gamma: f64,
    /// ATM total variance θ = σ²_ATM · T at this tenor.
    theta: f64,
}

#[derive(Serialize, Deserialize)]
struct SsviSliceRaw {
    forward: f64,
    expiry: f64,
    rho: f64,
    eta: f64,
    gamma: f64,
    theta: f64,
}

impl TryFrom<SsviSliceRaw> for SsviSlice {
    type Error = VolSurfError;
    fn try_from(raw: SsviSliceRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.forward,
            raw.expiry,
            raw.rho,
            raw.eta,
            raw.gamma,
            raw.theta,
        )
    }
}

impl From<SsviSlice> for SsviSliceRaw {
    fn from(s: SsviSlice) -> Self {
        Self {
            forward: s.forward,
            expiry: s.expiry,
            rho: s.rho,
            eta: s.eta,
            gamma: s.gamma,
            theta: s.theta,
        }
    }
}

impl SsviSlice {
    /// Create an SSVI slice at a fixed tenor.
    ///
    /// # Arguments
    /// * `forward` — Forward price, must be positive
    /// * `expiry` — Time to expiry in years, must be positive
    /// * `rho` — Skew parameter, must be in (−1, 1)
    /// * `eta` — Smile amplitude, must be positive
    /// * `gamma` — Term structure decay, must be in \[0, 1\]
    /// * `theta` — ATM total variance σ²T, must be positive
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::surface::SsviSlice;
    /// use volsurf::SmileSection;
    ///
    /// // 20% ATM vol at 1Y: theta = 0.20^2 * 1.0 = 0.04
    /// let slice = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.04)?;
    /// let vol = slice.vol(100.0)?;
    /// assert!((vol.0 - 0.20).abs() < 0.01);
    /// # Ok::<(), volsurf::VolSurfError>(())
    /// ```
    pub fn new(
        forward: f64,
        expiry: f64,
        rho: f64,
        eta: f64,
        gamma: f64,
        theta: f64,
    ) -> error::Result<Self> {
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho| must be less than 1, got {rho}"),
            });
        }
        validate_positive(eta, "eta")?;
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(VolSurfError::InvalidInput {
                message: format!("gamma must be in [0, 1], got {gamma}"),
            });
        }
        validate_positive(theta, "theta")?;
        Ok(Self {
            forward,
            expiry,
            rho,
            eta,
            gamma,
            theta,
        })
    }

    /// ATM total variance θ at this tenor.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Skew parameter ρ.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Smile amplitude η.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Term structure decay γ.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    fn phi(&self) -> f64 {
        self.eta / self.theta.powf(self.gamma)
    }

    fn total_variance(&self, k: f64) -> f64 {
        let phi = self.phi();
        let phi_k = phi * k;
        let one_minus_rho_sq = 1.0 - self.rho * self.rho;
        (self.theta / 2.0)
            * (1.0 + self.rho * phi_k + ((phi_k + self.rho).powi(2) + one_minus_rho_sq).sqrt())
    }

    // w'(k) = (θ/2) · [ρ·φ + φ·(φ·k + ρ) / R], R = √((φ·k + ρ)² + (1 − ρ²))
    fn w_prime(&self, k: f64) -> f64 {
        let phi = self.phi();
        let phi_k = phi * k;
        let one_minus_rho_sq = 1.0 - self.rho * self.rho;
        let r = ((phi_k + self.rho).powi(2) + one_minus_rho_sq).sqrt();
        (self.theta / 2.0) * (self.rho * phi + phi * (phi_k + self.rho) / r)
    }

    // w''(k) = (θ/2) · φ² · (1 − ρ²) / R³
    fn w_double_prime(&self, k: f64) -> f64 {
        let phi = self.phi();
        let phi_k = phi * k;
        let one_minus_rho_sq = 1.0 - self.rho * self.rho;
        let r = ((phi_k + self.rho).powi(2) + one_minus_rho_sq).sqrt();
        (self.theta / 2.0) * phi * phi * one_minus_rho_sq / (r * r * r)
    }

    // g(k) = (1 − k·w'/(2w))² − (w')²/4·(1/w + 1/4) + w''/2
    // g(k) ≥ 0 ⟺ no butterfly arbitrage (Gatheral & Jacquier 2014, §4)
    fn g_function(&self, k: f64) -> f64 {
        let w = self.total_variance(k);
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let wp = self.w_prime(k);
        let wpp = self.w_double_prime(k);
        let term1 = 1.0 - k * wp / (2.0 * w);
        term1 * term1 - wp * wp / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
    }
}

impl SmileSection for SsviSlice {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        validate_positive(strike, "strike")?;
        let k = (strike / self.forward).ln();
        let w = self.total_variance(k);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SSVI total variance is negative: w({k}) = {w}"),
            });
        }
        Ok(Vol((w / self.expiry).sqrt()))
    }

    fn variance(&self, strike: f64) -> error::Result<Variance> {
        validate_positive(strike, "strike")?;
        let k = (strike / self.forward).ln();
        let w = self.total_variance(k);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SSVI total variance is negative: w({k}) = {w}"),
            });
        }
        Ok(Variance(w))
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    /// Analytical risk-neutral density via the Gatheral g-function.
    ///
    /// ```text
    /// q(K) = g(k) · n(d₂) / (K · √w)
    /// ```
    /// where k = ln(K/F), d₂ = −k/√w − √w/2, and n(·) is the standard
    /// normal PDF.
    ///
    /// # Reference
    /// Breeden & Litzenberger (1978); Gatheral & Jacquier (2014), §4.
    fn density(&self, strike: f64) -> error::Result<f64> {
        validate_positive(strike, "strike")?;
        let k = (strike / self.forward).ln();
        let w = self.total_variance(k);
        if w <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SSVI total variance is non-positive at k={k}: w={w}"),
            });
        }
        let g = self.g_function(k);
        let sqrt_w = w.sqrt();
        let d2 = -k / sqrt_w - sqrt_w / 2.0;
        let n_d2 = (-d2 * d2 / 2.0).exp() / (2.0 * PI).sqrt();
        Ok(g * n_d2 / (strike * sqrt_w))
    }

    /// Check butterfly arbitrage by scanning the Gatheral g-function.
    ///
    /// Evaluates g(k) on a grid of 200 points over k ∈ \[−3, 3\].
    /// Points where g(k) < −tol are recorded as [`ButterflyViolation`]s
    /// with the actual risk-neutral density q(K) = g(k)·n(d₂)/(K·√w).
    ///
    /// # Reference
    /// Gatheral & Jacquier (2014), Theorem 4.1.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        /// Number of grid points for g-function arbitrage scan.
        const N: usize = 200;
        /// Minimum log-moneyness for arbitrage scan.
        const K_MIN: f64 = -3.0;
        /// Maximum log-moneyness for arbitrage scan.
        const K_MAX: f64 = 3.0;
        /// Tolerance for g-function negativity detection.
        const TOL: f64 = 1e-10;

        let mut violations = Vec::new();

        for i in 0..N {
            let k = K_MIN + (K_MAX - K_MIN) * (i as f64) / ((N - 1) as f64);
            let g = self.g_function(k);
            if g < -TOL {
                let strike = self.forward * k.exp();
                let d = match self.density(strike) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                violations.push(ButterflyViolation {
                    strike,
                    density: d,
                    magnitude: d.abs(),
                });
            }
        }

        if violations.is_empty() {
            Ok(ArbitrageReport::clean())
        } else {
            Ok(ArbitrageReport {
                is_free: false,
                butterfly_violations: violations,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Canonical SSVI parameters for a typical equity surface.
    fn equity_surface() -> SsviSurface {
        SsviSurface::new(
            -0.3,                             // rho
            0.5,                              // eta
            0.5,                              // gamma
            vec![0.25, 0.5, 1.0, 2.0],        // tenors
            vec![100.0, 100.0, 100.0, 100.0], // forwards
            vec![0.04, 0.08, 0.16, 0.32],     // thetas (σ_ATM=40% flat)
        )
        .unwrap()
    }

    // ========== Constructor validation ==========

    #[test]
    fn new_valid_params() {
        let s = equity_surface();
        assert_eq!(s.rho(), -0.3);
        assert_eq!(s.eta(), 0.5);
        assert_eq!(s.gamma(), 0.5);
        assert_eq!(s.tenors().len(), 4);
        assert_eq!(s.forwards().len(), 4);
        assert_eq!(s.thetas().len(), 4);
    }

    #[test]
    fn new_single_tenor() {
        let s = SsviSurface::new(0.0, 1.0, 0.5, vec![1.0], vec![100.0], vec![0.04]);
        assert!(s.is_ok());
    }

    #[test]
    fn calendar_arb_single_tenor_empty() {
        let surface = SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).unwrap();
        let violations = surface.calendar_arb_analytical();
        assert!(violations.is_empty());
    }

    #[test]
    fn new_rho_at_boundary_rejected() {
        assert!(SsviSurface::new(1.0, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-1.0, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_rho_nan_rejected() {
        assert!(SsviSurface::new(f64::NAN, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_rho_inf_rejected() {
        assert!(
            SsviSurface::new(f64::INFINITY, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err()
        );
    }

    #[test]
    fn new_eta_zero_rejected() {
        assert!(SsviSurface::new(-0.3, 0.0, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_eta_negative_rejected() {
        assert!(SsviSurface::new(-0.3, -0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_gamma_out_of_range_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, -0.1, vec![1.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 1.1, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_gamma_boundaries_accepted() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.0, vec![1.0], vec![100.0], vec![0.04]).is_ok());
        assert!(SsviSurface::new(-0.3, 0.5, 1.0, vec![1.0], vec![100.0], vec![0.04]).is_ok());
    }

    #[test]
    fn new_empty_tenors_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn new_length_mismatch_rejected() {
        // tenors vs forwards mismatch
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
        // tenors vs thetas mismatch
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn new_non_positive_tenor_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![0.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![-1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_non_positive_forward_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![0.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![-100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_non_positive_theta_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.0]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![-0.04]).is_err());
    }

    #[test]
    fn new_tenors_not_increasing_rejected() {
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![1.0, 0.5],
                vec![100.0, 100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
        // Equal tenors also rejected
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![1.0, 1.0],
                vec![100.0, 100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
    }

    #[test]
    fn new_thetas_not_increasing_rejected() {
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.08, 0.04],
            )
            .is_err()
        );
        // Equal thetas also rejected
        assert!(
            SsviSurface::new(
                -0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.04, 0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn new_nan_in_vectors_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![f64::NAN], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![f64::NAN], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![f64::NAN]).is_err());
    }

    #[test]
    fn new_inf_in_vectors_rejected() {
        assert!(
            SsviSurface::new(-0.3, 0.5, 0.5, vec![f64::INFINITY], vec![100.0], vec![0.04]).is_err()
        );
        assert!(
            SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![f64::INFINITY], vec![0.04]).is_err()
        );
        assert!(
            SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![f64::INFINITY]).is_err()
        );
    }

    // ========== SSVI formula ==========

    #[test]
    fn total_variance_atm_equals_theta() {
        // Gatheral-Jacquier (2014): w(0, θ) = θ for any θ.
        let s = equity_surface();
        for &theta in s.thetas() {
            assert_abs_diff_eq!(s.total_variance_at(theta, 0.0), theta, epsilon = 1e-14);
        }
    }

    #[test]
    fn total_variance_symmetric_when_rho_zero() {
        // When ρ = 0, the formula is symmetric in k: w(k, θ) = w(-k, θ).
        let s = SsviSurface::new(0.0, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        let theta = 0.16;
        for &k in &[-0.5, -0.2, -0.1, 0.1, 0.2, 0.5] {
            assert_abs_diff_eq!(
                s.total_variance_at(theta, k),
                s.total_variance_at(theta, -k),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn total_variance_positive_for_all_strikes() {
        let s = equity_surface();
        let theta = 0.16;
        for k in (-30..=30).map(|i| i as f64 * 0.1) {
            let w = s.total_variance_at(theta, k);
            assert!(w > 0.0, "w({k}, {theta}) = {w} must be positive");
        }
    }

    #[test]
    fn total_variance_known_value() {
        // Manual computation for rho=-0.3, eta=0.5, gamma=0.5, theta=0.16, k=0.1:
        // phi = 0.5 / 0.16^0.5 = 0.5 / 0.4 = 1.25
        // phi*k = 1.25 * 0.1 = 0.125
        // (phi*k + rho)^2 = (0.125 + (-0.3))^2 = (-0.175)^2 = 0.030625
        // 1 - rho^2 = 1 - 0.09 = 0.91
        // sqrt(0.030625 + 0.91) = sqrt(0.940625) = 0.969858...
        // w = (0.16/2) * (1 + (-0.3)*0.125 + 0.969858...)
        //   = 0.08 * (1 - 0.0375 + 0.969858...)
        //   = 0.08 * 1.932358...
        //   = 0.154589...
        let s = equity_surface();
        let w = s.total_variance_at(0.16, 0.1);
        assert_abs_diff_eq!(w, 0.154589, epsilon = 1e-5);
    }

    #[test]
    fn total_variance_skew_direction() {
        // With rho < 0 (equity-like skew), OTM puts (k < 0) have higher
        // implied variance than OTM calls (k > 0) at same |k|.
        let s = equity_surface();
        let theta = 0.16;
        let w_put = s.total_variance_at(theta, -0.2);
        let w_call = s.total_variance_at(theta, 0.2);
        assert!(w_put > w_call, "negative rho should produce put skew");
    }

    // ========== Phi function ==========

    #[test]
    fn phi_known_values() {
        let s = equity_surface();
        // phi(theta) = eta / theta^gamma = 0.5 / theta^0.5
        assert_abs_diff_eq!(s.phi(0.04), 0.5 / 0.04_f64.sqrt(), epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(0.16), 0.5 / 0.4, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn phi_gamma_zero_is_constant() {
        // gamma = 0 => phi(theta) = eta / theta^0 = eta for all theta.
        let s = SsviSurface::new(-0.3, 0.5, 0.0, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        assert_abs_diff_eq!(s.phi(0.04), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(10.0), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn phi_gamma_one_is_inverse() {
        // gamma = 1 => phi(theta) = eta / theta.
        let s = SsviSurface::new(-0.3, 0.5, 1.0, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        assert_abs_diff_eq!(s.phi(0.04), 0.5 / 0.04, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
    }

    // ========== Theta and forward interpolation ==========

    #[test]
    fn theta_forward_exact_match() {
        let s = equity_surface();
        // Exact tenor should return stored values.
        let (theta, fwd) = s.theta_and_forward_at(0.5);
        assert_abs_diff_eq!(theta, 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_between_tenors() {
        let s = equity_surface();
        // Midpoint between T=0.5 (theta=0.08) and T=1.0 (theta=0.16).
        let (theta, fwd) = s.theta_and_forward_at(0.75);
        // alpha = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // theta = 0.5 * 0.08 + 0.5 * 0.16 = 0.12
        // Log-linear with equal forwards: exp(ln(100)) ≈ 100 (within FP roundtrip)
        assert_abs_diff_eq!(theta, 0.12, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-12);
    }

    #[test]
    fn theta_forward_before_first_tenor() {
        let s = equity_surface();
        // T=0.1 < T_0=0.25. Flat vol: theta(0.1) = 0.04 * 0.1 / 0.25 = 0.016
        let (theta, fwd) = s.theta_and_forward_at(0.1);
        assert_abs_diff_eq!(theta, 0.016, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_after_last_tenor() {
        let s = equity_surface();
        // T=3.0 > T_n=2.0. Flat vol: theta(3.0) = 0.32 * 3.0 / 2.0 = 0.48
        let (theta, fwd) = s.theta_and_forward_at(3.0);
        assert_abs_diff_eq!(theta, 0.48, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_interpolation_with_different_forwards() {
        let s = SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 105.0],
            vec![0.08, 0.16],
        )
        .unwrap();
        // At T=0.75: alpha = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // Log-linear: F = exp(0.5*ln(100) + 0.5*ln(105)) = sqrt(100*105)
        let (theta, fwd) = s.theta_and_forward_at(0.75);
        assert_abs_diff_eq!(theta, 0.12, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, (100.0_f64 * 105.0).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn theta_forward_log_linear_geometric_mean() {
        let s = SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![1.0, 2.0],
            vec![100.0, 400.0],
            vec![0.04, 0.08],
        )
        .unwrap();
        // Midpoint: alpha = 0.5, log-linear → F = sqrt(100*400) = 200 (not 250)
        let (_, fwd) = s.theta_and_forward_at(1.5);
        assert_abs_diff_eq!(fwd, 200.0, epsilon = 1e-10);
    }

    // ========== Serde round-trip ==========

    #[test]
    fn serde_round_trip() {
        let s = equity_surface();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SsviSurface = serde_json::from_str(&json).unwrap();
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.eta(), s2.eta());
        assert_eq!(s.gamma(), s2.gamma());
        assert_eq!(s.tenors(), s2.tenors());
        assert_eq!(s.forwards(), s2.forwards());
        assert_eq!(s.thetas(), s2.thetas());
    }

    #[test]
    fn serde_no_one_minus_rho_sq_in_json() {
        let s = equity_surface();
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("one_minus_rho_sq"));
    }

    #[test]
    fn serde_rejects_invalid_rho() {
        let json = r#"{"rho":1.5,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_null_rho() {
        let json = r#"{"rho":null,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_plus_one() {
        let json = r#"{"rho":1.0,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_minus_one() {
        let json = r#"{"rho":-1.0,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_eta() {
        let json = r#"{"rho":-0.3,"eta":-0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_gamma_out_of_range() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":1.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_non_monotone_thetas() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[0.5,1.0],"forwards":[100.0,100.0],"thetas":[0.16,0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_mismatched_lengths() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[0.5,1.0],"forwards":[100.0],"thetas":[0.04,0.08]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_empty_tenors() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[],"forwards":[],"thetas":[]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_forward() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[-100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_theta() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.0]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_eta() {
        let json = r#"{"rho":-0.3,"eta":0.0,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_gamma() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":-0.1,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_non_increasing_tenors() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[1.0,0.5],"forwards":[100.0,100.0],"thetas":[0.04,0.08]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_tenor() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[0.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_inf_tenor() {
        let json = r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[1e999],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_forward() {
        let json =
            r#"{"rho":-0.3,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[0.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<SsviSurface>(json).is_err());
    }

    #[test]
    fn serde_error_contains_validation_message() {
        let json = r#"{"rho":1.5,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        let err = serde_json::from_str::<SsviSurface>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rho"),
            "serde error should contain domain message, got: {msg}"
        );
    }

    // ========== Send + Sync ==========

    #[test]
    fn ssvi_surface_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SsviSurface>();
    }

    // ========== Calibration tests (T09) ==========

    /// Generate synthetic SSVI market data by sampling a known surface.
    fn synthetic_ssvi_data(
        surface: &SsviSurface,
        tenors: &[f64],
        strikes_per_tenor: &[Vec<f64>],
    ) -> Vec<Vec<(f64, f64)>> {
        tenors
            .iter()
            .zip(strikes_per_tenor)
            .map(|(&t, strikes)| {
                strikes
                    .iter()
                    .map(|&k| (k, surface.black_vol(t, k).unwrap().0))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn calibrate_round_trip_equity() {
        // Create a known SSVI surface, sample it, calibrate, compare.
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0, 100.0, 100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();

        let mut total_rss = 0.0;
        let mut n_points = 0;
        for (i, &t) in tenors.iter().enumerate() {
            for &(strike, vol_obs) in &market_data[i] {
                let vol_fit = calibrated.black_vol(t, strike).unwrap().0;
                total_rss += (vol_fit - vol_obs).powi(2);
                n_points += 1;
            }
        }
        let rms = (total_rss / n_points as f64).sqrt();
        assert!(rms < 0.005, "round-trip RMS {rms} should be < 0.005");
    }

    #[test]
    fn calibrate_round_trip_2_tenors() {
        // Minimum: 2 tenors
        let original = SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.08, 0.16],
        )
        .unwrap();
        let tenors = vec![0.5, 1.0];
        let forwards = vec![100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..10).map(|i| 75.0 + 5.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();

        let mut total_rss = 0.0;
        let mut n_points = 0;
        for (i, &t) in tenors.iter().enumerate() {
            for &(strike, vol_obs) in &market_data[i] {
                let vol_fit = calibrated.black_vol(t, strike).unwrap().0;
                total_rss += (vol_fit - vol_obs).powi(2);
                n_points += 1;
            }
        }
        let rms = (total_rss / n_points as f64).sqrt();
        assert!(
            rms < 0.005,
            "2-tenor round-trip RMS {rms} should be < 0.005"
        );
    }

    #[test]
    fn calibrate_round_trip_varying_forwards() {
        // Different forward prices per tenor (term structure).
        let original = SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![0.25, 0.5, 1.0],
            vec![100.0, 102.0, 105.0],
            vec![0.04, 0.08, 0.16],
        )
        .unwrap();
        let tenors = vec![0.25, 0.5, 1.0];
        let forwards = vec![100.0, 102.0, 105.0];
        let strikes: Vec<Vec<f64>> = forwards
            .iter()
            .map(|&f| (0..10).map(|i| f * 0.8 + f * 0.04 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();

        let mut total_rss = 0.0;
        let mut n_points = 0;
        for (i, &t) in tenors.iter().enumerate() {
            for &(strike, vol_obs) in &market_data[i] {
                let vol_fit = calibrated.black_vol(t, strike).unwrap().0;
                total_rss += (vol_fit - vol_obs).powi(2);
                n_points += 1;
            }
        }
        let rms = (total_rss / n_points as f64).sqrt();
        assert!(
            rms < 0.005,
            "varying-forwards round-trip RMS {rms} should be < 0.005"
        );
    }

    #[test]
    fn calibrate_rejects_single_tenor() {
        let data = vec![vec![
            (90.0, 0.2),
            (95.0, 0.2),
            (100.0, 0.2),
            (105.0, 0.2),
            (110.0, 0.2),
        ]];
        let result = SsviSurface::calibrate(&data, &[1.0], &[100.0]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_empty_data() {
        let result = SsviSurface::calibrate(&[], &[], &[]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_length_mismatch() {
        let data = vec![
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
        ];
        // tenors has 2 but forwards has 1
        let result = SsviSurface::calibrate(&data, &[0.5, 1.0], &[100.0]);
        assert!(result.is_err());
    }

    #[test]
    fn calibrate_rejects_negative_forward() {
        let data = vec![
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
        ];
        let result = SsviSurface::calibrate(&data, &[0.5, 1.0], &[-100.0, 100.0]);
        assert!(result.is_err());
    }

    #[test]
    fn calibrate_rejects_zero_tenor() {
        let data = vec![
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
            vec![
                (90.0, 0.2),
                (95.0, 0.2),
                (100.0, 0.2),
                (105.0, 0.2),
                (110.0, 0.2),
            ],
        ];
        let result = SsviSurface::calibrate(&data, &[0.0, 1.0], &[100.0, 100.0]);
        assert!(result.is_err());
    }

    #[test]
    fn calibrate_rejects_non_monotone_atm_variances() {
        let short_tenor_data: Vec<(f64, f64)> = vec![
            (80.0, 0.45),
            (90.0, 0.42),
            (95.0, 0.41),
            (100.0, 0.40),
            (105.0, 0.41),
            (110.0, 0.42),
            (120.0, 0.45),
        ];
        let long_tenor_data: Vec<(f64, f64)> = vec![
            (80.0, 0.18),
            (90.0, 0.16),
            (95.0, 0.155),
            (100.0, 0.15),
            (105.0, 0.155),
            (110.0, 0.16),
            (120.0, 0.18),
        ];
        let market_data = vec![short_tenor_data, long_tenor_data];
        let tenors = vec![0.25, 1.0];
        let forwards = vec![100.0, 100.0];
        let result = SsviSurface::calibrate(&market_data, &tenors, &forwards);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-monotone"),
            "expected non-monotone error, got: {err}"
        );
    }

    #[test]
    fn calibration_error_format_ssvi_global_grid() {
        let err = VolSurfError::CalibrationError {
            message: "grid search found no valid starting point".into(),
            model: "SSVI",
            rms_error: None,
        };
        assert!(err.to_string().contains("grid search"));
    }

    #[test]
    fn calibrate_params_in_valid_range() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0, 100.0, 100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        assert!(calibrated.rho().abs() < 1.0, "rho out of range");
        assert!(calibrated.eta() > 0.0, "eta must be positive");
        assert!(
            (0.0..=1.0).contains(&calibrated.gamma()),
            "gamma out of [0,1]"
        );
        assert_eq!(calibrated.tenors().len(), 4);
        assert_eq!(calibrated.forwards().len(), 4);
        assert_eq!(calibrated.thetas().len(), 4);
    }

    #[test]
    fn calibrate_diagnostics_clean() {
        // Calibrated surface from conservative data should have clean diagnostics.
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0, 100.0, 100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let diag = calibrated.diagnostics().unwrap();
        assert!(
            diag.is_free,
            "calibrated SSVI from conservative input should be arb-free"
        );
    }

    #[test]
    fn calibrate_thetas_strictly_increasing() {
        // Verify the extracted thetas are monotonically increasing.
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0, 100.0, 100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_ssvi_data(&original, &tenors, &strikes);

        let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let thetas = calibrated.thetas();
        for w in thetas.windows(2) {
            assert!(
                w[1] > w[0],
                "thetas must be strictly increasing: {} <= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn calibrate_rejects_non_monotone_thetas() {
        // Earnings-like scenario: short tenor has inflated vol → theta[0] > theta[1].
        // ATM vol 50% at T=0.25 gives theta ~= 0.0625, but ATM vol 20% at T=0.5 gives theta ~= 0.02.
        let make_smile = |fwd: f64, atm_vol: f64| -> Vec<(f64, f64)> {
            let strikes: Vec<f64> = (0..10).map(|i| fwd * (0.85 + 0.03 * i as f64)).collect();
            strikes
                .iter()
                .map(|&k| {
                    let m = ((k / fwd).ln()).abs();
                    (k, atm_vol + 0.3 * m)
                })
                .collect()
        };
        let data = vec![
            make_smile(100.0, 0.50), // short tenor: very high vol (earnings)
            make_smile(100.0, 0.20), // long tenor: normal vol
        ];
        let result = SsviSurface::calibrate(&data, &[0.25, 0.50], &[100.0, 100.0]);
        let err = result.unwrap_err();
        assert!(matches!(err, VolSurfError::CalibrationError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("non-monotone"),
            "error should mention non-monotone: {msg}"
        );
    }

    #[test]
    fn calibrate_rejects_too_few_points_per_tenor() {
        // SVI calibration requires 5 points; providing only 3 per tenor should fail.
        let data = vec![
            vec![(90.0, 0.3), (100.0, 0.25), (110.0, 0.3)],
            vec![(90.0, 0.3), (100.0, 0.25), (110.0, 0.3)],
        ];
        let result = SsviSurface::calibrate(&data, &[0.5, 1.0], &[100.0, 100.0]);
        assert!(result.is_err());
    }

    // ========== Error type checks ==========

    #[test]
    fn validation_errors_are_invalid_input() {
        let err = SsviSurface::new(1.5, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).unwrap_err();
        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
    }

    // ========== VolSurface trait impl (T08) ==========

    #[test]
    fn black_vol_at_stored_tenor() {
        let s = equity_surface();
        // At T=1.0, K=100 (ATM), theta=0.16.
        // vol = sqrt(theta / T) = sqrt(0.16) = 0.4
        let vol = s.black_vol(1.0, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn black_variance_at_stored_tenor() {
        let s = equity_surface();
        // At T=1.0, K=100 (ATM), variance = theta = 0.16
        let var = s.black_variance(1.0, 100.0).unwrap();
        assert_abs_diff_eq!(var.0, 0.16, epsilon = 1e-14);
    }

    #[test]
    fn black_vol_variance_consistency() {
        // vol^2 * T == variance at multiple query points.
        let s = equity_surface();
        for &expiry in &[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] {
            for &strike in &[80.0, 100.0, 120.0] {
                let vol = s.black_vol(expiry, strike).unwrap();
                let var = s.black_variance(expiry, strike).unwrap();
                assert_abs_diff_eq!(vol.0 * vol.0 * expiry, var.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn black_vol_between_tenors() {
        let s = equity_surface();
        // At T=0.75 (between T=0.5 and T=1.0), the interpolated theta = 0.12.
        // ATM vol = sqrt(0.12 / 0.75) = sqrt(0.16) = 0.4
        let vol = s.black_vol(0.75, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn black_vol_before_first_tenor() {
        let s = equity_surface();
        // T=0.1 < T_0=0.25. Flat vol: theta(0.1) = 0.04 * 0.1/0.25 = 0.016.
        // ATM vol = sqrt(0.016 / 0.1) = sqrt(0.16) = 0.4
        let vol = s.black_vol(0.1, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn black_vol_after_last_tenor() {
        let s = equity_surface();
        // T=3.0 > T_n=2.0. Flat vol: theta(3.0) = 0.32 * 3.0/2.0 = 0.48.
        // ATM vol = sqrt(0.48 / 3.0) = sqrt(0.16) = 0.4
        let vol = s.black_vol(3.0, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn black_vol_rejects_invalid_inputs() {
        let s = equity_surface();
        assert!(s.black_vol(0.0, 100.0).is_err());
        assert!(s.black_vol(-1.0, 100.0).is_err());
        assert!(s.black_vol(1.0, 0.0).is_err());
        assert!(s.black_vol(1.0, -100.0).is_err());
    }

    #[test]
    fn smile_at_returns_working_section() {
        let s = equity_surface();
        let smile = s.smile_at(1.0).unwrap();
        assert_eq!(smile.forward(), 100.0);
        assert_eq!(smile.expiry(), 1.0);
        // ATM vol via smile should match surface query
        let smile_vol = smile.vol(100.0).unwrap();
        let surface_vol = s.black_vol(1.0, 100.0).unwrap();
        assert_abs_diff_eq!(smile_vol.0, surface_vol.0, epsilon = 1e-14);
    }

    #[test]
    fn smile_at_interpolated_tenor() {
        let s = equity_surface();
        let smile = s.smile_at(0.75).unwrap();
        // ATM variance from smile should match surface
        let smile_var = smile.variance(100.0).unwrap();
        let surface_var = s.black_variance(0.75, 100.0).unwrap();
        assert_abs_diff_eq!(smile_var.0, surface_var.0, epsilon = 1e-14);
    }

    #[test]
    fn smile_at_agrees_with_surface_otm() {
        // OTM strikes via smile should match direct surface query.
        let s = equity_surface();
        let smile = s.smile_at(1.0).unwrap();
        for &strike in &[80.0, 90.0, 110.0, 120.0] {
            let smile_vol = smile.vol(strike).unwrap();
            let surface_vol = s.black_vol(1.0, strike).unwrap();
            assert_abs_diff_eq!(smile_vol.0, surface_vol.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn diagnostics_clean_for_conservative_params() {
        // eta * (1 + |rho|) = 0.5 * 1.3 = 0.65 < 2, and thetas strictly increasing.
        let s = equity_surface();
        let diag = s.diagnostics().unwrap();
        assert!(diag.is_free, "conservative SSVI should be arb-free");
        assert!(diag.calendar_violations.is_empty());
        assert_eq!(diag.smile_reports.len(), 4);
        assert!(diag.smile_reports.iter().all(|r| r.is_free));
    }

    #[test]
    fn diagnostics_detects_butterfly_violations() {
        // Extreme params: eta * (1 + |rho|) = 3 * 1.95 = 5.85 >> 2.
        let s = SsviSurface::new(
            -0.95,
            3.0,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.08, 0.16],
        )
        .unwrap();
        let diag = s.diagnostics().unwrap();
        assert!(!diag.is_free);
        // At least one tenor should have butterfly violations
        assert!(diag.smile_reports.iter().any(|r| !r.is_free));
    }

    #[test]
    fn diagnostics_single_tenor_no_calendar() {
        let s = SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        let diag = s.diagnostics().unwrap();
        assert!(diag.calendar_violations.is_empty());
        assert_eq!(diag.smile_reports.len(), 1);
    }

    // ========== Calendar arbitrage tests (T12) ==========

    #[test]
    fn dw_dtheta_at_atm_equals_one() {
        // At k=0: w(0,θ) = θ, so ∂w/∂θ = 1 exactly.
        let s = equity_surface();
        for &theta in s.thetas() {
            let deriv = s.dw_dtheta(theta, 0.0);
            assert_abs_diff_eq!(deriv, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn dw_dtheta_non_negative_conservative() {
        // Scan ∂w/∂θ ≥ 0 across a wide grid for conservative parameters.
        // Gatheral-Jacquier (2014): power-law SSVI with γ ∈ [0,1] guarantees
        // ∂w/∂θ ≥ 0 for all (θ, k).
        let s = equity_surface();
        for &theta in s.thetas() {
            for i in -40..=40 {
                let k = i as f64 * 0.1;
                let deriv = s.dw_dtheta(theta, k);
                assert!(
                    deriv >= -1e-14,
                    "dw/dθ({theta}, {k}) = {deriv} should be non-negative"
                );
            }
        }
    }

    #[test]
    fn dw_dtheta_non_negative_extreme_params() {
        // Extreme rho and eta: ∂w/∂θ ≥ 0 must still hold.
        let s = SsviSurface::new(
            -0.95,
            3.0,
            0.99,
            vec![0.25, 0.5, 1.0],
            vec![100.0, 100.0, 100.0],
            vec![0.01, 0.05, 0.20],
        )
        .unwrap();
        for &theta in s.thetas() {
            for i in -30..=30 {
                let k = i as f64 * 0.1;
                let deriv = s.dw_dtheta(theta, k);
                assert!(
                    deriv >= -1e-12,
                    "dw/dθ({theta}, {k}) = {deriv} should be non-negative even for extreme params"
                );
            }
        }
    }

    #[test]
    fn dw_dtheta_gamma_zero_equals_w_over_theta() {
        // γ = 0 ⟹ φ is constant in θ ⟹ ∂w/∂θ = w/θ (no smile decay).
        let s = SsviSurface::new(-0.3, 0.5, 0.0, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        let theta = 0.16;
        for &k in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let deriv = s.dw_dtheta(theta, k);
            let w = s.total_variance_at(theta, k);
            assert_abs_diff_eq!(deriv, w / theta, epsilon = 1e-14);
        }
    }

    #[test]
    fn calendar_arb_analytical_clean_for_valid_surface() {
        // Valid SSVI surface: analytical check returns no violations.
        let s = equity_surface();
        let violations = s.calendar_arb_analytical();
        assert!(
            violations.is_empty(),
            "valid SSVI should be analytically calendar-arb-free, got {} violations",
            violations.len()
        );
    }

    #[test]
    fn calendar_arb_analytical_and_numerical_agree() {
        // Both numerical (diagnostics) and analytical checks agree:
        // no calendar violations for a valid SSVI surface.
        let s = equity_surface();
        let diag = s.diagnostics().unwrap();
        let analytical = s.calendar_arb_analytical();
        assert!(
            diag.calendar_violations.is_empty(),
            "numerical check should find no violations"
        );
        assert!(
            analytical.is_empty(),
            "analytical check should find no violations"
        );
    }

    #[test]
    fn calendar_violation_detected_for_inverted_ssvi_slices() {
        // Create SSVI slices with inverted thetas (short tenor has HIGHER
        // ATM total variance) and verify calendar violations are detected
        // when composed into a PiecewiseSurface.
        use crate::surface::PiecewiseSurface;

        let slice_short = SsviSlice::new(100.0, 0.5, -0.3, 0.5, 0.5, 0.20).unwrap();
        let slice_long = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.08).unwrap();

        let surface = PiecewiseSurface::new(
            vec![0.5, 1.0],
            vec![Box::new(slice_short), Box::new(slice_long)],
        )
        .unwrap();

        let diag = surface.diagnostics().unwrap();
        assert!(
            !diag.is_free,
            "inverted SSVI slices should have calendar violations"
        );
        assert!(
            !diag.calendar_violations.is_empty(),
            "should detect calendar violations for inverted thetas"
        );
        // Verify violation direction: short tenor variance > long tenor variance
        for v in &diag.calendar_violations {
            assert!(
                v.variance_short > v.variance_long,
                "short tenor variance ({}) should exceed long ({})",
                v.variance_short,
                v.variance_long
            );
        }
    }

    #[test]
    fn calendar_arb_analytical_single_tenor() {
        // Single tenor: no consecutive pairs, so no calendar violations.
        let s = SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.16]).unwrap();
        assert!(s.calendar_arb_analytical().is_empty());
    }

    #[test]
    fn calendar_arb_clean_for_barely_increasing_thetas() {
        // Thetas barely above each other — both checks should still pass.
        let s = SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.04, 0.04001],
        )
        .unwrap();
        let diag = s.diagnostics().unwrap();
        assert!(
            diag.calendar_violations.is_empty(),
            "barely increasing thetas should still pass numerical calendar checks"
        );
        let analytical = s.calendar_arb_analytical();
        assert!(
            analytical.is_empty(),
            "barely increasing thetas should still pass analytical calendar checks"
        );
    }

    // ================================================================
    // SsviSlice tests
    // ================================================================

    /// Canonical SSVI slice for a typical equity tenor.
    /// rho=-0.3, eta=0.5, gamma=0.5, F=100, T=1.0, theta=0.16
    fn equity_slice() -> SsviSlice {
        SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap()
    }

    #[test]
    fn slice_new_valid_params() {
        let s = equity_slice();
        assert_eq!(s.forward(), 100.0);
        assert_eq!(s.expiry(), 1.0);
        assert_eq!(s.theta(), 0.16);
    }

    #[test]
    fn slice_new_rejects_invalid_params() {
        // Bad forward
        assert!(SsviSlice::new(0.0, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        // Bad expiry
        assert!(SsviSlice::new(100.0, -1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        // Bad rho
        assert!(SsviSlice::new(100.0, 1.0, 1.0, 0.5, 0.5, 0.16).is_err());
        // Bad eta
        assert!(SsviSlice::new(100.0, 1.0, -0.3, 0.0, 0.5, 0.16).is_err());
        // Bad gamma
        assert!(SsviSlice::new(100.0, 1.0, -0.3, 0.5, 1.5, 0.16).is_err());
        // Bad theta
        assert!(SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.0).is_err());
    }

    #[test]
    fn slice_vol_atm_equals_sqrt_theta_over_t() {
        // ATM: strike = forward. k = 0. w(0) = theta.
        // vol = sqrt(theta / T) = sqrt(0.16 / 1.0) = 0.4
        let s = equity_slice();
        let vol = s.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn slice_variance_atm_equals_theta() {
        // At ATM: variance = theta exactly.
        let s = equity_slice();
        let var = s.variance(100.0).unwrap();
        assert_abs_diff_eq!(var.0, 0.16, epsilon = 1e-14);
    }

    #[test]
    fn slice_vol_variance_consistency() {
        // variance(K) == vol(K)^2 * T for any strike.
        let s = equity_slice();
        for &strike in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = s.vol(strike).unwrap();
            let var = s.variance(strike).unwrap();
            assert_abs_diff_eq!(var.0, vol.0 * vol.0 * s.expiry(), epsilon = 1e-14);
        }
    }

    #[test]
    fn slice_vol_matches_surface_formula() {
        // SsviSlice vol at (T=1, K) should match SsviSurface.total_variance_at(theta, k)
        // for the same theta.
        let surface = equity_surface();
        let slice = equity_slice();
        let theta = 0.16;
        for &strike in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let k = (strike / 100.0_f64).ln();
            let w_surface = surface.total_variance_at(theta, k);
            let w_slice = slice.variance(strike).unwrap().0;
            assert_abs_diff_eq!(w_surface, w_slice, epsilon = 1e-14);
        }
    }

    #[test]
    fn slice_vol_rejects_non_positive_strikes() {
        let s = equity_slice();
        assert!(s.vol(0.0).is_err());
        assert!(s.vol(-100.0).is_err());
        assert!(s.variance(0.0).is_err());
        assert!(s.variance(-100.0).is_err());
    }

    #[test]
    fn slice_skew_direction() {
        // rho < 0: OTM puts have higher vol than OTM calls.
        let s = equity_slice();
        let vol_put = s.vol(80.0).unwrap().0;
        let vol_call = s.vol(120.0).unwrap().0;
        assert!(vol_put > vol_call, "negative rho should produce put skew");
    }

    #[test]
    fn slice_density_positive_near_atm() {
        // density should be positive near ATM for well-behaved params.
        let s = equity_slice();
        for &strike in &[90.0, 95.0, 100.0, 105.0, 110.0] {
            let d = s.density(strike).unwrap();
            assert!(d > 0.0, "density({strike}) = {d} should be positive");
        }
    }

    #[test]
    fn slice_arb_free_conservative_params() {
        // eta * (1 + |rho|) = 0.5 * (1 + 0.3) = 0.65 < 2 => should be clean.
        let s = equity_slice();
        let report = s.is_arbitrage_free().unwrap();
        assert!(
            report.is_free,
            "conservative SSVI params should be arb-free"
        );
        assert!(report.butterfly_violations.is_empty());
    }

    #[test]
    fn slice_arb_free_detects_violations_extreme_params() {
        // eta * (1 + |rho|) = 3.0 * (1 + 0.95) = 5.85 >> 2 => likely violations.
        let s = SsviSlice::new(100.0, 1.0, -0.95, 3.0, 0.5, 0.16).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(
            !report.is_free,
            "extreme SSVI params should detect butterfly violations"
        );
        assert!(!report.butterfly_violations.is_empty());
    }

    #[test]
    fn slice_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SsviSlice>();
    }

    #[test]
    fn slice_serde_round_trip() {
        let s = equity_slice();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SsviSlice = serde_json::from_str(&json).unwrap();
        assert_eq!(s.forward(), s2.forward());
        assert_eq!(s.expiry(), s2.expiry());
        assert_eq!(s.theta(), s2.theta());
        // Verify vol agrees after deserialization
        assert_abs_diff_eq!(
            s.vol(100.0).unwrap().0,
            s2.vol(100.0).unwrap().0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn slice_serde_rejects_invalid_rho() {
        let json = r#"{"forward":100.0,"expiry":1.0,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_negative_forward() {
        let json =
            r#"{"forward":-100.0,"expiry":1.0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_zero_expiry() {
        let json =
            r#"{"forward":100.0,"expiry":0.0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_negative_eta() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"rho":-0.3,"eta":-0.1,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_gamma_out_of_range() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"rho":-0.3,"eta":0.5,"gamma":2.0,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_zero_theta() {
        let json = r#"{"forward":100.0,"expiry":1.0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.0}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_inf_forward() {
        let json =
            r#"{"forward":1e999,"expiry":1.0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_negative_expiry() {
        let json =
            r#"{"forward":100.0,"expiry":-1.0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_negative_gamma() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"rho":-0.3,"eta":0.5,"gamma":-0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_rejects_zero_eta() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"rho":-0.3,"eta":0.0,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<SsviSlice>(json).is_err());
    }

    #[test]
    fn slice_serde_error_contains_validation_message() {
        let json = r#"{"forward":100.0,"expiry":1.0,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        let err = serde_json::from_str::<SsviSlice>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rho"),
            "serde error should contain domain message, got: {msg}"
        );
    }

    #[test]
    fn slice_w_prime_finite_and_sign() {
        // For rho < 0, w'(k) should be negative for positive k (skew).
        let s = equity_slice();
        let wp = s.w_prime(0.2);
        assert!(wp.is_finite());
        assert!(wp < 0.0, "w'(0.2) = {wp} should be negative for rho < 0");
        // w'(k) for large negative k should be positive
        let wp_left = s.w_prime(-1.0);
        assert!(wp_left < wp, "smile should be steeper on put side");
    }

    #[test]
    fn slice_w_double_prime_always_positive() {
        // w''(k) = (theta/2) * phi^2 * (1-rho^2) / R^3 > 0 always.
        let s = equity_slice();
        for k in (-30..=30).map(|i| i as f64 * 0.1) {
            let wpp = s.w_double_prime(k);
            assert!(wpp > 0.0, "w''({k}) = {wpp} must be positive");
        }
    }
}
