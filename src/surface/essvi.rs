//! eSSVI (Extended SSVI) surface and slice.
//!
//! eSSVI extends Gatheral-Jacquier's SSVI by making the correlation parameter ρ
//! depend on maturity (through the ATM total variance θ). At the slice level the
//! formula is identical to SSVI — the difference is purely in how ρ is chosen.
//!
//! [`EssviSlice`] is a single-tenor slice implementing [`SmileSection`].
//! [`EssviSurface`] is the multi-tenor surface implementing [`VolSurface`].
//!
//! # References
//! - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::smile::arbitrage::ArbitrageReport;
use crate::surface::VolSurface;
use crate::surface::arbitrage::{CalendarViolation, SurfaceDiagnostics};
use crate::surface::ssvi::{CALENDAR_CHECK_GRID_SIZE, SsviSlice, strike_grid};
use crate::types::{Variance, Vol};
use crate::validate::validate_positive;

/// A structural calendar no-arb violation (Thm 4.1, Eq 4.10).
///
/// Returned by [`EssviSurface::calendar_check_structural()`] when the
/// continuous-time condition `(δ + ρ·γ)² ≤ γ²` fails at a stored tenor.
#[derive(Debug, Clone)]
pub struct StructuralViolation {
    /// Tenor (in years) where the violation was detected.
    pub tenor: f64,
    /// ATM total variance θ at that tenor.
    pub theta: f64,
    /// Left-hand side |δ + ρ·γ| (should be ≤ `condition_rhs`).
    pub condition_lhs: f64,
    /// Right-hand side γ (= 1 − γ_param).
    pub condition_rhs: f64,
}

/// A single-tenor slice through an eSSVI surface.
///
/// Evaluates the eSSVI total variance formula at a fixed ATM total variance θ,
/// providing a [`SmileSection`] interface for surface queries and downstream
/// consumers expecting a per-tenor smile.
///
/// At the slice level eSSVI and SSVI share the same formula — this is a thin
/// newtype over [`SsviSlice`]. The per-tenor ρ(θ) is evaluated by
/// [`EssviSurface`] at construction time and baked into the slice.
///
/// # Formula
///
/// At log-moneyness `k = ln(K/F)`:
///
/// ```text
/// w(k, θ) = (θ/2) · [1 + ρ(θ)·φ(θ)·k + √((φ(θ)·k + ρ(θ))² + (1 − ρ(θ)²))]
/// ```
///
/// where `φ(θ) = η / θ^γ` and `ρ(θ)` is a parametric family managed by
/// [`EssviSurface`].
///
/// # References
/// - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019), Eq. 2.2
/// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "EssviSliceRaw", into = "EssviSliceRaw")]
pub struct EssviSlice(SsviSlice);

#[derive(Serialize, Deserialize)]
struct EssviSliceRaw {
    forward: f64,
    expiry: f64,
    rho: f64,
    eta: f64,
    gamma: f64,
    theta: f64,
}

impl TryFrom<EssviSliceRaw> for EssviSlice {
    type Error = VolSurfError;
    fn try_from(raw: EssviSliceRaw) -> Result<Self, Self::Error> {
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

impl From<EssviSlice> for EssviSliceRaw {
    fn from(s: EssviSlice) -> Self {
        Self {
            forward: s.0.forward(),
            expiry: s.0.expiry(),
            rho: s.rho(),
            eta: s.eta(),
            gamma: s.gamma(),
            theta: s.0.theta(),
        }
    }
}

impl EssviSlice {
    /// Create an eSSVI slice at a fixed tenor.
    ///
    /// The `rho` parameter is the maturity-dependent correlation ρ(θ),
    /// typically computed by [`EssviSurface`] from its parametric family.
    /// All validation is delegated to [`SsviSlice::new()`].
    ///
    /// # Arguments
    /// * `forward` — Forward price, must be positive
    /// * `expiry` — Time to expiry in years, must be positive
    /// * `rho` — Maturity-dependent skew ρ(θ), must be in (−1, 1)
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
    /// use volsurf::surface::EssviSlice;
    /// use volsurf::SmileSection;
    ///
    /// let slice = EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.04)?;
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
        SsviSlice::new(forward, expiry, rho, eta, gamma, theta).map(EssviSlice)
    }

    /// ATM total variance θ at this tenor.
    pub fn theta(&self) -> f64 {
        self.0.theta()
    }

    /// Skew parameter ρ(θ) at this tenor.
    pub fn rho(&self) -> f64 {
        self.0.rho()
    }

    /// Smile amplitude η.
    pub fn eta(&self) -> f64 {
        self.0.eta()
    }

    /// Term structure decay γ.
    pub fn gamma(&self) -> f64 {
        self.0.gamma()
    }
}

impl SmileSection for EssviSlice {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        self.0.vol(strike)
    }

    fn variance(&self, strike: f64) -> error::Result<Variance> {
        self.0.variance(strike)
    }

    fn forward(&self) -> f64 {
        self.0.forward()
    }

    fn expiry(&self) -> f64 {
        self.0.expiry()
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        self.0.is_arbitrage_free()
    }
}

/// Extended SSVI surface with maturity-dependent correlation ρ(θ).
///
/// eSSVI extends Gatheral-Jacquier's SSVI by replacing the global correlation
/// parameter ρ with a parametric family ρ(θ) that transitions smoothly from
/// an initial skew ρ₀ (short maturities) to a long-term skew ρₘ.
///
/// # Parametric family (Hendriks-Martini 2019, Eq. 3.1)
///
/// ```text
/// ρ(θ) = ρ₀ + (ρₘ − ρ₀) · (θ / θ_max)^a
/// ```
///
/// # Construction
///
/// ```
/// use volsurf::surface::EssviSurface;
///
/// let surface = EssviSurface::new(
///     -0.4,                              // rho_0: short-term skew
///     -0.2,                              // rho_m: long-term skew
///     0.5,                               // a: shape parameter
///     0.5,                               // eta: smile amplitude
///     0.5,                               // gamma: term decay
///     vec![0.25, 0.5, 1.0],             // tenors (years)
///     vec![100.0, 100.0, 100.0],        // forward prices
///     vec![0.04, 0.08, 0.16],           // ATM total variances (theta)
/// ).unwrap();
/// ```
///
/// # References
/// - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)
/// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "EssviSurfaceRaw", into = "EssviSurfaceRaw")]
pub struct EssviSurface {
    rho_0: f64,
    rho_m: f64,
    a: f64,
    eta: f64,
    gamma: f64,
    tenors: Vec<f64>,
    forwards: Vec<f64>,
    thetas: Vec<f64>,
    theta_max: f64,
}

#[derive(Serialize, Deserialize)]
struct EssviSurfaceRaw {
    rho_0: f64,
    rho_m: f64,
    a: f64,
    eta: f64,
    gamma: f64,
    tenors: Vec<f64>,
    forwards: Vec<f64>,
    thetas: Vec<f64>,
}

impl TryFrom<EssviSurfaceRaw> for EssviSurface {
    type Error = VolSurfError;
    fn try_from(raw: EssviSurfaceRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.rho_0,
            raw.rho_m,
            raw.a,
            raw.eta,
            raw.gamma,
            raw.tenors,
            raw.forwards,
            raw.thetas,
        )
    }
}

impl From<EssviSurface> for EssviSurfaceRaw {
    fn from(s: EssviSurface) -> Self {
        Self {
            rho_0: s.rho_0,
            rho_m: s.rho_m,
            a: s.a,
            eta: s.eta,
            gamma: s.gamma,
            tenors: s.tenors,
            forwards: s.forwards,
            thetas: s.thetas,
        }
    }
}

// Hendriks-Martini Eq. 5.7: upper bound on `a` for calendar no-arb.
// Caller must check rho_diff != 0 before calling.
fn a_max_eq57(gamma: f64, rho_diff: f64, rho_m: f64) -> f64 {
    let gamma_thm = 1.0 - gamma;
    if rho_diff > 0.0 {
        gamma_thm * (1.0 - rho_m) / rho_diff
    } else {
        gamma_thm * (1.0 + rho_m) / (-rho_diff)
    }
}

impl EssviSurface {
    /// Create an eSSVI surface from parameters and per-tenor data.
    ///
    /// # Arguments
    /// * `rho_0` — Initial correlation at θ ≈ 0, must be in (−1, 1)
    /// * `rho_m` — Mature correlation at θ = θ_max, must be in (−1, 1)
    /// * `a` — Shape exponent for ρ(θ) power law, must be non-negative
    /// * `eta` — Smile amplitude, must be positive
    /// * `gamma` — Term structure decay, must be in \[0, 1\]
    /// * `tenors` — Expiries in years, strictly increasing, all positive
    /// * `forwards` — Forward prices at each tenor, all positive
    /// * `thetas` — ATM total variances θ_i, strictly increasing, all positive
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is invalid,
    /// including violation of the Hendriks-Martini Eq. 5.7 constraint on `a`.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        rho_0: f64,
        rho_m: f64,
        a: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> error::Result<Self> {
        if rho_0.abs() >= 1.0 || rho_0.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho_0| must be less than 1, got {rho_0}"),
            });
        }
        if rho_m.abs() >= 1.0 || rho_m.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho_m| must be less than 1, got {rho_m}"),
            });
        }
        if !a.is_finite() || a < 0.0 {
            return Err(VolSurfError::InvalidInput {
                message: format!("a must be non-negative and finite, got {a}"),
            });
        }
        validate_positive(eta, "eta")?;
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(VolSurfError::InvalidInput {
                message: format!("gamma must be in [0, 1], got {gamma}"),
            });
        }

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

        let theta_max = *thetas.last().unwrap();

        let rho_diff = rho_m - rho_0;
        if rho_diff.abs() > 1e-14 {
            let a_max = a_max_eq57(gamma, rho_diff, rho_m);
            if a > a_max + 1e-12 {
                return Err(VolSurfError::InvalidInput {
                    message: format!("a={a} exceeds calendar no-arb bound {a_max:.6} (Eq. 5.7)"),
                });
            }
        }

        Ok(Self {
            rho_0,
            rho_m,
            a,
            eta,
            gamma,
            tenors,
            forwards,
            thetas,
            theta_max,
        })
    }

    /// Calibrate an eSSVI surface from per-tenor market implied vols.
    ///
    /// Uses a three-stage approach:
    /// 1. Per-tenor SVI calibration to extract ATM total variances (θ) and skews (ρ)
    /// 2. Fit the ρ(θ) parametric curve to per-tenor skews (Hendriks-Martini Eq. 5.6)
    /// 3. Global optimization of (η, γ) via grid search + Nelder-Mead
    ///
    /// The Eq. 5.7 calendar no-arbitrage constraint on the shape exponent `a` is
    /// enforced dynamically as γ varies during Stage 3 optimization.
    ///
    /// # Arguments
    /// * `market_data` — Per-tenor slices of (strike, implied_vol) pairs (≥ 5 per tenor)
    /// * `tenors` — Expiry times in years, all positive and finite
    /// * `forwards` — Forward prices at each tenor, all positive and finite
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for invalid inputs (< 2 tenors,
    /// length mismatches, non-positive values). Returns [`VolSurfError::CalibrationError`]
    /// if per-tenor SVI calibration or global optimization fails.
    ///
    /// # Example
    /// ```
    /// use volsurf::surface::EssviSurface;
    ///
    /// let data_3m: Vec<(f64, f64)> = (0..10)
    ///     .map(|i| (80.0 + 4.0 * i as f64, 0.20 + 0.01 * (i as f64 - 5.0).abs()))
    ///     .collect();
    /// let data_1y: Vec<(f64, f64)> = (0..10)
    ///     .map(|i| (80.0 + 4.0 * i as f64, 0.18 + 0.008 * (i as f64 - 5.0).abs()))
    ///     .collect();
    ///
    /// let surface = EssviSurface::calibrate(
    ///     &[data_3m, data_1y],
    ///     &[0.25, 1.0],
    ///     &[100.0, 100.0],
    /// )?;
    /// # Ok::<(), volsurf::VolSurfError>(())
    /// ```
    pub fn calibrate(
        market_data: &[Vec<(f64, f64)>],
        tenors: &[f64],
        forwards: &[f64],
    ) -> error::Result<Self> {
        #[cfg(feature = "logging")]
        tracing::debug!(n_tenors = tenors.len(), "eSSVI calibration started");

        const MIN_TENORS: usize = 2;
        const GRID_N: usize = 15;
        const NM_MAX_ITER: usize = 300;
        const NM_DIAMETER_TOL: f64 = 1e-8;
        const NM_FVALUE_TOL: f64 = 1e-12;
        const A_SCAN_STEPS: usize = 20;
        const A_SCAN_MAX: f64 = 3.0;

        if tenors.len() < MIN_TENORS {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "at least {MIN_TENORS} tenors required for eSSVI calibration, got {}",
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

        // Stage 1: per-tenor SVI calibration → thetas + per-tenor rho estimates
        let n_tenors = tenors.len();
        let mut thetas = Vec::with_capacity(n_tenors);
        let mut rhos = Vec::with_capacity(n_tenors);

        for (i, market_vols) in market_data.iter().enumerate() {
            let svi = crate::smile::SviSmile::calibrate(forwards[i], tenors[i], market_vols)
                .map_err(|e| VolSurfError::CalibrationError {
                    message: format!(
                        "per-tenor SVI calibration failed for tenor[{i}]={}: {e}",
                        tenors[i]
                    ),
                    model: "eSSVI",
                    rms_error: None,
                })?;
            let theta = svi.variance(forwards[i])?.0;
            thetas.push(theta);
            rhos.push(svi.rho());
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
                    model: "eSSVI",
                    rms_error: None,
                });
            }
        }

        let theta_max = *thetas.last().unwrap();
        let xs: Vec<f64> = thetas.iter().map(|&t| t / theta_max).collect();

        // Stage 2: fit rho(theta) = rho_0 + (rho_m - rho_0) * (theta/theta_max)^a
        // For given (rho_0, rho_m), scan `a` values to minimize RSS vs per-tenor rhos.
        let fit_a = |r0: f64, rm: f64| -> (f64, f64) {
            let d = rm - r0;
            if d.abs() < 1e-14 {
                let rss: f64 = rhos.iter().map(|&r| (r0 - r).powi(2)).sum();
                return (0.0, rss);
            }
            let mut best_a = 0.5;
            let mut best_rss = f64::MAX;
            for ia in 0..=A_SCAN_STEPS {
                let a = A_SCAN_MAX * (ia as f64) / (A_SCAN_STEPS as f64);
                let rss: f64 = rhos
                    .iter()
                    .enumerate()
                    .map(|(i, &rho_obs)| {
                        let rho_pred = r0 + d * xs[i].powf(a);
                        (rho_pred - rho_obs).powi(2)
                    })
                    .sum();
                if rss < best_rss {
                    best_rss = rss;
                    best_a = a;
                }
            }
            (best_a, best_rss)
        };

        // Adaptive grid range centered on observed rhos
        let rho_min = rhos.iter().cloned().fold(f64::INFINITY, f64::min);
        let rho_max = rhos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rho_lo = (rho_min - 0.15).max(-0.99);
        let rho_hi = (rho_max + 0.15).min(0.99);

        let mut best_r0 = rho_min.clamp(-0.999, 0.999);
        let mut best_rm = rho_max.clamp(-0.999, 0.999);
        let mut best_rho_rss = f64::MAX;

        for ir0 in 0..GRID_N {
            let r0 = rho_lo + (rho_hi - rho_lo) * (ir0 as f64) / ((GRID_N - 1) as f64);
            for irm in 0..GRID_N {
                let rm = rho_lo + (rho_hi - rho_lo) * (irm as f64) / ((GRID_N - 1) as f64);
                let (_, rss) = fit_a(r0, rm);
                if rss < best_rho_rss {
                    best_rho_rss = rss;
                    best_r0 = r0;
                    best_rm = rm;
                }
            }
        }

        // Refine (rho_0, rho_m) with Nelder-Mead
        let rho_step = (rho_hi - rho_lo) / (GRID_N as f64) * 0.5;
        let nm_config = crate::optim::NelderMeadConfig {
            max_iter: NM_MAX_ITER,
            diameter_tol: NM_DIAMETER_TOL,
            fvalue_tol: NM_FVALUE_TOL,
        };
        let nm_rho = crate::optim::nelder_mead_2d(
            |r0, rm| {
                if r0.abs() >= 0.999 || rm.abs() >= 0.999 || !r0.is_finite() || !rm.is_finite() {
                    return f64::MAX;
                }
                fit_a(r0, rm).1
            },
            best_r0,
            best_rm,
            rho_step,
            rho_step,
            &nm_config,
        );

        let opt_rho_0 = nm_rho.x.clamp(-0.999, 0.999);
        let opt_rho_m = nm_rho.y.clamp(-0.999, 0.999);
        let (opt_a_fit, _) = fit_a(opt_rho_0, opt_rho_m);

        #[cfg(feature = "logging")]
        tracing::debug!(
            rho_0 = opt_rho_0,
            rho_m = opt_rho_m,
            a = opt_a_fit,
            "eSSVI Stage 2 rho(theta) fit complete"
        );

        // Observation triples: (theta, log_moneyness, total_variance)
        let mut all_points: Vec<(f64, f64, f64)> =
            Vec::with_capacity(market_data.iter().map(|v| v.len()).sum());
        for (i, market_vols) in market_data.iter().enumerate() {
            for &(strike, vol) in market_vols {
                let k = (strike / forwards[i]).ln();
                let w_obs = vol * vol * tenors[i];
                all_points.push((thetas[i], k, w_obs));
            }
        }

        // Stage 3: optimize (eta, gamma) with fitted rho(theta).
        // Eq. 5.7 enforced dynamically: a_eff = min(a_fitted, a_max(gamma)).
        let rho_diff = opt_rho_m - opt_rho_0;
        let objective = |eta: f64, gamma: f64| -> f64 {
            if eta <= 0.0 || !eta.is_finite() || !gamma.is_finite() || !(0.0..=1.0).contains(&gamma)
            {
                return f64::MAX;
            }
            let a_eff = if rho_diff.abs() > 1e-14 {
                let a_mx = a_max_eq57(gamma, rho_diff, opt_rho_m);
                if a_mx < 0.0 {
                    return f64::MAX;
                }
                opt_a_fit.min(a_mx)
            } else {
                opt_a_fit
            };

            let mut rss = 0.0;
            for &(theta, k, w_obs) in &all_points {
                let theta_c = theta.max(1e-10);
                let t_ratio = (theta_c / theta_max).clamp(0.0, 1.0);
                let rho = (opt_rho_0 + (opt_rho_m - opt_rho_0) * t_ratio.powf(a_eff))
                    .clamp(-0.999, 0.999);
                let phi = eta / theta_c.powf(gamma);
                let phi_k = phi * k;
                let one_minus_rho_sq = 1.0 - rho * rho;
                let w_pred = (theta / 2.0)
                    * (1.0 + rho * phi_k + ((phi_k + rho).powi(2) + one_minus_rho_sq).sqrt());
                if !w_pred.is_finite() {
                    return f64::MAX;
                }
                rss += (w_pred - w_obs).powi(2);
            }
            rss
        };

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
                model: "eSSVI",
                rms_error: None,
            });
        }

        let step_eta = (eta_hi - eta_lo) / (GRID_N as f64) * 0.5;
        let step_gamma = (gamma_hi - gamma_lo) / (GRID_N as f64) * 0.5;

        let nm_result = crate::optim::nelder_mead_2d(
            objective, best_eta, best_gamma, step_eta, step_gamma, &nm_config,
        );

        let opt_eta = nm_result.x.max(1e-6);
        let opt_gamma = nm_result.y.clamp(0.0, 1.0);

        let final_a = if rho_diff.abs() > 1e-14 {
            let a_mx = a_max_eq57(opt_gamma, rho_diff, opt_rho_m);
            opt_a_fit.min(a_mx.max(0.0))
        } else {
            opt_a_fit
        };

        let n_points = all_points.len();
        let rms = if n_points > 0 {
            (nm_result.fval / n_points as f64).sqrt()
        } else {
            0.0
        };

        #[cfg(feature = "logging")]
        tracing::debug!(
            rho_0 = opt_rho_0,
            rho_m = opt_rho_m,
            a = final_a,
            eta = opt_eta,
            gamma = opt_gamma,
            rms_total_variance = rms,
            "eSSVI calibration complete"
        );

        Self::new(
            opt_rho_0,
            opt_rho_m,
            final_a,
            opt_eta,
            opt_gamma,
            tenors.to_vec(),
            forwards.to_vec(),
            thetas,
        )
        .map_err(|e| VolSurfError::CalibrationError {
            message: format!("calibrated eSSVI params invalid: {e}"),
            model: "eSSVI",
            rms_error: Some(rms),
        })
    }

    /// Maturity-dependent correlation ρ(θ) (Hendriks-Martini 2019, Eq. 3.1).
    ///
    /// ```text
    /// ρ(θ) = ρ₀ + (ρₘ − ρ₀) · (θ / θ_max)^a
    /// ```
    ///
    /// Result is clamped to (−0.999, 0.999).
    pub fn rho(&self, theta: f64) -> f64 {
        let t = (theta / self.theta_max).clamp(0.0, 1.0);
        let r = self.rho_0 + (self.rho_m - self.rho_0) * t.powf(self.a);
        r.clamp(-0.999, 0.999)
    }

    pub fn rho_0(&self) -> f64 {
        self.rho_0
    }

    pub fn rho_m(&self) -> f64 {
        self.rho_m
    }

    pub fn a(&self) -> f64 {
        self.a
    }

    pub fn eta(&self) -> f64 {
        self.eta
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    pub fn forwards(&self) -> &[f64] {
        &self.forwards
    }

    pub fn thetas(&self) -> &[f64] {
        &self.thetas
    }

    pub fn theta_max(&self) -> f64 {
        self.theta_max
    }

    /// Evaluate eSSVI total variance at `(θ, k)` with maturity-dependent ρ.
    pub(crate) fn total_variance_at(&self, theta: f64, k: f64) -> f64 {
        let rho = self.rho(theta);
        let phi = self.eta / theta.powf(self.gamma);
        let phi_k = phi * k;
        let one_minus_rho_sq = 1.0 - rho * rho;
        (theta / 2.0) * (1.0 + rho * phi_k + ((phi_k + rho).powi(2) + one_minus_rho_sq).sqrt())
    }

    /// Interpolate `(θ, F)` at an arbitrary expiry.
    pub(crate) fn theta_and_forward_at(&self, expiry: f64) -> (f64, f64) {
        super::interp::interpolate_theta_forward(&self.tenors, &self.thetas, &self.forwards, expiry)
    }

    /// Structural calendar no-arb check (Hendriks-Martini 2019, Thm 4.1, Eq 4.10).
    ///
    /// For power-law φ = η/θ^γ with γ ∈ \[0, 1\], the theorem's γ_thm = 1 − γ
    /// also lies in \[0, 1\], so only the first branch applies:
    ///
    /// ```text
    /// (δ + ρ·γ_thm)² ≤ γ_thm²
    /// ```
    ///
    /// The second branch (2γ_thm − 1) is automatically satisfied when γ_thm ≤ 1,
    /// because γ_thm² ≤ γ_thm and the algebraic conditions in the appendix
    /// (A.10, A.15) follow. See the paper's Section 4.3 for the full argument.
    ///
    /// δ(θ) = a·(ρₘ − ρ₀)·(θ/θ_max)^a is the scaled derivative θ·ρ'(θ).
    ///
    /// Returns violations at each stored tenor where the condition fails.
    /// An empty vector means the surface is structurally calendar-arb-free.
    /// For surfaces passing the Eq. 5.7 constraint at construction, this
    /// always returns empty.
    pub fn calendar_check_structural(&self) -> Vec<StructuralViolation> {
        let gamma_thm = 1.0 - self.gamma;
        let mut violations = Vec::new();

        for (i, &theta) in self.thetas.iter().enumerate() {
            let rho = self.rho(theta);
            let t = (theta / self.theta_max).clamp(0.0, 1.0);
            let delta = self.a * (self.rho_m - self.rho_0) * t.powf(self.a);
            let lhs = (delta + rho * gamma_thm).abs();

            if lhs > gamma_thm + 1e-10 {
                violations.push(StructuralViolation {
                    tenor: self.tenors[i],
                    theta,
                    condition_lhs: lhs,
                    condition_rhs: gamma_thm,
                });
            }
        }
        violations
    }
}

impl VolSurface for EssviSurface {
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
                message: format!("eSSVI total variance is negative: w({k}) = {w}"),
            });
        }
        Ok(Variance(w))
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        validate_positive(expiry, "expiry")?;
        let (theta, forward) = self.theta_and_forward_at(expiry);
        let rho = self.rho(theta);
        let slice = EssviSlice::new(forward, expiry, rho, self.eta, self.gamma, theta)?;
        Ok(Box::new(slice))
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        let mut smile_reports = Vec::with_capacity(self.tenors.len());
        for (i, &tenor) in self.tenors.iter().enumerate() {
            let rho = self.rho(self.thetas[i]);
            let slice = EssviSlice::new(
                self.forwards[i],
                tenor,
                rho,
                self.eta,
                self.gamma,
                self.thetas[i],
            )?;
            smile_reports.push(slice.is_arbitrage_free()?);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn equity_slice() -> EssviSlice {
        EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap()
    }

    #[test]
    fn new_valid_params() {
        let s = equity_slice();
        assert_eq!(s.forward(), 100.0);
        assert_eq!(s.expiry(), 1.0);
        assert_eq!(s.theta(), 0.16);
        assert_eq!(s.rho(), -0.3);
        assert_eq!(s.eta(), 0.5);
        assert_eq!(s.gamma(), 0.5);
    }

    #[test]
    fn new_rejects_invalid_params() {
        assert!(EssviSlice::new(0.0, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(-1.0, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 0.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, -1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, 1.0, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -1.0, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.0, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, -0.1, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, -0.1, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 1.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.0).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, -0.01).is_err());
    }

    #[test]
    fn new_rejects_nan_and_inf() {
        assert!(EssviSlice::new(f64::NAN, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, f64::INFINITY, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, f64::NAN, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, f64::NEG_INFINITY, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, f64::NAN, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, f64::INFINITY).is_err());
    }

    #[test]
    fn new_gamma_boundaries_accepted() {
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.0, 0.16).is_ok());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 1.0, 0.16).is_ok());
    }

    #[test]
    fn vol_atm_equals_sqrt_theta_over_t() {
        // ATM: k = 0, w(0) = theta, vol = sqrt(theta / T) = sqrt(0.16) = 0.4
        let s = equity_slice();
        let vol = s.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn variance_atm_equals_theta() {
        let s = equity_slice();
        let var = s.variance(100.0).unwrap();
        assert_abs_diff_eq!(var.0, 0.16, epsilon = 1e-14);
    }

    #[test]
    fn vol_variance_consistency() {
        let s = equity_slice();
        for &strike in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = s.vol(strike).unwrap();
            let var = s.variance(strike).unwrap();
            assert_abs_diff_eq!(var.0, vol.0 * vol.0 * s.expiry(), epsilon = 1e-14);
        }
    }

    #[test]
    fn matches_ssvi_slice() {
        // Same params should produce identical vols — the newtype is transparent.
        let essvi = equity_slice();
        let ssvi = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap();
        for &strike in &[70.0, 85.0, 100.0, 115.0, 130.0] {
            let v_essvi = essvi.vol(strike).unwrap().0;
            let v_ssvi = ssvi.vol(strike).unwrap().0;
            assert_eq!(v_essvi.to_bits(), v_ssvi.to_bits(), "strike={strike}");
        }
    }

    #[test]
    fn vol_rejects_non_positive_strikes() {
        let s = equity_slice();
        assert!(s.vol(0.0).is_err());
        assert!(s.vol(-100.0).is_err());
        assert!(s.variance(0.0).is_err());
    }

    #[test]
    fn skew_direction() {
        let s = equity_slice();
        let vol_put = s.vol(80.0).unwrap().0;
        let vol_call = s.vol(120.0).unwrap().0;
        assert!(vol_put > vol_call, "negative rho should produce put skew");
    }

    #[test]
    fn density_positive_near_atm() {
        let s = equity_slice();
        for &strike in &[90.0, 95.0, 100.0, 105.0, 110.0] {
            let d = s.density(strike).unwrap();
            assert!(d > 0.0, "density({strike}) = {d}");
        }
    }

    #[test]
    fn arb_free_conservative_params() {
        let s = equity_slice();
        let report = s.is_arbitrage_free().unwrap();
        assert!(report.is_free);
        assert!(report.butterfly_violations.is_empty());
    }

    #[test]
    fn arb_detected_extreme_params() {
        // eta * (1 + |rho|) = 3.0 * 1.95 = 5.85 >> 2
        let s = EssviSlice::new(100.0, 1.0, -0.95, 3.0, 0.5, 0.16).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(!report.is_free);
        assert!(!report.butterfly_violations.is_empty());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EssviSlice>();
    }

    #[test]
    fn serde_round_trip() {
        let s = equity_slice();
        let json = serde_json::to_string(&s).unwrap();
        let s2: EssviSlice = serde_json::from_str(&json).unwrap();
        assert_eq!(s.forward(), s2.forward());
        assert_eq!(s.expiry(), s2.expiry());
        assert_eq!(s.theta(), s2.theta());
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.eta(), s2.eta());
        assert_eq!(s.gamma(), s2.gamma());
        assert_abs_diff_eq!(
            s.vol(90.0).unwrap().0,
            s2.vol(90.0).unwrap().0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn serde_rejects_invalid_rho() {
        let json = r#"{"forward":100,"expiry":1,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_forward() {
        let json = r#"{"forward":-100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_expiry() {
        let json = r#"{"forward":100,"expiry":0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_eta() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":-0.1,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_gamma_out_of_range() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":2.0,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_theta() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.0}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_error_contains_validation_message() {
        let json = r#"{"forward":100,"expiry":1,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        let err = serde_json::from_str::<EssviSlice>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rho"),
            "should contain domain message, got: {msg}"
        );
    }

    #[test]
    fn short_expiry_slice() {
        // 1-week tenor: T=7/365, theta=0.04^2 * 7/365 ≈ 3.07e-5
        let t = 7.0 / 365.0;
        let theta = 0.04 * t;
        let s = EssviSlice::new(100.0, t, -0.5, 0.8, 0.4, theta).unwrap();
        let vol = s.vol(100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 2.0);
    }

    #[test]
    fn long_expiry_slice() {
        // 5-year tenor
        let s = EssviSlice::new(100.0, 5.0, -0.2, 0.3, 0.6, 0.50).unwrap();
        let vol = s.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, (0.50 / 5.0_f64).sqrt(), epsilon = 1e-10);
    }

    // ── EssviSurface tests ──────────────────────────────────────────

    fn equity_surface() -> EssviSurface {
        // a_max = (1-0.5)*(1-(-0.2))/((-0.2)-(-0.4)) = 0.5*1.2/0.2 = 3.0
        EssviSurface::new(
            -0.4,
            -0.2,
            0.5,
            0.5,
            0.5,
            vec![0.25, 0.5, 1.0, 2.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![0.04, 0.08, 0.16, 0.32],
        )
        .unwrap()
    }

    fn two_tenor_surface() -> EssviSurface {
        EssviSurface::new(
            -0.3,
            -0.1,
            0.3,
            0.4,
            0.6,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.08, 0.16],
        )
        .unwrap()
    }

    #[test]
    fn surface_new_valid() {
        let s = equity_surface();
        assert_eq!(s.rho_0(), -0.4);
        assert_eq!(s.rho_m(), -0.2);
        assert_eq!(s.a(), 0.5);
        assert_eq!(s.eta(), 0.5);
        assert_eq!(s.gamma(), 0.5);
        assert_eq!(s.tenors(), &[0.25, 0.5, 1.0, 2.0]);
        assert_eq!(s.forwards(), &[100.0, 100.0, 100.0, 100.0]);
        assert_eq!(s.thetas(), &[0.04, 0.08, 0.16, 0.32]);
        assert_eq!(s.theta_max(), 0.32);
    }

    #[test]
    fn surface_new_single_tenor() {
        let s = EssviSurface::new(
            -0.3,
            -0.1,
            0.3,
            0.5,
            0.5,
            vec![1.0],
            vec![100.0],
            vec![0.04],
        )
        .unwrap();
        assert_eq!(s.tenors().len(), 1);
        assert_eq!(s.theta_max(), 0.04);
    }

    #[test]
    fn surface_new_rejects_rho_0_at_boundary() {
        assert!(
            EssviSurface::new(1.0, -0.2, 0.5, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04],)
                .is_err()
        );
        assert!(
            EssviSurface::new(
                -1.0,
                -0.2,
                0.5,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_rho_m_at_boundary() {
        assert!(
            EssviSurface::new(-0.3, 1.0, 0.5, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04],)
                .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -1.0,
                0.5,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_negative_a() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                -0.1,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_accepts_a_zero() {
        // a=0 means constant rho
        let s = EssviSurface::new(
            -0.3,
            -0.1,
            0.0,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.04, 0.08],
        )
        .unwrap();
        assert_eq!(s.a(), 0.0);
    }

    #[test]
    fn surface_new_rejects_bad_eta() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.0,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                -1.0,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_gamma_out_of_range() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                -0.1,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                1.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_gamma_boundaries_accepted() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.3,
                0.0,
                0.5,
                0.0,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_ok()
        );
        // gamma=1 forces constant rho (rho_0 == rho_m)
        assert!(
            EssviSurface::new(
                -0.3,
                -0.3,
                0.0,
                0.5,
                1.0,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_ok()
        );
    }

    #[test]
    fn surface_new_rejects_empty_tenors() {
        assert!(EssviSurface::new(-0.3, -0.1, 0.3, 0.5, 0.5, vec![], vec![], vec![],).is_err());
    }

    #[test]
    fn surface_new_rejects_length_mismatch() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
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
    fn surface_new_rejects_non_increasing_tenors() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![1.0, 0.5],
                vec![100.0, 100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_non_increasing_thetas() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.16, 0.08],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_nan_in_vectors() {
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![f64::NAN],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![1.0],
                vec![f64::INFINITY],
                vec![0.04],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.3,
                -0.1,
                0.3,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![f64::NEG_INFINITY],
            )
            .is_err()
        );
    }

    #[test]
    fn surface_new_rejects_a_exceeding_eq57() {
        // rho_0=-0.4, rho_m=-0.2, gamma=0.5 => a_max = 0.5*1.2/0.2 = 3.0
        assert!(
            EssviSurface::new(
                -0.4,
                -0.2,
                3.5,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.04, 0.08],
            )
            .is_err()
        );
        // exactly at boundary should pass
        assert!(
            EssviSurface::new(
                -0.4,
                -0.2,
                3.0,
                0.5,
                0.5,
                vec![0.5, 1.0],
                vec![100.0, 100.0],
                vec![0.04, 0.08],
            )
            .is_ok()
        );
    }

    #[test]
    fn surface_new_eq57_reverse_rho() {
        // rho_0 > rho_m: a_max = (1-gamma)*(1+rho_m)/(rho_0-rho_m)
        // rho_0=-0.1, rho_m=-0.4, gamma=0.5 => 0.5*0.6/0.3 = 1.0
        assert!(
            EssviSurface::new(
                -0.1,
                -0.4,
                1.5,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_err()
        );
        assert!(
            EssviSurface::new(
                -0.1,
                -0.4,
                1.0,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_ok()
        );
    }

    #[test]
    fn surface_new_equal_rho_any_a() {
        // rho_0 == rho_m means rho_diff=0, no constraint on a
        assert!(
            EssviSurface::new(
                -0.3,
                -0.3,
                100.0,
                0.5,
                0.5,
                vec![1.0],
                vec![100.0],
                vec![0.04],
            )
            .is_ok()
        );
    }

    #[test]
    fn rho_at_zero_theta() {
        let s = equity_surface();
        // theta=0 => (0/theta_max)^a = 0 => rho(0) = rho_0
        let r = s.rho(0.0);
        assert_abs_diff_eq!(r, -0.4, epsilon = 1e-14);
    }

    #[test]
    fn rho_at_theta_max() {
        let s = equity_surface();
        // theta=theta_max => (1)^a = 1 => rho = rho_0 + (rho_m - rho_0) = rho_m
        let r = s.rho(s.theta_max());
        assert_abs_diff_eq!(r, -0.2, epsilon = 1e-14);
    }

    #[test]
    fn rho_at_half_theta_max() {
        let s = equity_surface();
        // theta=0.16, theta_max=0.32 => t=0.5, a=0.5 => t^a = sqrt(0.5)
        let expected = -0.4 + (-0.2 - (-0.4)) * (0.5_f64).sqrt();
        let r = s.rho(0.16);
        assert_abs_diff_eq!(r, expected, epsilon = 1e-14);
    }

    #[test]
    fn rho_monotone_when_rho_m_greater() {
        let s = equity_surface(); // rho_0=-0.4, rho_m=-0.2
        let r1 = s.rho(0.04);
        let r2 = s.rho(0.16);
        let r3 = s.rho(0.32);
        assert!(r1 < r2, "rho should increase: {r1} < {r2}");
        assert!(r2 < r3, "rho should increase: {r2} < {r3}");
    }

    #[test]
    fn rho_decreasing_when_rho_m_less() {
        // rho_0=-0.1, rho_m=-0.4 => decreasing
        let s = EssviSurface::new(
            -0.1,
            -0.4,
            0.5,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.04, 0.08],
        )
        .unwrap();
        assert!(s.rho(0.04) > s.rho(0.08));
    }

    #[test]
    fn rho_constant_when_a_zero() {
        let s = EssviSurface::new(
            -0.3,
            -0.1,
            0.0,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.04, 0.08],
        )
        .unwrap();
        // a=0 => t^0 = 1 for all t>0, so rho = rho_0 + (rho_m - rho_0)*1 = rho_m
        // BUT at theta=0, t=0, 0^0 = 1 by f64 convention
        assert_abs_diff_eq!(s.rho(0.04), s.rho(0.08), epsilon = 1e-14);
    }

    #[test]
    fn rho_linear_when_a_one() {
        let s = EssviSurface::new(
            -0.4,
            -0.2,
            1.0,
            0.5,
            0.5,
            vec![0.5, 1.0],
            vec![100.0, 100.0],
            vec![0.04, 0.08],
        )
        .unwrap();
        // rho(theta) = rho_0 + (rho_m-rho_0)*(theta/theta_max)
        let expected = -0.4 + 0.2 * (0.04 / 0.08);
        assert_abs_diff_eq!(s.rho(0.04), expected, epsilon = 1e-14);
    }

    #[test]
    fn rho_clamped_to_valid_range() {
        // Even with extreme extrapolation, rho stays in (-0.999, 0.999)
        let s = equity_surface();
        let r = s.rho(100.0); // way beyond theta_max
        assert!(r > -0.999 && r < 0.999);
        let r0 = s.rho(0.0);
        assert!(r0 > -0.999 && r0 < 0.999);
    }

    #[test]
    fn surface_atm_variance_at_stored_tenor() {
        let s = equity_surface();
        for (i, &t) in s.tenors().iter().enumerate() {
            let var = s.black_variance(t, s.forwards()[i]).unwrap();
            assert_abs_diff_eq!(var.0, s.thetas()[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn surface_atm_vol_at_stored_tenor() {
        let s = equity_surface();
        for (i, &t) in s.tenors().iter().enumerate() {
            let vol = s.black_vol(t, s.forwards()[i]).unwrap();
            let expected = (s.thetas()[i] / t).sqrt();
            assert_abs_diff_eq!(vol.0, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn surface_vol_variance_consistency() {
        let s = equity_surface();
        for &t in &[0.25, 0.5, 1.0, 2.0] {
            for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
                let vol = s.black_vol(t, k).unwrap();
                let var = s.black_variance(t, k).unwrap();
                assert_abs_diff_eq!(var.0, vol.0 * vol.0 * t, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn surface_variance_positive() {
        let s = equity_surface();
        for &t in &[0.25, 0.5, 1.0, 2.0] {
            for &k in &[60.0, 80.0, 100.0, 120.0, 150.0] {
                let var = s.black_variance(t, k).unwrap();
                assert!(var.0 > 0.0, "variance should be positive at T={t}, K={k}");
            }
        }
    }

    #[test]
    fn surface_skew_matches_rho_sign() {
        let s = equity_surface(); // rho < 0 everywhere
        for &t in s.tenors() {
            let vol_put = s.black_vol(t, 80.0).unwrap().0;
            let vol_call = s.black_vol(t, 120.0).unwrap().0;
            assert!(
                vol_put > vol_call,
                "negative rho should produce put skew at T={t}"
            );
        }
    }

    #[test]
    fn surface_interpolation_between_tenors() {
        let s = equity_surface();
        // T=0.75 is between tenor[1]=0.5 and tenor[2]=1.0
        let vol = s.black_vol(0.75, 100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
        // variance should interpolate monotonically between neighbors
        let var = s.black_variance(0.75, 100.0).unwrap().0;
        let var_lo = s.black_variance(0.5, 100.0).unwrap().0;
        let var_hi = s.black_variance(1.0, 100.0).unwrap().0;
        assert!(
            var > var_lo && var < var_hi,
            "variance should interpolate monotonically"
        );
    }

    #[test]
    fn surface_extrapolation_before_first_tenor() {
        let s = equity_surface();
        // T=0.1 < tenor[0]=0.25: flat vol extrapolation => theta = theta_0 * T/T_0
        let var = s.black_variance(0.1, 100.0).unwrap();
        let expected_theta = 0.04 * 0.1 / 0.25;
        assert_abs_diff_eq!(var.0, expected_theta, epsilon = 1e-12);
    }

    #[test]
    fn surface_extrapolation_after_last_tenor() {
        let s = equity_surface();
        // T=3.0 > tenor[3]=2.0: flat vol extrapolation => theta = theta_n * T/T_n
        let var = s.black_variance(3.0, 100.0).unwrap();
        let expected_theta = 0.32 * 3.0 / 2.0;
        assert_abs_diff_eq!(var.0, expected_theta, epsilon = 1e-12);
    }

    #[test]
    fn surface_rejects_non_positive_expiry() {
        let s = equity_surface();
        assert!(s.black_vol(0.0, 100.0).is_err());
        assert!(s.black_vol(-1.0, 100.0).is_err());
        assert!(s.black_variance(0.0, 100.0).is_err());
    }

    #[test]
    fn surface_rejects_non_positive_strike() {
        let s = equity_surface();
        assert!(s.black_vol(1.0, 0.0).is_err());
        assert!(s.black_vol(1.0, -100.0).is_err());
        assert!(s.black_variance(1.0, 0.0).is_err());
    }

    #[test]
    fn surface_smile_at_returns_essvi_slice() {
        let s = equity_surface();
        let smile = s.smile_at(1.0).unwrap();
        assert_eq!(smile.forward(), 100.0);
        assert_abs_diff_eq!(smile.expiry(), 1.0, epsilon = 1e-10);
        // vol at ATM should match surface query
        let smile_vol = smile.vol(100.0).unwrap().0;
        let surface_vol = s.black_vol(1.0, 100.0).unwrap().0;
        assert_abs_diff_eq!(smile_vol, surface_vol, epsilon = 1e-14);
    }

    #[test]
    fn surface_smile_at_interpolated_tenor() {
        let s = equity_surface();
        let smile = s.smile_at(0.75).unwrap();
        assert!(smile.expiry() > 0.0);
        // vol at ATM should match the direct surface query
        let smile_vol = smile.vol(smile.forward()).unwrap().0;
        let surface_vol = s.black_vol(0.75, smile.forward()).unwrap().0;
        assert_abs_diff_eq!(smile_vol, surface_vol, epsilon = 1e-12);
    }

    #[test]
    fn surface_smile_at_rejects_non_positive() {
        let s = equity_surface();
        assert!(s.smile_at(0.0).is_err());
        assert!(s.smile_at(-1.0).is_err());
    }

    #[test]
    fn surface_diagnostics_clean() {
        let s = equity_surface();
        let diag = s.diagnostics().unwrap();
        assert_eq!(diag.smile_reports.len(), 4);
        assert!(diag.is_free, "conservative params should be arb-free");
        assert!(diag.calendar_violations.is_empty());
    }

    #[test]
    fn surface_diagnostics_two_tenor() {
        let s = two_tenor_surface();
        let diag = s.diagnostics().unwrap();
        assert_eq!(diag.smile_reports.len(), 2);
        assert!(diag.is_free);
    }

    #[test]
    fn surface_calendar_structural_clean() {
        let s = equity_surface();
        let violations = s.calendar_check_structural();
        assert!(
            violations.is_empty(),
            "valid params should pass structural check"
        );
    }

    #[test]
    fn surface_rho_differs_per_tenor_in_smile() {
        let s = equity_surface();
        let smile_short = s.smile_at(0.25).unwrap();
        let smile_long = s.smile_at(2.0).unwrap();
        // OTM put vol should differ due to different rho at each tenor
        let vol_short = smile_short.vol(80.0).unwrap().0;
        let vol_long = smile_long.vol(80.0).unwrap().0;
        // They shouldn't be identical (different rho at each theta)
        assert!(
            (vol_short - vol_long).abs() > 1e-6,
            "different rho(theta) should produce different skew"
        );
    }

    #[test]
    fn surface_serde_round_trip() {
        let s = equity_surface();
        let json = serde_json::to_string(&s).unwrap();
        let s2: EssviSurface = serde_json::from_str(&json).unwrap();
        assert_eq!(s.rho_0(), s2.rho_0());
        assert_eq!(s.rho_m(), s2.rho_m());
        assert_eq!(s.a(), s2.a());
        assert_eq!(s.eta(), s2.eta());
        assert_eq!(s.gamma(), s2.gamma());
        assert_eq!(s.tenors(), s2.tenors());
        assert_eq!(s.forwards(), s2.forwards());
        assert_eq!(s.thetas(), s2.thetas());
        assert_eq!(s.theta_max(), s2.theta_max());
        // vol query should be identical
        let v1 = s.black_vol(1.0, 90.0).unwrap().0;
        let v2 = s2.black_vol(1.0, 90.0).unwrap().0;
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    #[test]
    fn surface_serde_no_theta_max_in_json() {
        let s = equity_surface();
        let json = serde_json::to_string(&s).unwrap();
        assert!(
            !json.contains("theta_max"),
            "derived field should not be serialized"
        );
    }

    #[test]
    fn surface_serde_rejects_invalid_rho_0() {
        let json = r#"{"rho_0":1.0,"rho_m":-0.2,"a":0.5,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<EssviSurface>(json).is_err());
    }

    #[test]
    fn surface_serde_rejects_invalid_rho_m() {
        let json = r#"{"rho_0":-0.3,"rho_m":-1.0,"a":0.5,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<EssviSurface>(json).is_err());
    }

    #[test]
    fn surface_serde_rejects_negative_eta() {
        let json = r#"{"rho_0":-0.3,"rho_m":-0.1,"a":0.3,"eta":-0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        assert!(serde_json::from_str::<EssviSurface>(json).is_err());
    }

    #[test]
    fn surface_serde_rejects_non_increasing_thetas() {
        let json = r#"{"rho_0":-0.3,"rho_m":-0.1,"a":0.3,"eta":0.5,"gamma":0.5,"tenors":[0.5,1.0],"forwards":[100.0,100.0],"thetas":[0.16,0.08]}"#;
        assert!(serde_json::from_str::<EssviSurface>(json).is_err());
    }

    #[test]
    fn surface_serde_rejects_length_mismatch() {
        let json = r#"{"rho_0":-0.3,"rho_m":-0.1,"a":0.3,"eta":0.5,"gamma":0.5,"tenors":[0.5,1.0],"forwards":[100.0],"thetas":[0.04,0.08]}"#;
        assert!(serde_json::from_str::<EssviSurface>(json).is_err());
    }

    #[test]
    fn surface_serde_error_contains_field_name() {
        let json = r#"{"rho_0":1.5,"rho_m":-0.2,"a":0.5,"eta":0.5,"gamma":0.5,"tenors":[1.0],"forwards":[100.0],"thetas":[0.04]}"#;
        let err = serde_json::from_str::<EssviSurface>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("rho_0"), "should mention rho_0, got: {msg}");
    }

    #[test]
    fn surface_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EssviSurface>();
    }

    #[test]
    fn surface_as_trait_object() {
        let s = equity_surface();
        let _boxed: Box<dyn VolSurface> = Box::new(s);
    }

    fn synthetic_essvi_data(
        surface: &EssviSurface,
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

    fn rms_vol_error(
        calibrated: &EssviSurface,
        tenors: &[f64],
        market_data: &[Vec<(f64, f64)>],
    ) -> f64 {
        let mut total_rss = 0.0;
        let mut n_points = 0;
        for (i, &t) in tenors.iter().enumerate() {
            for &(strike, vol_obs) in &market_data[i] {
                let vol_fit = calibrated.black_vol(t, strike).unwrap().0;
                total_rss += (vol_fit - vol_obs).powi(2);
                n_points += 1;
            }
        }
        (total_rss / n_points as f64).sqrt()
    }

    #[test]
    fn calibrate_round_trip_equity() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let rms = rms_vol_error(&calibrated, &tenors, &market_data);
        assert!(rms < 0.005, "round-trip RMS {rms} should be < 0.005");
    }

    #[test]
    fn calibrate_round_trip_2_tenors() {
        let original = two_tenor_surface();
        let tenors = vec![0.5, 1.0];
        let forwards = vec![100.0, 100.0];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..10).map(|i| 75.0 + 5.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let rms = rms_vol_error(&calibrated, &tenors, &market_data);
        assert!(
            rms < 0.005,
            "2-tenor round-trip RMS {rms} should be < 0.005"
        );
    }

    #[test]
    fn calibrate_round_trip_varying_forwards() {
        let original = EssviSurface::new(
            -0.4,
            -0.2,
            0.5,
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
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let rms = rms_vol_error(&calibrated, &tenors, &market_data);
        assert!(
            rms < 0.005,
            "varying-forwards round-trip RMS {rms} should be < 0.005"
        );
    }

    #[test]
    fn calibrate_params_in_valid_range() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let c = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        assert!(c.rho_0().abs() < 1.0, "rho_0 out of range: {}", c.rho_0());
        assert!(c.rho_m().abs() < 1.0, "rho_m out of range: {}", c.rho_m());
        assert!(c.a() >= 0.0, "a must be non-negative: {}", c.a());
        assert!(c.eta() > 0.0, "eta must be positive: {}", c.eta());
        assert!(
            (0.0..=1.0).contains(&c.gamma()),
            "gamma out of [0,1]: {}",
            c.gamma()
        );
        assert_eq!(c.tenors().len(), 4);
        assert_eq!(c.forwards().len(), 4);
        assert_eq!(c.thetas().len(), 4);
    }

    #[test]
    fn calibrate_thetas_strictly_increasing() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        for w in calibrated.thetas().windows(2) {
            assert!(
                w[1] > w[0],
                "thetas must be strictly increasing: {} <= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn calibrate_diagnostics_clean() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let diag = calibrated.diagnostics().unwrap();
        assert!(
            diag.is_free,
            "calibrated eSSVI from conservative input should be arb-free"
        );
    }

    #[test]
    fn calibrate_calendar_structural_clean() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let violations = calibrated.calendar_check_structural();
        assert!(
            violations.is_empty(),
            "calibrated surface should pass structural check, got {} violations",
            violations.len()
        );
    }

    #[test]
    fn calibrate_serde_round_trip() {
        let original = equity_surface();
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];
        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();
        let market_data = synthetic_essvi_data(&original, &tenors, &strikes);

        let calibrated = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();
        let json = serde_json::to_string(&calibrated).unwrap();
        let deserialized: EssviSurface = serde_json::from_str(&json).unwrap();

        assert_abs_diff_eq!(calibrated.rho_0(), deserialized.rho_0(), epsilon = 1e-14);
        assert_abs_diff_eq!(calibrated.rho_m(), deserialized.rho_m(), epsilon = 1e-14);
        assert_abs_diff_eq!(calibrated.a(), deserialized.a(), epsilon = 1e-14);
        assert_abs_diff_eq!(calibrated.eta(), deserialized.eta(), epsilon = 1e-14);
        assert_abs_diff_eq!(calibrated.gamma(), deserialized.gamma(), epsilon = 1e-14);
        assert_eq!(calibrated.tenors(), deserialized.tenors());
        assert_eq!(calibrated.thetas(), deserialized.thetas());
        let v1 = calibrated.black_vol(1.0, 90.0).unwrap().0;
        let v2 = deserialized.black_vol(1.0, 90.0).unwrap().0;
        assert_abs_diff_eq!(v1, v2, epsilon = 1e-14);
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
        let result = EssviSurface::calibrate(&data, &[1.0], &[100.0]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_empty_data() {
        let result = EssviSurface::calibrate(&[], &[], &[]);
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
        let result = EssviSurface::calibrate(&data, &[0.5, 1.0], &[100.0]);
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
        let result = EssviSurface::calibrate(&data, &[0.5, 1.0], &[-100.0, 100.0]);
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
        let result = EssviSurface::calibrate(&data, &[0.0, 1.0], &[100.0, 100.0]);
        assert!(result.is_err());
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
        let data = vec![make_smile(100.0, 0.50), make_smile(100.0, 0.20)];
        let result = EssviSurface::calibrate(&data, &[0.25, 0.50], &[100.0, 100.0]);
        let err = result.unwrap_err();
        assert!(matches!(err, VolSurfError::CalibrationError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("non-monotone"),
            "error should mention non-monotone: {msg}"
        );
    }

    #[test]
    fn calibrate_rejects_too_few_points() {
        let data = vec![
            vec![(90.0, 0.3), (100.0, 0.25), (110.0, 0.3)],
            vec![(90.0, 0.3), (100.0, 0.25), (110.0, 0.3)],
        ];
        let result = EssviSurface::calibrate(&data, &[0.5, 1.0], &[100.0, 100.0]);
        assert!(result.is_err());
    }

    #[test]
    fn calibrate_exercises_a_clipping() {
        use crate::smile::SviSmile;

        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let forwards = vec![100.0; 4];

        // Step-like rho transition forces Stage 2 to fit a large `a`.
        // b decays with tenor to give the calibrator freedom on gamma.
        let svi_params: [(f64, f64, f64); 4] = [
            //  (a,     b,    rho)          m=0, sigma=0.1 for all
            (0.005, 0.06, -0.70),
            (0.015, 0.045, -0.68),
            (0.035, 0.03, -0.65),
            (0.070, 0.02, -0.15),
        ];

        let strikes: Vec<Vec<f64>> = tenors
            .iter()
            .map(|_| (0..15).map(|i| 70.0 + 4.0 * i as f64).collect())
            .collect();

        let market_data: Vec<Vec<(f64, f64)>> = tenors
            .iter()
            .zip(&svi_params)
            .zip(&strikes)
            .map(|((&t, &(a, b, rho)), ks)| {
                let svi = SviSmile::new(100.0, t, a, b, rho, 0.0, 0.1).unwrap();
                ks.iter().map(|&k| (k, svi.vol(k).unwrap().0)).collect()
            })
            .collect();

        let cal = EssviSurface::calibrate(&market_data, &tenors, &forwards).unwrap();

        let rho_diff = cal.rho_m() - cal.rho_0();
        let a_bound = a_max_eq57(cal.gamma(), rho_diff, cal.rho_m());

        // a_max well below A_SCAN_MAX=3.0 proves the constraint is non-trivial
        assert!(
            a_bound < 2.5,
            "a_max={a_bound:.3} should be below scan ceiling"
        );

        // Clipping was binding: calibrated a sits at the Eq. 5.7 bound
        assert!(
            cal.a() <= a_bound + 1e-10,
            "a={} exceeds a_max={a_bound}",
            cal.a()
        );
        assert!(
            (cal.a() - a_bound).abs() < 1e-10,
            "a={} should equal a_max={a_bound} (clipping binding)",
            cal.a()
        );

        let rms = rms_vol_error(&cal, &tenors, &market_data);
        assert!(rms < 0.02, "RMS {rms} too large");

        assert!(
            cal.calendar_check_structural().is_empty(),
            "calendar structural violations after a-clipping"
        );
    }
}
