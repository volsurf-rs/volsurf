//! eSSVI (Extended SSVI) surface and slice.
//!
//! eSSVI extends Gatheral-Jacquier's SSVI by making the correlation parameter ρ
//! depend on maturity (through the ATM total variance θ). At the slice level the
//! formula is identical to SSVI — the difference is purely in how ρ is chosen.
//!
//! [`EssviSlice`] is a single-tenor slice implementing [`SmileSection`].
//! [`EssviSurface`] is the multi-tenor surface (stub, implemented in issue #3).
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
                    message: format!(
                        "forwards must be positive and finite, got forwards[{i}]={f}"
                    ),
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

        // Hendriks-Martini Eq. 5.7: upper bound on `a` for calendar no-arb.
        // gamma_thm = 1 - gamma for the power-law phi.
        let rho_diff = rho_m - rho_0;
        if rho_diff.abs() > 1e-14 {
            let gamma_thm = 1.0 - gamma;
            let a_max = if rho_diff > 0.0 {
                gamma_thm * (1.0 - rho_m) / rho_diff
            } else {
                gamma_thm * (1.0 + rho_m) / (-rho_diff)
            };
            if a > a_max + 1e-12 {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "a={a} exceeds calendar no-arb bound {a_max:.6} (Eq. 5.7)"
                    ),
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

    /// Calibrate an eSSVI surface from market data.
    ///
    /// # Errors
    /// Returns [`VolSurfError::NumericalError`] — not yet implemented.
    pub fn calibrate(
        _market_data: &[Vec<(f64, f64)>],
        _tenors: &[f64],
        _forwards: &[f64],
    ) -> error::Result<Self> {
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
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
        (theta / 2.0)
            * (1.0 + rho * phi_k + ((phi_k + rho).powi(2) + one_minus_rho_sq).sqrt())
    }

    /// Interpolate `(θ, F)` at an arbitrary expiry.
    pub(crate) fn theta_and_forward_at(&self, expiry: f64) -> (f64, f64) {
        let n = self.tenors.len();

        for (i, &t) in self.tenors.iter().enumerate() {
            if (expiry - t).abs() < 1e-10 {
                return (self.thetas[i], self.forwards[i]);
            }
        }

        if expiry < self.tenors[0] {
            let theta = self.thetas[0] * expiry / self.tenors[0];
            return (theta, self.forwards[0]);
        }

        if expiry > self.tenors[n - 1] {
            let theta = self.thetas[n - 1] * expiry / self.tenors[n - 1];
            return (theta, self.forwards[n - 1]);
        }

        let right = self.tenors.partition_point(|&t| t < expiry);
        let left = right - 1;
        let alpha = (expiry - self.tenors[left]) / (self.tenors[right] - self.tenors[left]);
        let theta = (1.0 - alpha) * self.thetas[left] + alpha * self.thetas[right];
        let forward = (1.0 - alpha) * self.forwards[left] + alpha * self.forwards[right];
        (theta, forward)
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

        let is_free =
            smile_reports.iter().all(|r| r.is_free) && calendar_violations.is_empty();

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
}
