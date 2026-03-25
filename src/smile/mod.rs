//! Single-tenor volatility smile models.
//!
//! A smile represents how implied volatility varies with strike at a fixed
//! expiry. All models implement the [`SmileSection`] trait.
//!
//! ## Models
//!
//! - [`SviSmile`] — SVI parameterization (Gatheral), 5 parameters
//! - [`SabrSmile`] — SABR stochastic vol model (Hagan et al.), 4 parameters
//! - [`SplineSmile`] — Cubic spline on variance, non-parametric

pub mod arbitrage;
pub mod sabr;
pub mod spline;
pub mod svi;

pub use arbitrage::{ArbitrageReport, ButterflyViolation};
pub use sabr::SabrSmile;
pub use spline::SplineSmile;
pub use svi::SviSmile;

pub(crate) const BUTTERFLY_G_TOL: f64 = 1e-10;
pub(crate) const DENSITY_NEG_TOL: f64 = 1e-8;

use crate::error;
use crate::implied::black::black_price;
use crate::types::{OptionType, Strike, Variance, Vol};
use crate::validate::validate_positive;

/// Grid configuration for butterfly arbitrage scanning.
///
/// Controls the number of sample points and log-moneyness range
/// k = ln(K/F) used when checking `is_arbitrage_free_with`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArbitrageScanConfig {
    pub n_points: usize,
    pub k_min: f64,
    pub k_max: f64,
}

impl ArbitrageScanConfig {
    /// Default for SVI and SSVI models: 200 points over [-3, 3].
    pub fn svi_default() -> Self {
        Self {
            n_points: 200,
            k_min: -3.0,
            k_max: 3.0,
        }
    }

    /// Default for SABR model: 200 points over [-2, 2].
    ///
    /// Narrower range than SVI because Hagan formula breaks down in deep wings.
    pub fn sabr_default() -> Self {
        Self {
            n_points: 200,
            k_min: -2.0,
            k_max: 2.0,
        }
    }

    pub(crate) fn validate(&self) -> error::Result<()> {
        if self.n_points < 2 {
            return Err(error::VolSurfError::InvalidInput {
                message: format!("n_points must be >= 2, got {}", self.n_points),
            });
        }
        if !self.k_min.is_finite() || !self.k_max.is_finite() {
            return Err(error::VolSurfError::InvalidInput {
                message: "k_min and k_max must be finite".to_string(),
            });
        }
        if self.k_min >= self.k_max {
            return Err(error::VolSurfError::InvalidInput {
                message: format!("k_min ({}) must be < k_max ({})", self.k_min, self.k_max),
            });
        }
        Ok(())
    }
}

/// A single-tenor volatility smile.
///
/// Represents the relationship between strike and implied volatility at a
/// fixed expiry. Every smile model in this crate implements this trait,
/// enabling polymorphic use in surface construction.
///
/// # Thread Safety
/// All implementations must be `Send + Sync` for use in concurrent pricing.
///
/// # Error Handling
/// Methods return `Result` so implementations can report numerical failures
/// (e.g., negative variance, NaN) rather than panicking.
///
/// # Default Methods
///
/// [`density()`](SmileSection::density) computes the risk-neutral density
/// q(K) = d²C/dK² via Breeden-Litzenberger (1978) finite differences with
/// relative step h = K × 10⁻⁴. Override this in models with analytical
/// density (e.g., SVI via the g-function) for better accuracy.
///
/// [`variance()`](SmileSection::variance) derives total variance from
/// `vol()` as σ²T. Override when direct variance computation is cheaper.
///
/// # Examples
///
/// ```
/// use volsurf::smile::{SabrSmile, SmileSection};
/// use volsurf::types::Strike;
///
/// // Create a concrete smile and use it through the trait interface
/// let smile = SabrSmile::new(100.0, 1.0, 0.3, 1.0, -0.3, 0.4)?;
///
/// let vol = smile.vol(Strike(100.0))?;
/// assert!(vol.0 > 0.0);
///
/// let var = smile.variance(Strike(100.0))?;
/// assert!((var.0 - vol.0 * vol.0 * smile.expiry()).abs() < 1e-12);
///
/// let report = smile.is_arbitrage_free()?;
/// assert!(report.is_free());
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
pub trait SmileSection: Send + Sync + std::fmt::Debug {
    /// Implied Black volatility σ at the given strike.
    fn vol(&self, strike: Strike) -> error::Result<Vol>;

    /// Total Black variance σ²T at the given strike.
    ///
    /// Default implementation derives from [`vol`](SmileSection::vol):
    /// `variance(K) = vol(K)² × expiry`.
    fn variance(&self, strike: Strike) -> error::Result<Variance> {
        let v = self.vol(strike)?;
        Ok(Variance(v.0 * v.0 * self.expiry()))
    }

    /// Risk-neutral probability density q(K) via Breeden-Litzenberger.
    ///
    /// Default implementation uses finite differences on Black call prices:
    /// `q(K) = d²C/dK²` with step `h = K × 10⁻⁴`.
    ///
    /// Models with analytical density (e.g., SVI via the g-function) should
    /// override this for better accuracy and performance.
    fn density(&self, strike: Strike) -> error::Result<f64> {
        validate_positive(strike.0, "strike")?;
        // Relative perturbation for central finite difference
        let h = strike.0 * 1e-4;
        let k_lo = strike.0 - h;
        let k_hi = strike.0 + h;

        let v_lo = self.vol(Strike(k_lo))?;
        let v_mid = self.vol(strike)?;
        let v_hi = self.vol(Strike(k_hi))?;

        let c_lo = black_price(
            self.forward(),
            k_lo,
            v_lo.0,
            self.expiry(),
            OptionType::Call,
        )?;
        let c_mid = black_price(
            self.forward(),
            strike.0,
            v_mid.0,
            self.expiry(),
            OptionType::Call,
        )?;
        let c_hi = black_price(
            self.forward(),
            k_hi,
            v_hi.0,
            self.expiry(),
            OptionType::Call,
        )?;

        // Breeden-Litzenberger: q(K) = d²C/dK² (undiscounted)
        Ok((c_lo - 2.0 * c_mid + c_hi) / (h * h))
    }

    /// Forward price F at this tenor.
    fn forward(&self) -> f64;

    /// Time to expiry T in years.
    fn expiry(&self) -> f64;

    /// Human-readable model name (e.g. "SVI", "SABR", "CubicSpline").
    fn model_name(&self) -> &'static str;

    /// Check whether this smile is free of butterfly arbitrage.
    ///
    /// Uses model-specific defaults for scan grid. Override this or
    /// [`is_arbitrage_free_with`](SmileSection::is_arbitrage_free_with)
    /// for custom grid parameters.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        self.is_arbitrage_free_with(&ArbitrageScanConfig::svi_default())
    }

    /// Check butterfly arbitrage with custom scan grid configuration.
    ///
    /// Default implementation uses density-based detection (Breeden-Litzenberger)
    /// over `config.n_points` equally-spaced log-moneyness points in
    /// `[config.k_min, config.k_max]`. Models with analytical g-functions
    /// (SVI, SSVI) override this for better accuracy.
    fn is_arbitrage_free_with(
        &self,
        config: &ArbitrageScanConfig,
    ) -> error::Result<ArbitrageReport> {
        config.validate()?;
        let fwd = self.forward();
        let n = config.n_points;
        let mut violations = Vec::new();
        for i in 0..n {
            let k = config.k_min + (config.k_max - config.k_min) * (i as f64) / ((n - 1) as f64);
            let strike = fwd * k.exp();
            let d = match self.density(Strike(strike)) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if d < -DENSITY_NEG_TOL {
                violations.push(ButterflyViolation {
                    strike,
                    density: d,
                    magnitude: d.abs(),
                });
            }
        }
        Ok(ArbitrageReport {
            expiry: self.expiry(),
            butterfly_violations: violations,
        })
    }
}
