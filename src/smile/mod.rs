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

use crate::error;
use crate::implied::black::black_price;
use crate::types::{OptionType, Variance, Vol};
use crate::validate::validate_positive;

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
/// # Examples
///
/// ```
/// use volsurf::smile::{SabrSmile, SmileSection};
///
/// // Create a concrete smile and use it through the trait interface
/// let smile = SabrSmile::new(100.0, 1.0, 0.3, 1.0, -0.3, 0.4)?;
///
/// let vol = smile.vol(100.0)?;
/// assert!(vol.0 > 0.0);
///
/// let var = smile.variance(100.0)?;
/// assert!((var.0 - vol.0 * vol.0 * smile.expiry()).abs() < 1e-12);
///
/// let report = smile.is_arbitrage_free()?;
/// assert!(report.is_free);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
pub trait SmileSection: Send + Sync + std::fmt::Debug {
    /// Implied Black volatility σ at the given strike.
    fn vol(&self, strike: f64) -> error::Result<Vol>;

    /// Total Black variance σ²T at the given strike.
    ///
    /// Default implementation derives from [`vol`](SmileSection::vol):
    /// `variance(K) = vol(K)² × expiry`.
    fn variance(&self, strike: f64) -> error::Result<Variance> {
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
    fn density(&self, strike: f64) -> error::Result<f64> {
        validate_positive(strike, "strike")?;
        // Relative perturbation for central finite difference
        let h = strike * 1e-4;
        let k_lo = strike - h;
        let k_hi = strike + h;

        let v_lo = self.vol(k_lo)?;
        let v_mid = self.vol(strike)?;
        let v_hi = self.vol(k_hi)?;

        let c_lo = black_price(self.forward(), k_lo, v_lo.0, self.expiry(), OptionType::Call)?;
        let c_mid =
            black_price(self.forward(), strike, v_mid.0, self.expiry(), OptionType::Call)?;
        let c_hi = black_price(self.forward(), k_hi, v_hi.0, self.expiry(), OptionType::Call)?;

        // Breeden-Litzenberger: q(K) = d²C/dK² (undiscounted)
        Ok((c_lo - 2.0 * c_mid + c_hi) / (h * h))
    }

    /// Forward price F at this tenor.
    fn forward(&self) -> f64;

    /// Time to expiry T in years.
    fn expiry(&self) -> f64;

    /// Check whether this smile is free of butterfly arbitrage.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport>;
}
