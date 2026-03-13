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

use crate::error;
use crate::implied::black::black_price;
use crate::types::{OptionType, Strike, Variance, Vol};
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
/// assert!(report.is_free);
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

    /// Check whether this smile is free of butterfly arbitrage.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport>;
}
