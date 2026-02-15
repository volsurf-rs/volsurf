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
use crate::types::{Variance, Vol};

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
pub trait SmileSection: Send + Sync {
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
    /// Must be non-negative for an arbitrage-free smile.
    fn density(&self, strike: f64) -> error::Result<f64>;

    /// Forward price F at this tenor.
    fn forward(&self) -> f64;

    /// Time to expiry T in years.
    fn expiry(&self) -> f64;

    /// Check whether this smile is free of butterfly arbitrage.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport>;
}
