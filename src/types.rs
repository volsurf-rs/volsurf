//! Core domain types for volatility surface construction.
//!
//! These newtypes wrap `f64` to provide compile-time type safety, preventing
//! accidental parameter swapping (e.g., passing a strike where a tenor is expected).
//!
//! # Newtype Strategy
//!
//! **Outputs use newtypes** — [`Vol`], [`Variance`], [`Strike`], [`Tenor`] wrap
//! return values so callers can't accidentally mix a volatility with a variance.
//!
//! **Inputs use bare `f64`** — API methods like `vol(strike: f64)` accept raw
//! floats for ergonomics. Requiring `vol(Strike(100.0))` at every call site adds
//! ceremony without meaningful safety (the caller already knows they're passing
//! a strike). This is a deliberate trade-off: newtypes guard against *silent*
//! misuse of outputs, while inputs are self-documenting via parameter names.
//!
//! # Why no `Eq` or `Ord`?
//! These types wrap `f64`, which does not implement `Eq` or `Ord` because `NaN`
//! breaks total ordering. We derive `PartialEq` and `PartialOrd` only. Do not
//! add `Eq` without handling `NaN` explicitly.

use serde::{Deserialize, Serialize};

/// Strike price `K` of an option contract.
///
/// # Examples
/// ```
/// use volsurf::types::Strike;
/// let strike = Strike(100.0);
/// assert_eq!(strike.0, 100.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Strike(pub f64);

/// Time to expiry `T` in years (annualized).
///
/// A tenor of 0.25 represents 3 months, 1.0 represents 1 year.
///
/// # Examples
/// ```
/// use volsurf::types::Tenor;
/// let three_months = Tenor(0.25);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Tenor(pub f64);

/// Implied volatility `σ`, measured as annualized standard deviation.
///
/// A vol of 0.20 represents 20% annualized volatility.
///
/// # Examples
/// ```
/// use volsurf::types::Vol;
/// let vol = Vol(0.20);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Vol(pub f64);

/// Total variance `σ²T` or instantaneous variance `σ²`.
///
/// Variance is the square of volatility. Cross-tenor interpolation is performed
/// in variance space because total variance must be non-decreasing in time
/// for arbitrage-free surfaces.
///
/// # Examples
/// ```
/// use volsurf::types::Variance;
/// let var = Variance(0.04); // corresponds to 20% vol
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Variance(pub f64);

/// Option type: call or put.
///
/// Used in implied volatility extraction to determine the pricing formula branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    /// Right to buy at strike price.
    Call,
    /// Right to sell at strike price.
    Put,
}
