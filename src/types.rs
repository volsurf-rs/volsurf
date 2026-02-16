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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gap #1: Serde round-trip ---

    #[test]
    fn strike_serde_round_trip() {
        let s = Strike(123.456);
        let json = serde_json::to_string(&s).unwrap();
        let s2: Strike = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn tenor_serde_round_trip() {
        let t = Tenor(0.25);
        let json = serde_json::to_string(&t).unwrap();
        let t2: Tenor = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn vol_serde_round_trip() {
        let v = Vol(0.20);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Vol = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn variance_serde_round_trip() {
        let v = Variance(0.04);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Variance = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn option_type_call_serde_round_trip() {
        let ot = OptionType::Call;
        let json = serde_json::to_string(&ot).unwrap();
        let ot2: OptionType = serde_json::from_str(&json).unwrap();
        assert_eq!(ot, ot2);
    }

    #[test]
    fn option_type_put_serde_round_trip() {
        let ot = OptionType::Put;
        let json = serde_json::to_string(&ot).unwrap();
        let ot2: OptionType = serde_json::from_str(&json).unwrap();
        assert_eq!(ot, ot2);
    }

    #[test]
    fn serde_preserves_special_float_values() {
        // NaN cannot round-trip via JSON (not valid JSON), but Inf can't either.
        // Verify that normal values survive exactly.
        let values = [0.0, -0.0, 1e-300, 1e300, f64::MIN_POSITIVE];
        for &v in &values {
            let vol = Vol(v);
            let json = serde_json::to_string(&vol).unwrap();
            let vol2: Vol = serde_json::from_str(&json).unwrap();
            assert_eq!(vol.0.to_bits(), vol2.0.to_bits());
        }
    }
}
