//! Black (lognormal) implied volatility via Jäckel's "Let's Be Rational" algorithm.
//!
//! Wraps the [`implied_vol`] crate which achieves 3 ULP accuracy.

use crate::error::VolSurfError;
use crate::types::{OptionType, Vol};

/// Black (lognormal) implied volatility calculator.
///
/// Uses Peter Jäckel's rational approximation algorithm (2013) for
/// near-machine-precision extraction of Black implied volatility.
///
/// # References
/// - Jäckel, P. "Let's Be Rational" (2013)
pub struct BlackImpliedVol;

impl BlackImpliedVol {
    /// Compute Black implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Market price of the vanilla option (must be > 0)
    /// * `forward` — Forward price at expiry (must be > 0)
    /// * `strike` — Strike price (must be > 0)
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `option_type` — Call or Put
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for non-positive inputs,
    /// [`VolSurfError::NumericalError`] if the algorithm cannot converge.
    pub fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: OptionType,
    ) -> crate::error::Result<Vol> {
        let _ = (option_price, forward, strike, expiry, option_type);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}
