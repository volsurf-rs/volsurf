//! Bachelier (normal) implied volatility.
//!
//! Used in markets where the normal model is standard (short-dated FX, fixed income).
//! The Bachelier formula assumes arithmetic (Gaussian) returns rather than lognormal.
//!
//! # Formula
//! ```text
//! C(F, K, T, σ) = (F - K)·N(d) + σ·√T·n(d)
//! where d = (F - K) / (σ√T)
//! ```

use crate::error::VolSurfError;
use crate::types::{OptionType, Vol};

/// Bachelier (normal) implied volatility calculator.
///
/// Extracts implied volatility under the normal model using root-finding.
#[derive(Debug)]
pub struct NormalImpliedVol;

impl NormalImpliedVol {
    /// Compute normal (Bachelier) implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Market price of the vanilla option
    /// * `forward` — Forward price at expiry
    /// * `strike` — Strike price
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `option_type` — Call or Put
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for invalid inputs,
    /// [`VolSurfError::NumericalError`] if the solver cannot converge.
    pub fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: OptionType,
    ) -> crate::error::Result<Vol> {
        let _ = (option_price, forward, strike, expiry, option_type);
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }
}
