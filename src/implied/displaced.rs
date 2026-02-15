//! Displaced diffusion implied volatility.
//!
//! A hybrid model that interpolates between the normal model (β=0) and
//! the Black model (β=1) via a displacement parameter β ∈ \[0, 1\].

#![allow(dead_code)] // Stub — not yet implemented (v0.2+ scope)

use crate::error::VolSurfError;
use crate::types::{OptionType, Vol};

/// Displaced diffusion implied volatility calculator.
///
/// The displacement parameter `beta` controls the blend:
/// - β = 0 → pure normal (Bachelier) model
/// - β = 1 → pure Black (lognormal) model
/// - 0 < β < 1 → intermediate CEV-like behavior
pub struct DisplacedImpliedVol {
    /// Displacement parameter β ∈ [0, 1].
    beta: f64,
}

impl DisplacedImpliedVol {
    /// Create a new displaced diffusion calculator with the given displacement.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if `beta` is not in \[0, 1\].
    pub fn new(beta: f64) -> crate::error::Result<Self> {
        if !(0.0..=1.0).contains(&beta) {
            return Err(VolSurfError::InvalidInput {
                message: format!("beta must be in [0, 1], got {beta}"),
            });
        }
        Ok(Self { beta })
    }

    /// Compute displaced diffusion implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Market price of the vanilla option
    /// * `forward` — Forward price at expiry
    /// * `strike` — Strike price
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `option_type` — Call or Put
    ///
    /// # Errors
    /// Returns [`VolSurfError::NumericalError`] if the solver cannot converge.
    pub fn compute(
        &self,
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: OptionType,
    ) -> crate::error::Result<Vol> {
        let _ = (self.beta, option_price, forward, strike, expiry, option_type);
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }
}
