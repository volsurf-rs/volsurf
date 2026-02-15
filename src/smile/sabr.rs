//! SABR (Stochastic Alpha Beta Rho) smile model.
//!
//! SABR models the forward price as a CEV process with stochastic volatility:
//!
//! ```text
//! dF = σ · F^β · dW₁
//! dσ = ν · σ · dW₂
//! dW₁·dW₂ = ρ dt
//! ```
//!
//! The Hagan formula provides a closed-form approximation for Black implied
//! volatility as a function of strike.
//!
//! # References
//! - Hagan, P. et al. "Managing Smile Risk" (2002)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::arbitrage::ArbitrageReport;
use crate::smile::SmileSection;
use crate::types::Vol;

/// SABR volatility smile with 4 parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SabrSmile {
    forward: f64,
    expiry: f64,
    /// ATM vol scale α > 0.
    alpha: f64,
    /// CEV exponent β ∈ \[0, 1\].
    beta: f64,
    /// Spot-vol correlation ρ ∈ \[−1, 1\].
    rho: f64,
    /// Vol-of-vol ν > 0.
    nu: f64,
}

impl SabrSmile {
    /// Create a SABR smile from calibrated parameters.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if parameters are out of range.
    pub fn new(
        forward: f64,
        expiry: f64,
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
    ) -> error::Result<Self> {
        let _ = (forward, expiry, alpha, beta, rho, nu);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    /// Calibrate SABR parameters from market (strike, vol) observations.
    ///
    /// # Errors
    /// Returns [`VolSurfError::CalibrationError`] if the optimizer fails to converge.
    pub fn calibrate(
        forward: f64,
        expiry: f64,
        market_vols: &[(f64, f64)],
    ) -> error::Result<Self> {
        let _ = (forward, expiry, market_vols);
        Err(VolSurfError::CalibrationError(
            "not yet implemented".to_string(),
        ))
    }
}

impl SmileSection for SabrSmile {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        let _ = strike;
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    fn density(&self, strike: f64) -> error::Result<f64> {
        let _ = strike;
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}
