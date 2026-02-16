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

#![allow(dead_code)] // Stub — not yet implemented (v0.2 scope)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::arbitrage::ArbitrageReport;
use crate::smile::SmileSection;
use crate::types::Vol;
use crate::validate::{validate_non_negative, validate_positive};

/// SABR volatility smile with 4 parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SabrSmile {
    forward: f64,
    expiry: f64,
    /// ATM vol scale α > 0.
    alpha: f64,
    /// CEV exponent β ∈ \[0, 1\].
    beta: f64,
    /// Spot-vol correlation ρ ∈ (−1, 1).
    rho: f64,
    /// Vol-of-vol ν ≥ 0 (ν = 0 reduces to CEV model).
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
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        validate_positive(alpha, "alpha")?;

        if !(0.0..=1.0).contains(&beta) {
            return Err(VolSurfError::InvalidInput {
                message: format!("beta must be in [0, 1], got {beta}"),
            });
        }

        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("rho must be in (-1, 1), got {rho}"),
            });
        }

        validate_non_negative(nu, "nu")?;

        Ok(Self {
            forward,
            expiry,
            alpha,
            beta,
            rho,
            nu,
        })
    }

    /// Returns the alpha (ATM vol scale) parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the beta (CEV exponent) parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the rho (spot-vol correlation) parameter.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Returns the nu (vol-of-vol) parameter.
    pub fn nu(&self) -> f64 {
        self.nu
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
        Err(VolSurfError::CalibrationError {
            message: "not yet implemented".to_string(),
            model: "SABR",
            rms_error: None,
        })
    }
}

impl SmileSection for SabrSmile {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        let _ = strike;
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical test parameters
    const F: f64 = 100.0;
    const T: f64 = 1.0;
    const ALPHA: f64 = 0.2;
    const BETA: f64 = 0.5;
    const RHO: f64 = -0.3;
    const NU: f64 = 0.4;

    fn make_smile() -> SabrSmile {
        SabrSmile::new(F, T, ALPHA, BETA, RHO, NU).unwrap()
    }

    // --- Valid construction ---

    #[test]
    fn new_valid_params() {
        let s = make_smile();
        assert_eq!(s.forward(), F);
        assert_eq!(s.expiry(), T);
        assert_eq!(s.alpha(), ALPHA);
        assert_eq!(s.beta(), BETA);
        assert_eq!(s.rho(), RHO);
        assert_eq!(s.nu(), NU);
    }

    #[test]
    fn new_positive_rho() {
        let s = SabrSmile::new(F, T, ALPHA, BETA, 0.5, NU).unwrap();
        assert_eq!(s.rho(), 0.5);
    }

    // --- Edge cases that must succeed ---

    #[test]
    fn new_beta_zero() {
        // Normal SABR (beta=0)
        let s = SabrSmile::new(F, T, ALPHA, 0.0, RHO, NU).unwrap();
        assert_eq!(s.beta(), 0.0);
    }

    #[test]
    fn new_beta_one() {
        // Lognormal SABR (beta=1)
        let s = SabrSmile::new(F, T, ALPHA, 1.0, RHO, NU).unwrap();
        assert_eq!(s.beta(), 1.0);
    }

    #[test]
    fn new_nu_zero() {
        // CEV limit (no stochastic vol)
        let s = SabrSmile::new(F, T, ALPHA, BETA, RHO, 0.0).unwrap();
        assert_eq!(s.nu(), 0.0);
    }

    #[test]
    fn new_rho_near_plus_one() {
        let s = SabrSmile::new(F, T, ALPHA, BETA, 0.999, NU).unwrap();
        assert_eq!(s.rho(), 0.999);
    }

    #[test]
    fn new_rho_near_minus_one() {
        let s = SabrSmile::new(F, T, ALPHA, BETA, -0.999, NU).unwrap();
        assert_eq!(s.rho(), -0.999);
    }

    #[test]
    fn new_rho_zero() {
        let s = SabrSmile::new(F, T, ALPHA, BETA, 0.0, NU).unwrap();
        assert_eq!(s.rho(), 0.0);
    }

    // --- Invalid forward ---

    #[test]
    fn new_rejects_zero_forward() {
        let r = SabrSmile::new(0.0, T, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_forward() {
        let r = SabrSmile::new(-1.0, T, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_forward() {
        let r = SabrSmile::new(f64::NAN, T, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_forward() {
        let r = SabrSmile::new(f64::INFINITY, T, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Invalid expiry ---

    #[test]
    fn new_rejects_zero_expiry() {
        let r = SabrSmile::new(F, 0.0, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_expiry() {
        let r = SabrSmile::new(F, -1.0, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_expiry() {
        let r = SabrSmile::new(F, f64::NAN, ALPHA, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Invalid alpha ---

    #[test]
    fn new_rejects_zero_alpha() {
        let r = SabrSmile::new(F, T, 0.0, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_alpha() {
        let r = SabrSmile::new(F, T, -0.1, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_alpha() {
        let r = SabrSmile::new(F, T, f64::NAN, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_alpha() {
        let r = SabrSmile::new(F, T, f64::INFINITY, BETA, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Invalid beta ---

    #[test]
    fn new_rejects_negative_beta() {
        let r = SabrSmile::new(F, T, ALPHA, -0.1, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_beta_above_one() {
        let r = SabrSmile::new(F, T, ALPHA, 1.1, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_beta() {
        let r = SabrSmile::new(F, T, ALPHA, f64::NAN, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_beta() {
        let r = SabrSmile::new(F, T, ALPHA, f64::INFINITY, RHO, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Invalid rho ---

    #[test]
    fn new_rejects_rho_plus_one() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, 1.0, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_rho_minus_one() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, -1.0, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_rho_above_one() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, 1.5, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_rho() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, f64::NAN, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_rho() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, f64::INFINITY, NU);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Invalid nu ---

    #[test]
    fn new_rejects_negative_nu() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, RHO, -0.1);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_nu() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, RHO, f64::NAN);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_nu() {
        let r = SabrSmile::new(F, T, ALPHA, BETA, RHO, f64::INFINITY);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Accessors from SmileSection ---

    #[test]
    fn smile_section_forward_and_expiry() {
        let s = make_smile();
        assert_eq!(SmileSection::forward(&s), F);
        assert_eq!(SmileSection::expiry(&s), T);
    }

    // --- Serde round-trip ---

    #[test]
    fn serde_round_trip() {
        let s = make_smile();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SabrSmile = serde_json::from_str(&json).unwrap();
        assert_eq!(s.alpha(), s2.alpha());
        assert_eq!(s.beta(), s2.beta());
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.nu(), s2.nu());
    }
}
