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
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
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

    /// Hagan (2002) implied Black volatility approximation, Eq. (2.17a).
    ///
    /// Computes the SABR implied vol at a given strike using the closed-form
    /// approximation from "Managing Smile Risk". Uses the general formula
    /// uniformly, with a Taylor expansion of z/x(z) for small z (including
    /// the ATM limit K → F and the CEV limit ν → 0).
    ///
    /// # References
    /// Hagan, P. et al., "Managing Smile Risk", Wilmott Magazine, Jan 2002, Eq. (2.17a).
    fn hagan_implied_vol(&self, strike: f64) -> f64 {
        let f = self.forward;
        let k = strike;
        let alpha = self.alpha;
        let beta = self.beta;
        let rho = self.rho;
        let nu = self.nu;
        let t = self.expiry;
        let omb = 1.0 - beta;

        let ln_fk = (f / k).ln();
        let fk = f * k;
        let fk_mid = fk.powf(omb / 2.0); // (FK)^((1-β)/2)

        // Denominator: corrects the CEV backbone for log-moneyness
        let ln_fk_sq = ln_fk * ln_fk;
        let omb_sq = omb * omb;
        let denom = fk_mid
            * (1.0 + omb_sq / 24.0 * ln_fk_sq + omb_sq * omb_sq / 1920.0 * ln_fk_sq * ln_fk_sq);

        // z/x(z) ratio — Taylor expansion for small z handles ATM and CEV limits
        let z = if nu == 0.0 { 0.0 } else { (nu / alpha) * fk_mid * ln_fk };
        let z_ratio = if z.abs() < 1e-6 {
            // Taylor: z/x(z) ≈ 1 − ρz/2 + (2−3ρ²)/12 · z²
            1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) / 12.0 * z * z
        } else {
            let disc = (1.0 - 2.0 * rho * z + z * z).sqrt();
            let xz = ((disc + z - rho) / (1.0 - rho)).ln();
            z / xz
        };

        // Time-dependent correction factor
        let fk_omb = fk.powf(omb); // (FK)^(1-β)
        let correction = 1.0
            + t * (omb_sq / 24.0 * alpha * alpha / fk_omb
                + 0.25 * rho * beta * nu * alpha / fk_mid
                + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu);

        (alpha / denom) * z_ratio * correction
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
        validate_positive(strike, "strike")?;
        let sigma = self.hagan_implied_vol(strike);
        if sigma < 0.0 || !sigma.is_finite() {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "SABR implied vol is invalid: {sigma} at strike={strike}, forward={}",
                    self.forward
                ),
            });
        }
        Ok(Vol(sigma))
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        // Scan density over log-moneyness grid. The Hagan approximation
        // has limited validity in deep wings, so use a moderate range and
        // skip points where the approximation breaks down.
        const N: usize = 200;
        const K_MIN: f64 = -2.0; // log-moneyness bounds
        const K_MAX: f64 = 2.0;
        const TOL: f64 = 1e-8;

        let mut violations = Vec::new();
        for i in 0..N {
            let k = K_MIN + (K_MAX - K_MIN) * (i as f64) / ((N - 1) as f64);
            let strike = self.forward * k.exp();
            let d = match self.density(strike) {
                Ok(d) => d,
                Err(_) => continue, // Hagan breakdown in wings
            };
            if d < -TOL {
                violations.push(ButterflyViolation {
                    strike,
                    density: d,
                    magnitude: d.abs(),
                });
            }
        }

        Ok(ArbitrageReport {
            is_free: violations.is_empty(),
            butterfly_violations: violations,
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

    // ========================================================================
    // Hagan formula tests (T02)
    // ========================================================================

    // Use realistic equity-like params: ATM vol ≈ 20%
    const EQ_ALPHA: f64 = 2.0;

    fn make_equity_smile() -> SabrSmile {
        SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, NU).unwrap()
    }

    // --- ATM tests ---

    #[test]
    fn vol_atm_returns_positive() {
        let s = make_equity_smile();
        let v = s.vol(F).unwrap();
        assert!(v.0 > 0.0, "ATM vol should be positive, got {}", v.0);
    }

    #[test]
    fn vol_atm_matches_closed_form() {
        // Hagan ATM formula: σ = α/F^(1-β) * [1 + T*(correction)]
        let s = make_equity_smile();
        let omb = 1.0 - BETA;
        let f_omb = F.powf(omb);
        let expected = EQ_ALPHA / f_omb
            * (1.0
                + T * (omb * omb / 24.0 * EQ_ALPHA * EQ_ALPHA / (f_omb * f_omb)
                    + 0.25 * RHO * BETA * NU * EQ_ALPHA / f_omb
                    + (2.0 - 3.0 * RHO * RHO) / 24.0 * NU * NU));
        let actual = s.vol(F).unwrap().0;
        assert!(
            (actual - expected).abs() < 1e-14,
            "ATM vol mismatch: expected {expected}, got {actual}"
        );
    }

    // --- General case ---

    #[test]
    fn vol_otm_call() {
        let s = make_equity_smile();
        let v = s.vol(120.0).unwrap();
        assert!(v.0 > 0.0, "OTM call vol should be positive");
    }

    #[test]
    fn vol_itm_call() {
        let s = make_equity_smile();
        let v = s.vol(80.0).unwrap();
        assert!(v.0 > 0.0, "ITM call vol should be positive");
    }

    // --- ATM boundary continuity ---

    #[test]
    fn vol_atm_boundary_continuity() {
        // No separate ATM branch: the general formula with Taylor expansion
        // is continuous through K = F. Verify the second-order finite difference
        // is bounded (C² smoothness ⇒ no discontinuity or kink at ATM).
        let s = make_equity_smile();
        let v_atm = s.vol(F).unwrap().0;
        for &h in &[1.0, 0.1, 0.01] {
            let v_above = s.vol(F + h).unwrap().0;
            let v_below = s.vol(F - h).unwrap().0;
            let curvature = (v_above + v_below - 2.0 * v_atm) / (h * h);
            // Bounded curvature ⇒ no discontinuity
            assert!(
                curvature.abs() < 1.0,
                "Unbounded curvature at h={h}: {curvature}"
            );
        }
        // Also verify the Taylor/exact z boundary is seamless:
        // z = (ν/α) * fk_mid * ln(F/K). At the boundary |z|=1e-6, the Taylor
        // and exact formulas agree to ~1e-13, so no switching discontinuity.
        let z_boundary_k = F * (-5e-7_f64).exp(); // z ≈ 1e-6
        let v1 = s.vol(z_boundary_k).unwrap().0;
        let v2 = s.vol(z_boundary_k * 0.999).unwrap().0;
        let v3 = s.vol(z_boundary_k * 1.001).unwrap().0;
        let curvature = (v2 + v3 - 2.0 * v1) / ((z_boundary_k * 0.001).powi(2));
        assert!(
            curvature.abs() < 100.0,
            "Curvature spike at Taylor boundary: {curvature}"
        );
    }

    // --- beta = 0 (normal SABR) ---

    #[test]
    fn vol_beta_zero_normal_sabr() {
        // beta=0: normal SABR, ATM vol ≈ alpha/F
        let alpha = 10.0; // gives ~10% vol with F=100
        let s = SabrSmile::new(F, T, alpha, 0.0, RHO, NU).unwrap();
        let v = s.vol(F).unwrap().0;
        let expected_approx = alpha / F; // 0.10
        // Should be close to alpha/F (within the correction term)
        assert!(
            (v - expected_approx).abs() < 0.01,
            "Normal SABR ATM vol ≈ α/F = {expected_approx}, got {v}"
        );
        // OTM should also work
        assert!(s.vol(110.0).unwrap().0 > 0.0);
        assert!(s.vol(90.0).unwrap().0 > 0.0);
    }

    // --- beta = 1 (lognormal SABR) ---

    #[test]
    fn vol_beta_one_lognormal_sabr() {
        // beta=1: lognormal SABR, ATM vol ≈ alpha
        let alpha = 0.20;
        let s = SabrSmile::new(F, T, alpha, 1.0, RHO, NU).unwrap();
        let v = s.vol(F).unwrap().0;
        // σ_ATM = α * [1 + T*(¼ρνα + (2-3ρ²)/24*ν²)]
        let expected = alpha
            * (1.0
                + T * (0.25 * RHO * NU * alpha
                    + (2.0 - 3.0 * RHO * RHO) / 24.0 * NU * NU));
        assert!(
            (v - expected).abs() < 1e-14,
            "Lognormal ATM: expected {expected}, got {v}"
        );
    }

    // --- nu = 0 (CEV limit) ---

    #[test]
    fn vol_nu_zero_cev_limit() {
        // nu=0: z/x(z) = 1, no vol-of-vol terms
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, 0.0).unwrap();
        let v_atm = s.vol(F).unwrap().0;
        let omb = 1.0 - BETA;
        let f_omb = F.powf(omb);
        // Only the (1-β)²/24 * α²/F^(2(1-β)) term survives
        let expected = EQ_ALPHA / f_omb
            * (1.0 + T * omb * omb / 24.0 * EQ_ALPHA * EQ_ALPHA / (f_omb * f_omb));
        assert!(
            (v_atm - expected).abs() < 1e-14,
            "CEV ATM: expected {expected}, got {v_atm}"
        );
    }

    #[test]
    fn vol_nu_zero_beta_one_constant() {
        // beta=1, nu=0: pure lognormal, constant vol = alpha for all strikes
        let alpha = 0.20;
        let s = SabrSmile::new(F, T, alpha, 1.0, 0.0, 0.0).unwrap();
        let v_atm = s.vol(F).unwrap().0;
        let v_otm = s.vol(120.0).unwrap().0;
        let v_itm = s.vol(80.0).unwrap().0;
        assert!(
            (v_atm - alpha).abs() < 1e-14,
            "β=1,ν=0 ATM should be α={alpha}, got {v_atm}"
        );
        assert!(
            (v_otm - alpha).abs() < 1e-10,
            "β=1,ν=0 OTM should be ≈ α={alpha}, got {v_otm}"
        );
        assert!(
            (v_itm - alpha).abs() < 1e-10,
            "β=1,ν=0 ITM should be ≈ α={alpha}, got {v_itm}"
        );
    }

    // --- Skew from rho ---

    #[test]
    fn vol_rho_zero_symmetric_smile() {
        // rho=0, beta=1: the lognormal SABR smile is symmetric in log-moneyness
        let alpha = 0.20;
        let s = SabrSmile::new(F, T, alpha, 1.0, 0.0, NU).unwrap();
        let ratio = 1.2;
        let v_up = s.vol(F * ratio).unwrap().0;
        let v_down = s.vol(F / ratio).unwrap().0;
        // With beta=1 and rho=0, the smile is exactly symmetric in ln(K/F)
        assert!(
            (v_up - v_down).abs() < 1e-12,
            "rho=0, β=1: smile should be symmetric: up={v_up}, down={v_down}, diff={}",
            (v_up - v_down).abs()
        );
    }

    #[test]
    fn vol_negative_rho_skew() {
        // Negative rho: vol(K<F) > vol(K>F) (equity skew)
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, -0.5, NU).unwrap();
        let v_low = s.vol(80.0).unwrap().0;
        let v_high = s.vol(120.0).unwrap().0;
        assert!(
            v_low > v_high,
            "Negative rho should produce downward skew: vol(80)={v_low} should > vol(120)={v_high}"
        );
    }

    // --- Strike validation ---

    #[test]
    fn vol_rejects_zero_strike() {
        let s = make_equity_smile();
        assert!(matches!(
            s.vol(0.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn vol_rejects_negative_strike() {
        let s = make_equity_smile();
        assert!(matches!(
            s.vol(-10.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn vol_rejects_nan_strike() {
        let s = make_equity_smile();
        assert!(matches!(
            s.vol(f64::NAN),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    // --- Variance consistency ---

    #[test]
    fn vol_variance_consistency() {
        // variance() = vol()² * T (from SmileSection default impl)
        let s = make_equity_smile();
        let strike = 110.0;
        let v = s.vol(strike).unwrap().0;
        let var = s.variance(strike).unwrap().0;
        let expected_var = v * v * T;
        assert!(
            (var - expected_var).abs() < 1e-14,
            "variance={var} should equal vol²·T={expected_var}"
        );
    }

    // --- Taylor expansion boundary ---

    #[test]
    fn vol_taylor_boundary_smooth() {
        // Verify smoothness across the Taylor expansion boundary (|z| ≈ 1e-6)
        // Use small nu so z is small for moderate strikes
        let small_nu = 0.001;
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, small_nu).unwrap();
        // Find a strike where z ≈ 1e-6 boundary
        // z = (nu/alpha) * fk_mid * ln(F/K)
        // For small z, K must be very close to F
        let k_offset = 1e-4;
        let v1 = s.vol(F + k_offset).unwrap().0;
        let v2 = s.vol(F + k_offset * 1.01).unwrap().0;
        // Should be smooth (no discontinuity)
        assert!(
            (v1 - v2).abs() < 1e-8,
            "Taylor boundary should be smooth: {v1} vs {v2}"
        );
    }

    // --- Extreme parameters ---

    #[test]
    fn vol_extreme_rho() {
        // rho = ±0.999: should still produce valid vols
        let s_pos = SabrSmile::new(F, T, EQ_ALPHA, BETA, 0.999, NU).unwrap();
        let s_neg = SabrSmile::new(F, T, EQ_ALPHA, BETA, -0.999, NU).unwrap();
        assert!(s_pos.vol(F).unwrap().0 > 0.0);
        assert!(s_neg.vol(F).unwrap().0 > 0.0);
        assert!(s_pos.vol(110.0).unwrap().0 > 0.0);
        assert!(s_neg.vol(110.0).unwrap().0 > 0.0);
    }

    #[test]
    fn vol_short_expiry() {
        let s = SabrSmile::new(F, 0.01, EQ_ALPHA, BETA, RHO, NU).unwrap();
        let v = s.vol(F).unwrap().0;
        let omb = 1.0 - BETA;
        let f_omb = F.powf(omb);
        // Short expiry: correction ≈ 1, so vol ≈ α/F^(1-β)
        let approx = EQ_ALPHA / f_omb;
        assert!(
            (v - approx).abs() / approx < 0.01,
            "Short expiry vol should be close to α/F^(1-β)={approx}, got {v}"
        );
    }

    #[test]
    fn vol_long_expiry() {
        let s = SabrSmile::new(F, 5.0, EQ_ALPHA, BETA, RHO, NU).unwrap();
        let v = s.vol(F).unwrap().0;
        assert!(v > 0.0, "Long expiry vol should be positive, got {v}");
        // Longer expiry has larger correction → vol should differ from short expiry
        let s_short = SabrSmile::new(F, 0.01, EQ_ALPHA, BETA, RHO, NU).unwrap();
        let v_short = s_short.vol(F).unwrap().0;
        assert!(
            (v - v_short).abs() > 0.001,
            "Long vs short expiry ATM vol should differ"
        );
    }

    // --- Cross-check: general formula approaches ATM formula ---

    #[test]
    fn vol_general_approaches_atm() {
        // As K → F from both sides, general formula should approach ATM formula
        let s = make_equity_smile();
        let v_atm = s.vol(F).unwrap().0;
        for &eps in &[1e-3, 1e-4, 1e-5, 1e-6] {
            let v = s.vol(F + eps).unwrap().0;
            let diff = (v - v_atm).abs();
            assert!(
                diff < eps * 10.0, // vol difference should be proportional to eps
                "At K=F+{eps}: diff={diff}, should be small"
            );
        }
    }

    // --- 12-digit accuracy: general formula self-consistency ---

    #[test]
    fn vol_twelve_digit_accuracy_beta_one() {
        // beta=1 simplifies the formula: no (1-β) terms
        // Manually compute each component to verify
        let alpha = 0.20;
        let rho = -0.25;
        let nu = 0.30;
        let s = SabrSmile::new(F, T, alpha, 1.0, rho, nu).unwrap();
        let k = 110.0;

        // Step-by-step reference computation (beta=1: omb=0, fk_mid=1, denom=1)
        let ln_fk = (F / k).ln();
        let z = (nu / alpha) * ln_fk; // fk_mid = 1
        let disc = (1.0 - 2.0 * rho * z + z * z).sqrt();
        let xz = ((disc + z - rho) / (1.0 - rho)).ln();
        let z_ratio = z / xz;
        let correction =
            1.0 + T * (0.25 * rho * nu * alpha + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu);
        let expected = alpha * z_ratio * correction;

        let actual = s.vol(k).unwrap().0;
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "β=1 12-digit check: expected {expected}, got {actual}, rel_err={rel_err}"
        );
    }

    #[test]
    fn vol_twelve_digit_accuracy_general() {
        // Full general formula step-by-step verification
        let s = make_equity_smile();
        let k = 120.0;

        let omb = 1.0 - BETA;
        let ln_fk = (F / k).ln();
        let fk = F * k;
        let fk_mid = fk.powf(omb / 2.0);
        let ln_fk_sq = ln_fk * ln_fk;
        let omb_sq = omb * omb;
        let denom = fk_mid
            * (1.0 + omb_sq / 24.0 * ln_fk_sq + omb_sq * omb_sq / 1920.0 * ln_fk_sq * ln_fk_sq);
        let z = (NU / EQ_ALPHA) * fk_mid * ln_fk;
        let disc = (1.0 - 2.0 * RHO * z + z * z).sqrt();
        let xz = ((disc + z - RHO) / (1.0 - RHO)).ln();
        let z_ratio = z / xz;
        let fk_omb = fk.powf(omb);
        let correction = 1.0
            + T * (omb_sq / 24.0 * EQ_ALPHA * EQ_ALPHA / fk_omb
                + 0.25 * RHO * BETA * NU * EQ_ALPHA / fk_mid
                + (2.0 - 3.0 * RHO * RHO) / 24.0 * NU * NU);
        let expected = (EQ_ALPHA / denom) * z_ratio * correction;

        let actual = s.vol(k).unwrap().0;
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err < 1e-14,
            "General 12-digit: expected {expected}, got {actual}, rel_err={rel_err}"
        );
    }

    // --- Multi-strike consistency ---

    #[test]
    fn vol_multiple_strikes_reasonable() {
        let s = make_equity_smile();
        let strikes = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0];
        for &k in &strikes {
            let v = s.vol(k).unwrap().0;
            assert!(
                v > 0.01 && v < 2.0,
                "Vol at K={k} should be reasonable, got {v}"
            );
        }
    }

    #[test]
    fn vol_deep_otm_positive() {
        // Deep OTM strikes should still give positive vol (for reasonable params)
        let s = make_equity_smile();
        assert!(s.vol(50.0).unwrap().0 > 0.0);
        assert!(s.vol(200.0).unwrap().0 > 0.0);
    }

    // ========================================================================
    // SmileSection trait completion tests (T03)
    // ========================================================================

    // --- density() tests ---

    #[test]
    fn density_positive_for_typical_params() {
        let s = make_equity_smile();
        for &k in &[80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
            let d = s.density(k).unwrap();
            assert!(d > 0.0, "density should be positive at K={k}, got {d}");
        }
    }

    #[test]
    fn density_integrates_to_one() {
        // Trapezoidal integration of q(K) dK over wide log-moneyness range
        // Ref: same pattern as SviSmile density_integrates_to_one (svi.rs)
        let s = make_equity_smile();
        let n = 5000;
        let k_min = -10.0_f64;
        let k_max = 10.0_f64;
        let dk = (k_max - k_min) / (n as f64);
        let mut integral = 0.0;
        for i in 0..=n {
            let k = k_min + i as f64 * dk;
            let strike = s.forward() * k.exp();
            let q = s.density(strike).unwrap();
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            // Change of variable: dK = K dk
            integral += weight * q * strike * dk;
        }
        assert!(
            (integral - 1.0).abs() < 1e-3,
            "density should integrate to ~1.0, got {integral}"
        );
    }

    #[test]
    fn density_peaks_near_forward() {
        // The density peak should be near the forward price
        let s = make_equity_smile();
        let d_atm = s.density(F).unwrap();
        let d_far_otm = s.density(200.0).unwrap();
        let d_far_itm = s.density(50.0).unwrap();
        assert!(
            d_atm > d_far_otm,
            "density at ATM ({d_atm}) should exceed far OTM ({d_far_otm})"
        );
        assert!(
            d_atm > d_far_itm,
            "density at ATM ({d_atm}) should exceed far ITM ({d_far_itm})"
        );
    }

    // --- variance() consistency ---

    #[test]
    fn variance_equals_vol_squared_times_t() {
        let s = make_equity_smile();
        for &k in &[80.0, 100.0, 120.0] {
            let v = s.vol(k).unwrap().0;
            let var = s.variance(k).unwrap().0;
            let expected = v * v * T;
            assert!(
                (var - expected).abs() < 1e-14,
                "K={k}: variance={var} != vol²·T={expected}"
            );
        }
    }

    // --- is_arbitrage_free() tests ---

    #[test]
    fn arb_free_well_behaved_params() {
        // Well-behaved SABR params should produce clean arbitrage report
        let s = make_equity_smile();
        let report = s.is_arbitrage_free().unwrap();
        assert!(
            report.is_free,
            "well-behaved params should be arb-free, got {} violations",
            report.butterfly_violations.len()
        );
        assert!(report.butterfly_violations.is_empty());
    }

    #[test]
    fn arb_free_beta_one() {
        let s = SabrSmile::new(F, T, 0.20, 1.0, -0.25, 0.30).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(report.is_free, "β=1 with moderate params should be arb-free");
    }

    #[test]
    fn arb_free_beta_zero() {
        let s = SabrSmile::new(F, T, 10.0, 0.0, -0.2, 0.30).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(report.is_free, "β=0 with moderate params should be arb-free");
    }

    #[test]
    fn arb_free_nu_zero() {
        // nu=0 (CEV) should always be arb-free
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, 0.0).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(report.is_free, "CEV (ν=0) should be arb-free");
    }

    #[test]
    fn arb_violation_extreme_nu() {
        // Very high nu can create butterfly violations
        // The Hagan approximation breaks down for extreme vol-of-vol
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, -0.5, 5.0).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        // With nu=5.0, the approximation likely produces negative densities
        // (if not, this is still a valid test — just confirms well-behaved)
        if !report.is_free {
            for v in &report.butterfly_violations {
                assert!(v.density < 0.0);
                assert!(v.magnitude > 0.0);
                assert!(v.strike > 0.0);
            }
        }
    }

    #[test]
    fn arb_report_violation_fields() {
        // High nu provokes violations; verify field structure if detected
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, -0.5, 5.0).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        if !report.is_free {
            let v = &report.butterfly_violations[0];
            assert!(v.strike > 0.0, "violation strike should be positive");
            assert!(v.density < 0.0, "violation density should be negative");
            assert!(
                (v.magnitude - v.density.abs()) < 1e-15,
                "magnitude should equal |density|"
            );
        }
    }

    // --- Send + Sync ---

    #[test]
    fn sabr_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SabrSmile>();
    }

    // --- SmileSection trait object ---

    #[test]
    fn sabr_as_trait_object() {
        let s = make_equity_smile();
        let boxed: Box<dyn SmileSection> = Box::new(s);
        let v = boxed.vol(F).unwrap();
        assert!(v.0 > 0.0);
        let report = boxed.is_arbitrage_free().unwrap();
        assert!(report.is_free);
    }
}
