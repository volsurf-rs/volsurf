//! SVI (Stochastic Volatility Inspired) smile model.
//!
//! The SVI parameterization models total implied variance as:
//!
//! ```text
//! w(k) = a + b·[ρ(k − m) + √((k − m)² + σ²)]
//! ```
//!
//! where `k = ln(K/F)` is log-moneyness and `(a, b, ρ, m, σ)` are the five
//! SVI parameters.
//!
//! # References
//! - Gatheral, J. "The Volatility Surface: A Practitioner's Guide" (2006)
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::arbitrage::ArbitrageReport;
use crate::smile::SmileSection;
use crate::types::Vol;

/// SVI volatility smile with 5 parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SviSmile {
    forward: f64,
    expiry: f64,
    /// Minimum variance level.
    a: f64,
    /// Variance slope (controls skew magnitude).
    b: f64,
    /// Skew direction ρ ∈ \[−1, 1\].
    rho: f64,
    /// Moneyness shift.
    m: f64,
    /// Curvature (smile convexity).
    sigma: f64,
}

impl SviSmile {
    /// Create an SVI smile from calibrated parameters.
    ///
    /// Validates the Gatheral-Jacquier no-arbitrage conditions:
    /// - `b ≥ 0` (non-negative slope)
    /// - `|ρ| < 1` (strict)
    /// - `σ > 0` (positive curvature)
    /// - `a + bσ√(1 − ρ²) ≥ 0` (non-negative minimum variance)
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if parameters violate
    /// Gatheral-Jacquier no-arbitrage conditions or if forward/expiry
    /// are non-positive.
    pub fn new(
        forward: f64,
        expiry: f64,
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
    ) -> error::Result<Self> {
        if forward <= 0.0 || forward.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "forward must be positive, got {forward}"
            )));
        }
        if expiry <= 0.0 || expiry.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "expiry must be positive, got {expiry}"
            )));
        }
        if b < 0.0 || b.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "b must be non-negative, got {b}"
            )));
        }
        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "|rho| must be less than 1, got {rho}"
            )));
        }
        if sigma <= 0.0 || sigma.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        let min_variance = a + b * sigma * (1.0 - rho * rho).sqrt();
        if min_variance < 0.0 || min_variance.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "minimum variance is negative: a + b*sigma*sqrt(1-rho^2) = {min_variance}"
            )));
        }

        Ok(Self {
            forward,
            expiry,
            a,
            b,
            rho,
            m,
            sigma,
        })
    }

    /// Calibrate SVI parameters from market (strike, vol) observations.
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

    /// Evaluate the raw SVI total variance w(k) at log-moneyness k.
    ///
    /// ```text
    /// w(k) = a + b·[ρ(k − m) + √((k − m)² + σ²)]
    /// ```
    fn total_variance_at_k(&self, k: f64) -> f64 {
        let dk = k - self.m;
        self.a + self.b * (self.rho * dk + (dk * dk + self.sigma * self.sigma).sqrt())
    }
}

impl SmileSection for SviSmile {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        if strike <= 0.0 || strike.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "strike must be positive, got {strike}"
            )));
        }
        let k = (strike / self.forward).ln();
        let w = self.total_variance_at_k(k);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError(format!(
                "SVI total variance is negative: w({k}) = {w}"
            )));
        }
        Ok(Vol((w / self.expiry).sqrt()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Canonical test parameters: equity-like SVI
    const F: f64 = 100.0;
    const T: f64 = 1.0;
    const A: f64 = 0.04;
    const B: f64 = 0.4;
    const RHO: f64 = -0.4;
    const M: f64 = 0.0;
    const SIGMA: f64 = 0.1;

    fn make_smile() -> SviSmile {
        SviSmile::new(F, T, A, B, RHO, M, SIGMA).unwrap()
    }

    // --- Constructor validation ---

    #[test]
    fn new_valid_params() {
        let smile = SviSmile::new(F, T, A, B, RHO, M, SIGMA);
        assert!(smile.is_ok());
    }

    #[test]
    fn new_rejects_negative_forward() {
        let r = SviSmile::new(-1.0, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_zero_forward() {
        let r = SviSmile::new(0.0, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_nan_forward() {
        let r = SviSmile::new(f64::NAN, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_zero_expiry() {
        let r = SviSmile::new(F, 0.0, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_negative_expiry() {
        let r = SviSmile::new(F, -1.0, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_negative_b() {
        let r = SviSmile::new(F, T, A, -0.1, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_allows_zero_b() {
        // b = 0 is valid (flat smile at level a)
        let r = SviSmile::new(F, T, 0.04, 0.0, 0.0, M, SIGMA);
        assert!(r.is_ok());
    }

    #[test]
    fn new_rejects_rho_at_1() {
        let r = SviSmile::new(F, T, A, B, 1.0, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_rho_at_neg1() {
        let r = SviSmile::new(F, T, A, B, -1.0, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_rho_above_1() {
        let r = SviSmile::new(F, T, A, B, 1.5, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_zero_sigma() {
        let r = SviSmile::new(F, T, A, B, RHO, M, 0.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_negative_sigma() {
        let r = SviSmile::new(F, T, A, B, RHO, M, -0.1);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_rejects_negative_min_variance() {
        // a = -1.0 makes a + b*sigma*sqrt(1-rho^2) < 0
        let r = SviSmile::new(F, T, -1.0, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn new_allows_zero_min_variance() {
        // Exactly zero minimum variance is allowed
        let min_var = B * SIGMA * (1.0 - RHO * RHO).sqrt();
        let r = SviSmile::new(F, T, -min_var, B, RHO, M, SIGMA);
        assert!(r.is_ok());
    }

    // --- vol() basic behavior ---

    #[test]
    fn vol_atm() {
        let smile = make_smile();
        let vol = smile.vol(F).unwrap();
        // ATM (k=0, m=0): w = a + b*(rho*0 + sqrt(0 + sigma^2)) = a + b*sigma
        // w = 0.04 + 0.4*0.1 = 0.08, vol = sqrt(0.08/1.0) = 0.28284...
        let expected_w = A + B * SIGMA;
        let expected_vol = (expected_w / T).sqrt();
        assert_abs_diff_eq!(vol.0, expected_vol, epsilon = 1e-14);
    }

    #[test]
    fn vol_known_value_otm_put() {
        let smile = make_smile();
        // Strike 80 => k = ln(80/100) = ln(0.8)
        let k = (80.0_f64 / F).ln();
        let dk = k - M;
        let expected_w = A + B * (RHO * dk + (dk * dk + SIGMA * SIGMA).sqrt());
        let expected_vol = (expected_w / T).sqrt();
        let vol = smile.vol(80.0).unwrap();
        assert_abs_diff_eq!(vol.0, expected_vol, epsilon = 1e-14);
    }

    #[test]
    fn vol_known_value_otm_call() {
        let smile = make_smile();
        // Strike 120 => k = ln(120/100) = ln(1.2)
        let k = (120.0_f64 / F).ln();
        let dk = k - M;
        let expected_w = A + B * (RHO * dk + (dk * dk + SIGMA * SIGMA).sqrt());
        let expected_vol = (expected_w / T).sqrt();
        let vol = smile.vol(120.0).unwrap();
        assert_abs_diff_eq!(vol.0, expected_vol, epsilon = 1e-14);
    }

    // --- ATM symmetry with rho=0, m=0 ---

    #[test]
    fn atm_symmetry_rho_zero() {
        let smile = SviSmile::new(F, T, A, B, 0.0, 0.0, SIGMA).unwrap();
        // Symmetric strikes around forward: K and F^2/K have same vol
        let strikes = [80.0, 90.0, 95.0, 98.0];
        for &k in &strikes {
            let mirror = F * F / k; // mirror strike
            let v1 = smile.vol(k).unwrap();
            let v2 = smile.vol(mirror).unwrap();
            assert_abs_diff_eq!(v1.0, v2.0, epsilon = 1e-14);
        }
    }

    // --- Skew with rho < 0 ---

    #[test]
    fn skew_negative_rho() {
        let smile = make_smile(); // rho = -0.4
        let vol_low = smile.vol(80.0).unwrap();
        let vol_high = smile.vol(120.0).unwrap();
        assert!(
            vol_low.0 > vol_high.0,
            "negative rho should produce higher vol for lower strikes: \
             vol(80)={} should be > vol(120)={}",
            vol_low.0,
            vol_high.0
        );
    }

    #[test]
    fn skew_positive_rho() {
        let smile = SviSmile::new(F, T, A, B, 0.4, M, SIGMA).unwrap();
        let vol_low = smile.vol(80.0).unwrap();
        let vol_high = smile.vol(120.0).unwrap();
        assert!(
            vol_high.0 > vol_low.0,
            "positive rho should produce higher vol for higher strikes"
        );
    }

    // --- vol() error cases ---

    #[test]
    fn vol_rejects_zero_strike() {
        let smile = make_smile();
        let r = smile.vol(0.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    #[test]
    fn vol_rejects_negative_strike() {
        let smile = make_smile();
        let r = smile.vol(-10.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput(_))));
    }

    // --- variance() default impl consistency ---

    #[test]
    fn variance_consistent_with_vol() {
        let smile = make_smile();
        let vol = smile.vol(100.0).unwrap();
        let var = smile.variance(100.0).unwrap();
        // variance = vol^2 * T
        assert_abs_diff_eq!(var.0, vol.0 * vol.0 * T, epsilon = 1e-14);
    }

    // --- b = 0 gives flat smile ---

    #[test]
    fn flat_smile_with_zero_b() {
        let smile = SviSmile::new(F, T, 0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        let vol_80 = smile.vol(80.0).unwrap();
        let vol_100 = smile.vol(100.0).unwrap();
        let vol_120 = smile.vol(120.0).unwrap();
        // b=0 => w(k) = a for all k => vol = sqrt(a/T)
        let expected = (0.04_f64 / T).sqrt();
        assert_abs_diff_eq!(vol_80.0, expected, epsilon = 1e-14);
        assert_abs_diff_eq!(vol_100.0, expected, epsilon = 1e-14);
        assert_abs_diff_eq!(vol_120.0, expected, epsilon = 1e-14);
    }

    // --- forward() and expiry() ---

    #[test]
    fn forward_and_expiry_accessors() {
        let smile = make_smile();
        assert_eq!(smile.forward(), F);
        assert_eq!(smile.expiry(), T);
    }
}
