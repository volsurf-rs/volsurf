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

use std::f64::consts::PI;

use crate::error::{self, VolSurfError};
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
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

    /// First derivative of total variance: w'(k) = b·[ρ + (k−m)/√((k−m)² + σ²)].
    fn w_prime(&self, k: f64) -> f64 {
        let dk = k - self.m;
        let r = (dk * dk + self.sigma * self.sigma).sqrt();
        self.b * (self.rho + dk / r)
    }

    /// Second derivative of total variance: w''(k) = b·σ²/((k−m)² + σ²)^(3/2).
    fn w_double_prime(&self, k: f64) -> f64 {
        let dk = k - self.m;
        let r2 = dk * dk + self.sigma * self.sigma;
        self.b * self.sigma * self.sigma / (r2 * r2.sqrt())
    }

    /// Gatheral g-function for butterfly arbitrage detection.
    ///
    /// g(k) ≥ 0 everywhere implies no butterfly arbitrage.
    ///
    /// # Reference
    /// Gatheral & Jacquier (2014), Definition 4.1.
    fn g_function(&self, k: f64) -> f64 {
        let w = self.total_variance_at_k(k);
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let wp = self.w_prime(k);
        let wpp = self.w_double_prime(k);
        let term1 = 1.0 - k * wp / (2.0 * w);
        term1 * term1 - wp * wp / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
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

    /// Risk-neutral density q(K) via the Gatheral g-function.
    ///
    /// Uses the analytical formula:
    /// ```text
    /// q(K) = g(k) · n(d₂) / (K · √w)
    /// ```
    /// where k = ln(K/F), d₂ = −k/√w − √w/2, and n(·) is the standard
    /// normal PDF.
    ///
    /// # Reference
    /// Breeden & Litzenberger (1978); Gatheral & Jacquier (2014), §4.
    fn density(&self, strike: f64) -> error::Result<f64> {
        if strike <= 0.0 || strike.is_nan() {
            return Err(VolSurfError::InvalidInput(format!(
                "strike must be positive, got {strike}"
            )));
        }
        let k = (strike / self.forward).ln();
        let w = self.total_variance_at_k(k);
        if w <= 0.0 {
            return Err(VolSurfError::NumericalError(format!(
                "SVI total variance is non-positive at k={k}: w={w}"
            )));
        }
        let g = self.g_function(k);
        let sqrt_w = w.sqrt();
        let d2 = -k / sqrt_w - sqrt_w / 2.0;
        let n_d2 = (-d2 * d2 / 2.0).exp() / (2.0 * PI).sqrt();
        Ok(g * n_d2 / (strike * sqrt_w))
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    /// Check butterfly arbitrage by scanning the Gatheral g-function.
    ///
    /// Evaluates g(k) on a grid of 200 points over k ∈ \[−3, 3\].
    /// Points where g(k) < −tol are recorded as [`ButterflyViolation`]s.
    ///
    /// # Reference
    /// Gatheral & Jacquier (2014), Theorem 4.1.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        const N: usize = 200;
        const K_MIN: f64 = -3.0;
        const K_MAX: f64 = 3.0;
        const TOL: f64 = 1e-10;

        let mut violations = Vec::new();

        for i in 0..N {
            let k = K_MIN + (K_MAX - K_MIN) * (i as f64) / ((N - 1) as f64);
            let g = self.g_function(k);
            if g < -TOL {
                violations.push(ButterflyViolation {
                    strike: self.forward * k.exp(),
                    density: g,
                    magnitude: g.abs(),
                });
            }
        }

        if violations.is_empty() {
            Ok(ArbitrageReport::clean())
        } else {
            Ok(ArbitrageReport {
                is_free: false,
                butterfly_violations: violations,
            })
        }
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

    // --- density() tests ---

    fn make_arb_free_smile() -> SviSmile {
        // Wider sigma for clean arb-free behavior
        SviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2).unwrap()
    }

    #[test]
    fn density_atm_positive() {
        let smile = make_arb_free_smile();
        let d = smile.density(100.0).unwrap();
        assert!(d > 0.0, "ATM density should be positive, got {d}");
    }

    #[test]
    fn density_non_negative_arb_free() {
        let smile = make_arb_free_smile();
        let strikes: Vec<f64> = (1..=200)
            .map(|i| 100.0 * ((-3.0 + 6.0 * i as f64 / 200.0).exp()))
            .collect();
        for &k in &strikes {
            let d = smile.density(k).unwrap();
            assert!(
                d >= -1e-15,
                "density should be non-negative for arb-free params at K={k}, got {d}"
            );
        }
    }

    #[test]
    fn density_positive_across_strikes() {
        // For symmetric arb-free SVI, density should be positive everywhere
        let smile = SviSmile::new(100.0, 1.0, 0.04, 0.4, 0.0, 0.0, 0.2).unwrap();
        for &strike in &[50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0] {
            let d = smile.density(strike).unwrap();
            assert!(d > 0.0, "density should be positive at K={strike}, got {d}");
        }
    }

    #[test]
    fn density_integrates_to_one() {
        let smile = make_arb_free_smile();
        let n = 5000;
        let k_min = -10.0_f64;
        let k_max = 10.0_f64;
        let dk = (k_max - k_min) / (n as f64);
        let mut integral = 0.0;
        for i in 0..=n {
            let k = k_min + i as f64 * dk;
            let strike = smile.forward() * k.exp();
            let q = smile.density(strike).unwrap();
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            // q(K) dK = q(K) * K * dk  (change of variable: dK = K dk)
            integral += weight * q * strike * dk;
        }
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn density_rejects_zero_strike() {
        let smile = make_arb_free_smile();
        assert!(matches!(
            smile.density(0.0),
            Err(VolSurfError::InvalidInput(_))
        ));
    }

    #[test]
    fn density_cross_check_breeden_litzenberger() {
        // Verify analytical density matches numerical d^2C/dK^2
        use crate::implied::black_price;
        use crate::types::OptionType;

        let smile = make_arb_free_smile();
        let f = smile.forward();
        let t = smile.expiry();

        for &strike in &[80.0, 100.0, 120.0] {
            let h = strike * 1e-4;
            let vol_m = smile.vol(strike - h).unwrap().0;
            let vol_0 = smile.vol(strike).unwrap().0;
            let vol_p = smile.vol(strike + h).unwrap().0;
            let c_m = black_price(f, strike - h, vol_m, t, OptionType::Call).unwrap();
            let c_0 = black_price(f, strike, vol_0, t, OptionType::Call).unwrap();
            let c_p = black_price(f, strike + h, vol_p, t, OptionType::Call).unwrap();
            let numerical = (c_p - 2.0 * c_0 + c_m) / (h * h);
            let analytical = smile.density(strike).unwrap();
            assert_abs_diff_eq!(analytical, numerical, epsilon = 1e-4);
        }
    }

    // --- is_arbitrage_free() tests ---

    #[test]
    fn arb_free_params_clean_report() {
        let smile = make_arb_free_smile();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(report.is_free, "expected arb-free report");
        assert!(report.butterfly_violations.is_empty());
    }

    #[test]
    fn violated_params_detect_violations() {
        // Aggressive slope with small curvature: g(k) goes negative in wings
        let smile = SviSmile::new(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05).unwrap();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(!report.is_free, "expected violations");
        assert!(
            !report.butterfly_violations.is_empty(),
            "expected non-empty violations"
        );
        for v in &report.butterfly_violations {
            assert!(v.density < 0.0, "violation density should be negative");
            assert!(v.magnitude > 0.0, "violation magnitude should be positive");
            assert!(v.strike > 0.0, "violation strike should be positive");
        }
    }

    #[test]
    fn flat_smile_is_arb_free() {
        let smile = SviSmile::new(100.0, 1.0, 0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(report.is_free);
    }
}
