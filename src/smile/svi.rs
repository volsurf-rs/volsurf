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

use nalgebra::{DMatrix, DVector};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
use crate::types::Vol;
use crate::validate::validate_positive;

/// SVI volatility smile with 5 parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SviSmileRaw", into = "SviSmileRaw")]
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

#[derive(Serialize, Deserialize)]
struct SviSmileRaw {
    forward: f64,
    expiry: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
}

impl TryFrom<SviSmileRaw> for SviSmile {
    type Error = VolSurfError;
    fn try_from(raw: SviSmileRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.forward,
            raw.expiry,
            raw.a,
            raw.b,
            raw.rho,
            raw.m,
            raw.sigma,
        )
    }
}

impl From<SviSmile> for SviSmileRaw {
    fn from(s: SviSmile) -> Self {
        Self {
            forward: s.forward,
            expiry: s.expiry,
            a: s.a,
            b: s.b,
            rho: s.rho,
            m: s.m,
            sigma: s.sigma,
        }
    }
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
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        if b < 0.0 || b.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("b must be non-negative, got {b}"),
            });
        }
        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho| must be less than 1, got {rho}"),
            });
        }
        if sigma <= 0.0 || sigma.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("sigma must be positive, got {sigma}"),
            });
        }
        if !m.is_finite() {
            return Err(VolSurfError::InvalidInput {
                message: format!("m must be finite, got {m}"),
            });
        }
        if !a.is_finite() {
            return Err(VolSurfError::InvalidInput {
                message: format!("a must be finite, got {a}"),
            });
        }
        let min_variance = a + b * sigma * (1.0 - rho * rho).sqrt();
        if min_variance < 0.0 || min_variance.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "minimum variance is negative: a + b*sigma*sqrt(1-rho^2) = {min_variance}"
                ),
            });
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

    /// Skew parameter.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Calibrate SVI parameters from market (strike, vol) observations.
    ///
    /// Uses the Zeliade (2009) quasi-explicit method: for fixed (m, σ),
    /// the remaining parameters (a, b·ρ, b) enter linearly and are solved
    /// via least-squares. A grid search + Nelder-Mead optimizes (m, σ).
    ///
    /// # Arguments
    /// * `forward` — Forward price (must be > 0)
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `market_vols` — Slice of (strike, implied_vol) pairs (min 5)
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for insufficient data,
    /// [`VolSurfError::CalibrationError`] if the optimizer fails.
    ///
    /// # References
    /// - Zeliade Systems, "Quasi-Explicit Calibration of Gatheral's SVI Model" (2009)
    pub fn calibrate(forward: f64, expiry: f64, market_vols: &[(f64, f64)]) -> error::Result<Self> {
        #[cfg(feature = "logging")]
        tracing::debug!(
            forward,
            expiry,
            n_quotes = market_vols.len(),
            "SVI calibration started"
        );

        /// Minimum market quotes for SVI calibration (5 free params).
        const MIN_POINTS: usize = 5;
        /// Grid search resolution for (m, sigma) initialization.
        const GRID_N: usize = 15;
        /// Nelder-Mead iteration limit.
        const NM_MAX_ITER: usize = 300;
        /// Simplex diameter convergence threshold.
        const NM_DIAMETER_TOL: f64 = 1e-8;
        /// Objective value spread convergence threshold.
        const NM_FVALUE_TOL: f64 = 1e-12;

        // Input validation
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        if market_vols.len() < MIN_POINTS {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "at least {MIN_POINTS} market points required, got {}",
                    market_vols.len()
                ),
            });
        }
        for &(strike, vol) in market_vols {
            validate_positive(strike, "strike")?;
            validate_positive(vol, "implied vol")?;
        }

        // Convert to log-moneyness / total-variance
        let k_vals: Vec<f64> = market_vols
            .iter()
            .map(|&(s, _)| (s / forward).ln())
            .collect();
        let w_vals: Vec<f64> = market_vols.iter().map(|&(_, v)| v * v * expiry).collect();

        let k_min = k_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let k_max = k_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let k_range = (k_max - k_min).max(0.1);

        // Inner linear solve: for fixed (m, sigma), solve for (a, b*rho, b)
        let inner_solve = |m: f64, sigma: f64| -> Option<(f64, f64, f64, f64)> {
            let n = k_vals.len();
            let a_mat = DMatrix::<f64>::from_fn(n, 3, |i, j| {
                let dk = k_vals[i] - m;
                match j {
                    0 => 1.0,
                    1 => dk,
                    2 => (dk * dk + sigma * sigma).sqrt(),
                    _ => unreachable!(),
                }
            });
            let b_vec = DVector::from_column_slice(&w_vals);

            let ata = a_mat.transpose() * &a_mat;
            let atb = a_mat.transpose() * &b_vec;
            let x = ata.qr().solve(&atb)?;

            let residual = &a_mat * &x - &b_vec;
            let rss = residual.dot(&residual);
            Some((x[0], x[1], x[2], rss)) // (a, b_rho, b, rss)
        };

        // Objective function: RSS with penalty for invalid params
        let objective = |m: f64, sigma: f64| -> f64 {
            if sigma <= 0.0 {
                return f64::MAX;
            }
            match inner_solve(m, sigma) {
                None => f64::MAX,
                Some((a, b_rho, b, rss)) => {
                    if b < -1e-10 {
                        return f64::MAX;
                    }
                    let b_clamped = b.max(0.0);
                    let rho = if b_clamped < 1e-10 {
                        0.0
                    } else {
                        (b_rho / b_clamped).clamp(-0.999, 0.999)
                    };
                    let min_var = a + b_clamped * sigma * (1.0 - rho * rho).sqrt();
                    if min_var < -1e-10 {
                        return f64::MAX;
                    }
                    rss
                }
            }
        };

        // Grid search
        let m_lo = k_min - 0.5 * k_range;
        let m_hi = k_max + 0.5 * k_range;
        let sigma_lo = 0.01_f64;
        let sigma_hi = k_range.max(0.5);

        let mut best_m = 0.0;
        let mut best_sigma = 0.1;
        let mut best_rss = f64::MAX;

        for im in 0..GRID_N {
            let m = m_lo + (m_hi - m_lo) * (im as f64) / ((GRID_N - 1) as f64);
            for is in 0..GRID_N {
                let sigma = sigma_lo + (sigma_hi - sigma_lo) * (is as f64) / ((GRID_N - 1) as f64);
                let rss = objective(m, sigma);
                if rss < best_rss {
                    best_rss = rss;
                    best_m = m;
                    best_sigma = sigma;
                }
            }
        }

        if best_rss >= f64::MAX {
            return Err(VolSurfError::CalibrationError {
                message: "grid search found no valid starting point".into(),
                model: "SVI",
                rms_error: None,
            });
        }

        // Nelder-Mead 2D optimization over (m, sigma)
        let step_m = (m_hi - m_lo) / (GRID_N as f64) * 0.5;
        let step_s = ((sigma_hi - sigma_lo) / (GRID_N as f64) * 0.5).max(0.001);

        let nm_config = crate::optim::NelderMeadConfig {
            max_iter: NM_MAX_ITER,
            diameter_tol: NM_DIAMETER_TOL,
            fvalue_tol: NM_FVALUE_TOL,
        };
        let nm_result =
            crate::optim::nelder_mead_2d(objective, best_m, best_sigma, step_m, step_s, &nm_config);

        // Recover final parameters
        let (opt_m, opt_sigma) = (nm_result.x, nm_result.y);

        let (a, b_rho, b, _rss) =
            inner_solve(opt_m, opt_sigma).ok_or_else(|| VolSurfError::CalibrationError {
                message: "linear solve failed at optimal (m, sigma)".into(),
                model: "SVI",
                rms_error: None,
            })?;

        let b = b.max(0.0);
        let rho = if b < 1e-10 {
            0.0
        } else {
            (b_rho / b).clamp(-0.999, 0.999)
        };

        #[cfg(feature = "logging")]
        tracing::debug!(
            a,
            b,
            rho,
            m = opt_m,
            sigma = opt_sigma,
            "SVI calibration complete"
        );

        Self::new(forward, expiry, a, b, rho, opt_m, opt_sigma.max(1e-6)).map_err(|e| {
            VolSurfError::CalibrationError {
                message: format!("calibrated params invalid: {e}"),
                model: "SVI",
                rms_error: None,
            }
        })
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
        validate_positive(strike, "strike")?;
        let k = (strike / self.forward).ln();
        let w = self.total_variance_at_k(k);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SVI total variance is negative: w({k}) = {w}"),
            });
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
        validate_positive(strike, "strike")?;
        let k = (strike / self.forward).ln();
        let w = self.total_variance_at_k(k);
        if w <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("SVI total variance is non-positive at k={k}: w={w}"),
            });
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
    /// Points where g(k) < −tol are recorded as [`ButterflyViolation`]s
    /// with the actual risk-neutral density q(K) = g(k)·n(d₂)/(K·√w).
    ///
    /// # Reference
    /// Gatheral & Jacquier (2014), Theorem 4.1.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        /// Number of grid points for g-function arbitrage scan.
        const N: usize = 200;
        /// Minimum log-moneyness for arbitrage scan.
        const K_MIN: f64 = -3.0;
        /// Maximum log-moneyness for arbitrage scan.
        const K_MAX: f64 = 3.0;
        /// Tolerance for g-function negativity detection.
        const TOL: f64 = 1e-10;

        let mut violations = Vec::new();

        for i in 0..N {
            let k = K_MIN + (K_MAX - K_MIN) * (i as f64) / ((N - 1) as f64);
            let g = self.g_function(k);
            if g < -TOL {
                let strike = self.forward * k.exp();
                let d = match self.density(strike) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                violations.push(ButterflyViolation {
                    strike,
                    density: d,
                    magnitude: d.abs(),
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

    #[test]
    fn new_valid_params() {
        let smile = SviSmile::new(F, T, A, B, RHO, M, SIGMA);
        assert!(smile.is_ok());
    }

    #[test]
    fn new_rejects_negative_forward() {
        let r = SviSmile::new(-1.0, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_zero_forward() {
        let r = SviSmile::new(0.0, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_forward() {
        let r = SviSmile::new(f64::NAN, T, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_zero_expiry() {
        let r = SviSmile::new(F, 0.0, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_expiry() {
        let r = SviSmile::new(F, -1.0, A, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_b() {
        let r = SviSmile::new(F, T, A, -0.1, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
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
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_rho_at_neg1() {
        let r = SviSmile::new(F, T, A, B, -1.0, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_rho_above_1() {
        let r = SviSmile::new(F, T, A, B, 1.5, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_zero_sigma() {
        let r = SviSmile::new(F, T, A, B, RHO, M, 0.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_sigma() {
        let r = SviSmile::new(F, T, A, B, RHO, M, -0.1);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_negative_min_variance() {
        // a = -1.0 makes a + b*sigma*sqrt(1-rho^2) < 0
        let r = SviSmile::new(F, T, -1.0, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_allows_zero_min_variance() {
        // Exactly zero minimum variance is allowed
        let min_var = B * SIGMA * (1.0 - RHO * RHO).sqrt();
        let r = SviSmile::new(F, T, -min_var, B, RHO, M, SIGMA);
        assert!(r.is_ok());
    }

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

    #[test]
    fn vol_rejects_zero_strike() {
        let smile = make_smile();
        let r = smile.vol(0.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn vol_rejects_negative_strike() {
        let smile = make_smile();
        let r = smile.vol(-10.0);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn variance_consistent_with_vol() {
        let smile = make_smile();
        let vol = smile.vol(100.0).unwrap();
        let var = smile.variance(100.0).unwrap();
        // variance = vol^2 * T
        assert_abs_diff_eq!(var.0, vol.0 * vol.0 * T, epsilon = 1e-14);
    }

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

    #[test]
    fn forward_and_expiry_accessors() {
        let smile = make_smile();
        assert_eq!(smile.forward(), F);
        assert_eq!(smile.expiry(), T);
    }

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
            Err(VolSurfError::InvalidInput { .. })
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
    fn violation_density_matches_density_method() {
        let smile = SviSmile::new(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05).unwrap();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(!report.butterfly_violations.is_empty());
        for v in &report.butterfly_violations {
            let expected = smile.density(v.strike).unwrap();
            assert_abs_diff_eq!(v.density, expected, epsilon = 1e-14);
            assert_abs_diff_eq!(v.magnitude, expected.abs(), epsilon = 1e-14);
        }
    }

    #[test]
    fn flat_smile_is_arb_free() {
        let smile = SviSmile::new(100.0, 1.0, 0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(report.is_free);
    }

    #[test]
    fn g_function_at_atm_known_value() {
        let smile = make_smile();
        let g0 = smile.g_function(0.0);
        // a=0.04, b=0.4, rho=-0.4, m=0, sigma=0.1
        // w(0)=0.08, w'(0)=-0.16, w''(0)=4.0, term1=1.0
        // g = 1 - 0.0256/4*(12.5+0.25) + 2.0 = 2.9184
        assert_abs_diff_eq!(g0, 2.9184, epsilon = 1e-10);
    }

    fn vol_rms(a: &SviSmile, b: &SviSmile, strikes: &[f64]) -> f64 {
        (strikes
            .iter()
            .map(|&k| {
                let diff = a.vol(k).unwrap().0 - b.vol(k).unwrap().0;
                diff * diff
            })
            .sum::<f64>()
            / strikes.len() as f64)
            .sqrt()
    }

    /// Generate synthetic market data from known SVI params.
    fn synthetic_market_data(smile: &SviSmile, strikes: &[f64]) -> Vec<(f64, f64)> {
        strikes
            .iter()
            .map(|&k| (k, smile.vol(k).unwrap().0))
            .collect()
    }

    #[test]
    fn calibrate_round_trip_canonical() {
        let original = make_smile();
        let strikes: Vec<f64> = (0..20).map(|i| 60.0 + 4.0 * i as f64).collect();
        let data = synthetic_market_data(&original, &strikes);

        let calibrated = SviSmile::calibrate(F, T, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_round_trip_skewed() {
        let original = SviSmile::new(100.0, 0.5, 0.02, 0.6, -0.6, 0.05, 0.15).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let calibrated =
            SviSmile::calibrate(100.0, 0.5, &synthetic_market_data(&original, &strikes)).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_round_trip_symmetric() {
        let original = SviSmile::new(100.0, 1.0, 0.04, 0.3, 0.0, 0.0, 0.2).unwrap();
        let strikes: Vec<f64> = (0..20).map(|i| 60.0 + 4.0 * i as f64).collect();
        let calibrated =
            SviSmile::calibrate(100.0, 1.0, &synthetic_market_data(&original, &strikes)).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_round_trip_non_uniform_strikes() {
        let original = make_smile();
        let strikes = vec![
            60.0, 70.0, 80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0,
            115.0, 120.0, 130.0, 140.0,
        ];
        let calibrated =
            SviSmile::calibrate(F, T, &synthetic_market_data(&original, &strikes)).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_rejects_too_few_points() {
        let data = vec![(90.0, 0.2), (100.0, 0.2), (110.0, 0.2)];
        let result = SviSmile::calibrate(100.0, 1.0, &data);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_negative_forward() {
        let data = vec![
            (80.0, 0.2),
            (90.0, 0.2),
            (100.0, 0.2),
            (110.0, 0.2),
            (120.0, 0.2),
        ];
        let result = SviSmile::calibrate(-100.0, 1.0, &data);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_negative_vol() {
        let data = vec![
            (80.0, 0.2),
            (90.0, -0.1),
            (100.0, 0.2),
            (110.0, 0.2),
            (120.0, 0.2),
        ];
        let result = SviSmile::calibrate(100.0, 1.0, &data);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_grid_search_fails_frown() {
        // SVI w(k) = a + b[rho*dk + sqrt(dk^2 + sigma^2)] is always convex in k.
        // A concave "frown" (high ATM, low wings) forces b < 0 or min_var < 0
        // for every grid point, so the grid search finds no valid starting point.
        let data = vec![
            (60.0, 0.001),
            (80.0, 0.001),
            (90.0, 0.40),
            (100.0, 0.50),
            (110.0, 0.40),
            (120.0, 0.001),
            (140.0, 0.001),
        ];
        let result = SviSmile::calibrate(100.0, 1.0, &data);
        let err = result.unwrap_err();
        assert!(
            matches!(err, VolSurfError::CalibrationError { model: "SVI", .. }),
            "expected SVI CalibrationError, got: {err}"
        );
        assert!(err.to_string().contains("grid search"));
    }

    #[test]
    fn calibration_error_format_svi_linear_solve() {
        let err = VolSurfError::CalibrationError {
            message: "linear solve failed at optimal (m, sigma)".into(),
            model: "SVI",
            rms_error: None,
        };
        assert!(err.to_string().contains("linear solve failed"));
    }

    #[test]
    fn calibrate_params_pass_new_validation() {
        let original = make_smile();
        let strikes: Vec<f64> = (0..20).map(|i| 60.0 + 4.0 * i as f64).collect();
        let data = synthetic_market_data(&original, &strikes);
        // calibrate() internally calls new(), so if it succeeds the params are valid
        let calibrated = SviSmile::calibrate(F, T, &data).unwrap();
        assert!(calibrated.forward() > 0.0);
        assert!(calibrated.expiry() > 0.0);
    }

    #[test]
    fn calibrate_exact_5_points() {
        // Minimum data: exactly 5 points
        let original = make_smile();
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let data = synthetic_market_data(&original, &strikes);
        let calibrated = SviSmile::calibrate(F, T, &data).unwrap();
        let rms = (data
            .iter()
            .map(|&(k, sigma)| {
                let fitted = calibrated.vol(k).unwrap().0;
                (fitted - sigma).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64)
            .sqrt();
        assert!(
            rms < 0.001,
            "round-trip RMS {rms} with 5 points should be < 0.001"
        );
    }

    // Gap #4: m and a parameter validation

    #[test]
    fn new_rejects_nan_m() {
        let r = SviSmile::new(F, T, A, B, RHO, f64::NAN, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_m() {
        let r = SviSmile::new(F, T, A, B, RHO, f64::INFINITY, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_neg_inf_m() {
        let r = SviSmile::new(F, T, A, B, RHO, f64::NEG_INFINITY, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_inf_a() {
        let r = SviSmile::new(F, T, f64::INFINITY, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn new_rejects_nan_a() {
        let r = SviSmile::new(F, T, f64::NAN, B, RHO, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
    }

    // Gap #2: vol() negative variance error path

    /// Construct an SVI with invalid params (bypassing new() validation)
    /// to test error paths that require negative total variance.
    fn make_invalid_svi() -> SviSmile {
        // Direct struct construction is possible within the same module.
        // a = -0.1 makes total variance negative at ATM (k=0):
        // w(0) = a + b*(rho*0 + sqrt(0 + sigma^2)) = -0.1 + 0.01*0.1 = -0.099
        SviSmile {
            forward: 100.0,
            expiry: 1.0,
            a: -0.1,
            b: 0.01,
            rho: 0.0,
            m: 0.0,
            sigma: 0.1,
        }
    }

    #[test]
    fn vol_returns_error_when_total_variance_is_negative() {
        let smile = make_invalid_svi();
        let result = smile.vol(100.0);
        assert!(
            matches!(result, Err(VolSurfError::NumericalError { .. })),
            "vol() should return NumericalError for negative variance, got {result:?}"
        );
    }

    // Gap #3: density() non-positive variance error path

    #[test]
    fn density_returns_error_when_total_variance_non_positive() {
        let smile = make_invalid_svi();
        let result = smile.density(100.0);
        assert!(
            matches!(result, Err(VolSurfError::NumericalError { .. })),
            "density() should return NumericalError for non-positive variance, got {result:?}"
        );
    }

    #[test]
    fn serde_round_trip() {
        let s = make_smile();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SviSmile = serde_json::from_str(&json).unwrap();
        assert_eq!(SmileSection::forward(&s), SmileSection::forward(&s2));
        assert_eq!(SmileSection::expiry(&s), SmileSection::expiry(&s2));
        assert_eq!(s.rho(), s2.rho());
        let json2 = serde_json::to_string(&s2).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn serde_rejects_negative_forward() {
        let json =
            r#"{"forward":-100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":-0.4,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_expiry() {
        let json =
            r#"{"forward":100.0,"expiry":0.0,"a":0.04,"b":0.4,"rho":-0.4,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_b() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":-0.1,"rho":-0.4,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_plus_one() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":1.0,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_minus_one() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":-1.0,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_sigma() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":-0.4,"m":0.0,"sigma":0.0}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_sigma() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":-0.4,"m":0.0,"sigma":-0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_inf_m() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":0.04,"b":0.4,"rho":-0.4,"m":1e999,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_inf_a() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":1e999,"b":0.4,"rho":-0.4,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_min_variance_violation() {
        let json =
            r#"{"forward":100.0,"expiry":1.0,"a":-1.0,"b":0.4,"rho":-0.4,"m":0.0,"sigma":0.1}"#;
        assert!(serde_json::from_str::<SviSmile>(json).is_err());
    }
}
