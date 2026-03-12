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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// - `b(1 + |ρ|) ≤ 2` (Roger Lee moment bound)
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
        let lee_bound = b * (1.0 + rho.abs());
        if lee_bound > 2.0 {
            return Err(VolSurfError::InvalidInput {
                message: format!("Roger Lee bound violated: b*(1+|rho|) = {lee_bound} > 2"),
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
        const GRID_N: usize = 21;

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

        // Vega weights: n(d₁) for each point. ATM options have highest vega and
        // naturally dominate the weighted fit, preventing degenerate basin drift.
        // F·√T cancels across strikes so we use just the normal PDF value.
        let sqrt_t = expiry.sqrt();
        let sqrt_vega: Vec<f64> = market_vols
            .iter()
            .map(|&(strike, vol)| {
                let d1 = (-(strike / forward).ln() + 0.5 * vol * vol * expiry) / (vol * sqrt_t);
                ((-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt())
                    .sqrt()
                    .max(1e-8)
            })
            .collect();

        // Pre-filter: remove vol-cliff artifacts (cabinet-level OTM quotes).
        // Sort by log-moneyness, scan for >50% vol drops between consecutive points,
        // keep the side with more points. Fallback to full data if too few survive.
        let (k_vals, w_vals, sqrt_vega) = {
            let vols: Vec<f64> = market_vols.iter().map(|&(_, v)| v).collect();
            let mut order: Vec<usize> = (0..k_vals.len()).collect();
            order.sort_by(|&a, &b| k_vals[a].total_cmp(&k_vals[b]));

            let mut has_drop = false;
            let mut has_rise = false;
            let mut cliff_idx = None;
            for i in 0..order.len().saturating_sub(1) {
                let v_cur = vols[order[i]];
                let v_next = vols[order[i + 1]];
                if v_next < 0.5 * v_cur {
                    has_drop = true;
                    if cliff_idx.is_none() {
                        cliff_idx = Some(i);
                    }
                }
                if v_next > 2.0 * v_cur {
                    has_rise = true;
                }
            }

            // Both rise and drop means frown/W-shape — don't filter
            if let Some(ci) = cliff_idx.filter(|_| !has_rise || !has_drop) {
                let left_count = ci + 1;
                let right_count = order.len() - left_count;
                let keep: &[usize] = if left_count >= right_count {
                    &order[..left_count]
                } else {
                    &order[left_count..]
                };
                if keep.len() >= MIN_POINTS {
                    let k_f: Vec<f64> = keep.iter().map(|&i| k_vals[i]).collect();
                    let w_f: Vec<f64> = keep.iter().map(|&i| w_vals[i]).collect();
                    let vw_f: Vec<f64> = keep.iter().map(|&i| sqrt_vega[i]).collect();
                    (k_f, w_f, vw_f)
                } else {
                    (k_vals, w_vals, sqrt_vega)
                }
            } else {
                (k_vals, w_vals, sqrt_vega)
            }
        };

        let k_min = k_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let k_max = k_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let k_range = (k_max - k_min).max(0.1);

        // Interpolate ATM total variance from nearest-ATM input points.
        // Only meaningful when data brackets k=0 (both sides of ATM present).
        // One-sided data after vol-cliff filtering produces unreliable ATM estimates.
        let w_atm = {
            let mut best_below = (f64::NEG_INFINITY, 0.0_f64);
            let mut best_above = (f64::INFINITY, 0.0_f64);
            for (&k, &w) in k_vals.iter().zip(w_vals.iter()) {
                if k <= 0.0 && k > best_below.0 {
                    best_below = (k, w);
                }
                if k >= 0.0 && k < best_above.0 {
                    best_above = (k, w);
                }
            }
            let has_both_sides = best_below.0 != f64::NEG_INFINITY && best_above.0 != f64::INFINITY;
            if !has_both_sides {
                None
            } else if (best_above.0 - best_below.0).abs() < 1e-15 {
                Some(best_below.1)
            } else {
                let t = (0.0 - best_below.0) / (best_above.0 - best_below.0);
                Some(best_below.1 + t * (best_above.1 - best_below.1))
            }
        };

        // Inner linear solve: for fixed (m, sigma), solve for (a, b*rho, b)
        // using vega-weighted least squares. Premultiplying rows by √ν_i converts
        // to standard LS on the transformed system (Zeliade QR structure preserved).
        let inner_solve = |m: f64, sigma: f64| -> Option<(f64, f64, f64, f64)> {
            let n = k_vals.len();
            let a_mat = DMatrix::<f64>::from_fn(n, 3, |i, j| {
                let dk = k_vals[i] - m;
                let val = match j {
                    0 => 1.0,
                    1 => dk,
                    2 => (dk * dk + sigma * sigma).sqrt(),
                    _ => unreachable!(),
                };
                val * sqrt_vega[i]
            });
            let b_vec = DVector::from_fn(n, |i, _| w_vals[i] * sqrt_vega[i]);

            let ata = a_mat.transpose() * &a_mat;
            let atb = a_mat.transpose() * &b_vec;
            let x = ata.qr().solve(&atb)?;

            let residual = &a_mat * &x - &b_vec;
            let rss = residual.dot(&residual);
            Some((x[0], x[1], x[2], rss))
        };

        // Bound m to a sensible range around the data. Without this, Nelder-Mead
        // can drift m far from the data range, producing degenerate fits where
        // all points lie on one asymptote (b → ∞, ρ → ±1, m far away).
        // Allow m to range 1.5× the data span beyond each edge.
        let m_bound_lo = k_min - 1.5 * k_range;
        let m_bound_hi = k_max + 1.5 * k_range;

        let objective = |m: f64, sigma: f64| -> f64 {
            if sigma <= 0.0 || m < m_bound_lo || m > m_bound_hi {
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
                    if b_clamped * (1.0 + rho.abs()) > 2.0 {
                        return f64::MAX;
                    }
                    let min_var = a + b_clamped * sigma * (1.0 - rho * rho).sqrt();
                    if min_var < -1e-10 {
                        return f64::MAX;
                    }
                    rss
                }
            }
        };

        // Multi-start grid search: 8 starts with different (m, σ) ranges to escape
        // local minima in the 2D objective landscape (see Zeliade 2009, §3.2).
        // At least 3 starts are k_range-independent to avoid correlated basin drift.
        let mut k_sorted = k_vals.clone();
        k_sorted.sort_by(|a, b| a.total_cmp(b));
        let k_median = k_sorted[k_sorted.len() / 2];
        let sigma_atm = w_atm
            .map(|w| w.max(0.0).sqrt().clamp(0.01, 2.0))
            .unwrap_or(0.2);
        let starts: [(f64, f64, f64, f64); 8] = [
            // Start 0: original wide range (k_range-dependent)
            (
                k_min - 0.5 * k_range,
                k_max + 0.5 * k_range,
                0.01,
                k_range.max(0.5),
            ),
            // Start 1: same m, tighter σ
            (
                k_min - 0.5 * k_range,
                k_max + 0.5 * k_range,
                0.005,
                (k_range / 2.0).max(0.2),
            ),
            // Start 2: ATM-centered, wide σ (fixed range)
            (-0.2, 0.2, 0.01, 1.0),
            // Start 3: ATM-variance-centered — σ anchored to observed ATM level
            (
                -0.15,
                0.15,
                (sigma_atm * 0.3).max(0.005),
                (sigma_atm * 3.0).max(0.3),
            ),
            // Start 4: wide m (±2× data range), moderate σ
            (
                k_min - k_range,
                k_max + k_range,
                0.02,
                (k_range * 0.7).max(0.3),
            ),
            // Start 5: median-k centered m, tight σ for short tenors
            (k_median - 0.1, k_median + 0.1, 0.003, 0.15),
            // Start 6: fixed near-ATM, very tight σ (short-dated equity)
            (-0.05, 0.05, 0.002, 0.08),
            // Start 7: fixed wide m, large σ (long-dated / high-vol regimes)
            (-0.5, 0.5, 0.05, 2.0),
        ];

        let nm_config = crate::optim::NelderMeadConfig::calibration();

        let mut best_m = 0.0;
        let mut best_sigma = 0.1;
        let mut best_rss = f64::MAX;

        for &(m_lo, m_hi, sigma_lo, sigma_hi) in &starts {
            let mut start_m = 0.0;
            let mut start_sigma = 0.1;
            let mut start_rss = f64::MAX;

            for im in 0..GRID_N {
                let m = m_lo + (m_hi - m_lo) * (im as f64) / ((GRID_N - 1) as f64);
                for is in 0..GRID_N {
                    let t = is as f64 / (GRID_N - 1) as f64;
                    let sigma = sigma_lo * (sigma_hi / sigma_lo).powf(t);
                    let rss = objective(m, sigma);
                    if rss < start_rss {
                        start_rss = rss;
                        start_m = m;
                        start_sigma = sigma;
                    }
                }
            }

            if start_rss >= f64::MAX {
                continue;
            }

            let step_m = (m_hi - m_lo) / (GRID_N as f64) * 0.5;
            let step_s =
                (start_sigma * (sigma_hi / sigma_lo).ln() / ((GRID_N - 1) as f64) * 0.5).max(0.001);

            let nm = crate::optim::nelder_mead_2d(
                objective,
                start_m,
                start_sigma,
                step_m,
                step_s,
                &nm_config,
            );

            if nm.fval < best_rss {
                best_rss = nm.fval;
                best_m = nm.x;
                best_sigma = nm.y;
            }
        }

        if best_rss >= f64::MAX {
            return Err(VolSurfError::CalibrationError {
                message: "grid search found no valid starting point".into(),
                model: "SVI",
                rms_error: None,
            });
        }

        // Recover final parameters
        let (opt_m, opt_sigma) = (best_m, best_sigma);

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

        // One-sided data has no reliable ATM reference — skip sanity check
        if w_atm.is_some() {
            let dk_atm = -opt_m;
            let w_atm_fitted =
                a + b * (rho * dk_atm + (dk_atm * dk_atm + opt_sigma * opt_sigma).sqrt());
            let mut w_sorted = w_vals.clone();
            w_sorted.sort_by(|x, y| x.total_cmp(y));
            let w_median = if w_sorted.len() % 2 == 0 {
                (w_sorted[w_sorted.len() / 2 - 1] + w_sorted[w_sorted.len() / 2]) / 2.0
            } else {
                w_sorted[w_sorted.len() / 2]
            };
            if w_atm_fitted < 0.0 || (w_median > 0.0 && w_atm_fitted / w_median > 4.0) {
                return Err(VolSurfError::CalibrationError {
                    message: format!(
                        "ATM total variance {w_atm_fitted:.6} is degenerate (median input {w_median:.6})"
                    ),
                    model: "SVI",
                    rms_error: None,
                });
            }
        }

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
    fn new_rejects_roger_lee_bound() {
        // b=1.5, rho=0.5 → 1.5*(1+0.5) = 2.25 > 2
        let r = SviSmile::new(F, T, 0.5, 1.5, 0.5, M, SIGMA);
        assert!(matches!(r, Err(VolSurfError::InvalidInput { .. })));
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("Roger Lee"), "expected Roger Lee in: {msg}");
    }

    #[test]
    fn new_accepts_roger_lee_boundary() {
        // b=1.25, rho=0.6 → 1.25*(1+0.6) = 2.0 exactly — should be accepted
        let r = SviSmile::new(F, T, 0.01, 1.25, 0.6, M, SIGMA);
        assert!(r.is_ok());
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
    fn is_arbitrage_free_skips_density_errors() {
        // make_invalid_svi has w < 0 everywhere, so g = NEG_INFINITY at all
        // grid points but density() returns Err — violations get skipped.
        let smile = make_invalid_svi();
        let report = smile.is_arbitrage_free().unwrap();
        assert!(report.butterfly_violations.is_empty());
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

    // Real market data tests — ES futures options with cabinet-level OTM call artifacts.
    // Source: vol-arb/scripts/essvi_fixtures/fail_00.json tenor[1], fail_10.json tenor[0].

    #[test]
    fn calibrate_real_data_fail_00_tenor1() {
        // T=0.212, F=6099.69, 40 strikes. Vol cliff at k~+0.002: put vols 19-28%, call vols 7-8%.
        let forward = 6099.694029589387;
        let expiry = 0.21189634192714274;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (6110.0, 0.0730505873), (6170.0, 0.0755070638), (6175.0, 0.0756816599),
            (6180.0, 0.0758177423), (6190.0, 0.0761682847), (6200.0, 0.0764665067),
            (6275.0, 0.0783416580), (6350.0, 0.0797123411),
            (4950.0, 0.2817714778), (5000.0, 0.2752316740), (5025.0, 0.2721449775),
            (5075.0, 0.2660942206), (5080.0, 0.2654167174), (5120.0, 0.2605702346),
            (5140.0, 0.2580952222), (5200.0, 0.2513427505), (5230.0, 0.2480107054),
            (5275.0, 0.2430436076), (5325.0, 0.2378082330), (5350.0, 0.2352714337),
            (5360.0, 0.2342019069), (5470.0, 0.2235805604), (5525.0, 0.2185461412),
            (5625.0, 0.2100749330), (5640.0, 0.2088287216), (5695.0, 0.2045126109),
            (5710.0, 0.2033832903), (5750.0, 0.2003749512), (5800.0, 0.1969502219),
            (5845.0, 0.1941444803), (5855.0, 0.1935502308), (5870.0, 0.1927241764),
            (5875.0, 0.1924352123), (5890.0, 0.1917753903), (5925.0, 0.1902211011),
            (5930.0, 0.1899562582), (5935.0, 0.1897236482), (5990.0, 0.1884446645),
            (6025.0, 0.1879697111), (6045.0, 0.1880807256),
        ];

        // One-sided data after vol-cliff filter: calibration may produce a valid
        // (albeit imprecise) fit, or fail with Roger Lee / ATM sanity rejection.
        match SviSmile::calibrate(forward, expiry, &data) {
            Ok(smile) => {
                let atm_vol = smile.vol(forward).unwrap().0;
                assert!(
                    atm_vol > 0.10 && atm_vol < 1.0,
                    "ATM vol {atm_vol:.4} outside [0.10, 1.0]"
                );
            }
            Err(VolSurfError::CalibrationError { message, .. }) => {
                assert!(
                    message.contains("Roger Lee")
                        || message.contains("ATM total variance")
                        || message.contains("grid search"),
                    "unexpected rejection reason: {message}"
                );
            }
            Err(e) => panic!("unexpected error variant: {e}"),
        }
    }

    #[test]
    fn calibrate_real_data_fail_10_tenor0() {
        // T=0.045, F=6056.25, 41 strikes. Extreme cliff: put vols 39-49%, call vols 3-8%.
        let forward = 6056.245540908916;
        let expiry = 0.04487413491520268;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (6060.0, 0.0332844455), (6065.0, 0.0364709299), (6110.0, 0.0522470818),
            (6170.0, 0.0656003863), (6175.0, 0.0667243682), (6180.0, 0.0676347474),
            (6190.0, 0.0692914683), (6200.0, 0.0710054533), (6275.0, 0.0844261158),
            (4950.0, 0.4949932398), (5000.0, 0.4804532933), (5025.0, 0.4741619460),
            (5075.0, 0.4613972029), (5080.0, 0.4603371389), (5120.0, 0.4503366675),
            (5140.0, 0.4455547253), (5200.0, 0.4321813183), (5230.0, 0.4261146526),
            (5275.0, 0.4180120045), (5325.0, 0.4094725198), (5350.0, 0.4061600199),
            (5360.0, 0.4042961866), (5470.0, 0.3915701721), (5525.0, 0.3888935341),
            (5625.0, 0.3903589289), (5640.0, 0.3932377995), (5695.0, 0.3971587872),
            (5710.0, 0.3956144257), (5750.0, 0.3996213817), (5800.0, 0.4071047302),
            (5845.0, 0.4154294991), (5855.0, 0.4179475079), (5870.0, 0.4215683415),
            (5875.0, 0.4228200998), (5890.0, 0.4267101409), (5925.0, 0.4369235105),
            (5930.0, 0.4384938957), (5935.0, 0.4402228744), (5990.0, 0.4631120008),
            (6025.0, 0.4777678857), (6045.0, 0.4874670663),
        ];

        // Extreme case (16 DTE, non-monotonic put wing): one-sided data after
        // vol-cliff filter leaves SVI ill-conditioned. May produce a degenerate
        // fit rejected by Roger Lee bound or ATM sanity check.
        match SviSmile::calibrate(forward, expiry, &data) {
            Ok(smile) => {
                let atm_vol = smile.vol(forward).unwrap().0;
                assert!(
                    atm_vol.is_finite() && atm_vol > 0.0,
                    "ATM vol should be finite and positive, got {atm_vol}"
                );
            }
            Err(VolSurfError::CalibrationError { message, .. }) => {
                assert!(
                    message.contains("Roger Lee")
                        || message.contains("ATM total variance")
                        || message.contains("grid search"),
                    "unexpected rejection reason: {message}"
                );
            }
            Err(e) => panic!("unexpected error variant: {e}"),
        }
    }

    // Real-data fixture tests: ES futures options from vol-arb/scripts/essvi_fixtures/essvi_successes.json.
    // These exercise SVI calibration on clean market data (no vol-cliff artifacts) across
    // different tenors to validate the multi-start grid search (#92, #93).

    fn rms_vol_error(smile: &SviSmile, data: &[(f64, f64)]) -> f64 {
        (data
            .iter()
            .map(|&(strike, vol_obs)| {
                let vol_fit = smile.vol(strike).unwrap().0;
                (vol_fit - vol_obs).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64)
            .sqrt()
    }

    #[test]
    fn calibrate_real_es_15d() {
        // Source: essvi_successes.json snapshot[2] tenor[0], DTE=15.3, 45 strikes
        let forward = 6055.4718518824;
        let expiry = 0.041889117043121;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (4920.0, 0.4223900019718342), (4940.0, 0.4160130335559546),
            (4975.0, 0.4059536047220062), (4980.0, 0.4041102924314969),
            (4990.0, 0.4013275163056883), (5010.0, 0.3957023478220799),
            (5120.0, 0.3620851769504406), (5125.0, 0.3602334713020688),
            (5170.0, 0.3469515503931146), (5210.0, 0.3345908464586309),
            (5280.0, 0.3129955324874803), (5325.0, 0.2996367643338352),
            (5350.0, 0.2920767932130048), (5380.0, 0.2829159766349225),
            (5410.0, 0.2739666090456421), (5425.0, 0.2694025281972882),
            (5440.0, 0.2647809527980169), (5475.0, 0.2544764927597573),
            (5490.0, 0.2503163879294692), (5500.0, 0.2470554952911026),
            (5515.0, 0.2436761666236449), (5535.0, 0.2375425371287627),
            (5555.0, 0.2323741188316714), (5590.0, 0.2233370271498877),
            (5625.0, 0.2143433323941795), (5650.0, 0.208_700_651_085_321),
            (5655.0, 0.2073374741979704), (5660.0, 0.2063312810156373),
            (5720.0, 0.1939179206594279), (5795.0, 0.1804443709124559),
            (5805.0, 0.1788357094872418), (5815.0, 0.1771920609937395),
            (5870.0, 0.1686669163559528), (5875.0, 0.1678888445524988),
            (5990.0, 0.1517688737309481), (6025.0, 0.1472334289977049),
            (6055.0, 0.1439107310546137), (6070.0, 0.116_864_682_154_732),
            (6075.0, 0.1164335961807721), (6100.0, 0.1138490521891819),
            (6190.0, 0.1053720514799709), (6225.0, 0.1026879055355855),
            (6290.0, 0.1009732043280713), (6375.0, 0.1059670932035282),
            (6390.0, 0.1074644124985002),
        ];

        let smile =
            SviSmile::calibrate(forward, expiry, &data).expect("15d calibration should succeed");
        let rms = rms_vol_error(&smile, &data);
        assert!(rms < 0.05, "15d RMSE {rms:.4} exceeds 0.05");
    }

    #[test]
    fn calibrate_real_es_28d() {
        // Source: essvi_successes.json snapshot[0] tenor[0], DTE=28.3, 40 strikes
        let forward = 6064.7034250788;
        let expiry = 0.077481177275838;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (4920.0, 0.341_693_728_892_668), (4940.0, 0.3372436822896396),
            (4975.0, 0.3288045902621563), (4980.0, 0.3273480025257966),
            (4990.0, 0.3248606073729282), (5010.0, 0.320_279_883_241_698),
            (5120.0, 0.2934696300530372), (5170.0, 0.2815187173578774),
            (5210.0, 0.2716571313313472), (5325.0, 0.2447426026475534),
            (5380.0, 0.2315572971616197), (5410.0, 0.2246660262574118),
            (5440.0, 0.2175852660785433), (5490.0, 0.2063294444235871),
            (5515.0, 0.2009467418403518), (5535.0, 0.1962947443753465),
            (5555.0, 0.192_129_603_140_449), (5625.0, 0.1771372847059336),
            (5650.0, 0.1717237143340814), (5655.0, 0.1709675620197509),
            (5660.0, 0.1697392404023169), (5720.0, 0.1576601889574069),
            (5795.0, 0.143_159_763_874_207), (5805.0, 0.1413305137617879),
            (5815.0, 0.1393860453068289), (5870.0, 0.1289347395905459),
            (5875.0, 0.1279352498586123), (5990.0, 0.1054165247568596),
            (6025.0, 0.0977740702110937), (6055.0, 0.0908526262627002),
            (6070.0, 0.1546241067359042), (6075.0, 0.1534158532190984),
            (6100.0, 0.1474494709148409), (6190.0, 0.1304545118718378),
            (6225.0, 0.1257347370719426), (6290.0, 0.1193094914139261),
            (6375.0, 0.1151721266035816), (6390.0, 0.1151906054625841),
            (6410.0, 0.115_503_289_206_868), (6475.0, 0.1188455800934396),
        ];

        let smile =
            SviSmile::calibrate(forward, expiry, &data).expect("28d calibration should succeed");
        let rms = rms_vol_error(&smile, &data);
        assert!(rms < 0.05, "28d RMSE {rms:.4} exceeds 0.05");
    }

    #[test]
    fn calibrate_real_es_83d() {
        // Source: essvi_successes.json snapshot[0] tenor[2], DTE=83.3, 38 strikes
        let forward = 6103.9160617949;
        let expiry = 0.228062970568104;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (5000.0, 0.2562297957594517), (5050.0, 0.2490618316108794),
            (5160.0, 0.2337597518498384), (5170.0, 0.2324908560023248),
            (5210.0, 0.2269728829263208), (5260.0, 0.2200759574087432),
            (5290.0, 0.2160010369206193), (5325.0, 0.2113685978852301),
            (5370.0, 0.2055824688522813), (5450.0, 0.1951151124589882),
            (5480.0, 0.1910658140657824), (5570.0, 0.1795072032487414),
            (5575.0, 0.1788502851813726), (5590.0, 0.1769929549249026),
            (5600.0, 0.1756900311035595), (5640.0, 0.1706158274794452),
            (5650.0, 0.1693022104680442), (5670.0, 0.1667850546861248),
            (5675.0, 0.1661165625505582), (5710.0, 0.1616590548007632),
            (5835.0, 0.1457385103525783), (5860.0, 0.1425241762807806),
            (5885.0, 0.139_297_512_782_805), (5915.0, 0.1353827950561679),
            (5990.0, 0.1254076875624386), (6055.0, 0.1166151961160997),
            (6060.0, 0.1159464746590826), (6190.0, 0.1360088739957373),
            (6225.0, 0.1321805610552402), (6240.0, 0.130_652_593_916_186),
            (6280.0, 0.1269166651139219), (6300.0, 0.125_278_994_432_085),
            (6350.0, 0.1217407755629008), (6370.0, 0.1204814861580182),
            (6475.0, 0.1157617669455559), (6480.0, 0.1155635029959559),
            (6600.0, 0.1129563774031786), (6650.0, 0.1127541347933357),
        ];

        let smile =
            SviSmile::calibrate(forward, expiry, &data).expect("83d calibration should succeed");
        let rms = rms_vol_error(&smile, &data);
        assert!(rms < 0.05, "83d RMSE {rms:.4} exceeds 0.05");
    }

    #[test]
    fn calibrate_real_data_deterministic() {
        // SVI calibration must be deterministic: same input → same output.
        // Uses 83d fixture (longest tenor, widest k-range).
        let forward = 6103.9160617949;
        let expiry = 0.228062970568104;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (5000.0, 0.2562297957594517), (5050.0, 0.2490618316108794),
            (5160.0, 0.2337597518498384), (5170.0, 0.2324908560023248),
            (5210.0, 0.2269728829263208), (5260.0, 0.2200759574087432),
            (5290.0, 0.2160010369206193), (5325.0, 0.2113685978852301),
            (5370.0, 0.2055824688522813), (5450.0, 0.1951151124589882),
            (5480.0, 0.1910658140657824), (5570.0, 0.1795072032487414),
            (5575.0, 0.1788502851813726), (5590.0, 0.1769929549249026),
            (5600.0, 0.1756900311035595), (5640.0, 0.1706158274794452),
            (5650.0, 0.1693022104680442), (5670.0, 0.1667850546861248),
            (5675.0, 0.1661165625505582), (5710.0, 0.1616590548007632),
            (5835.0, 0.1457385103525783), (5860.0, 0.1425241762807806),
            (5885.0, 0.139_297_512_782_805), (5915.0, 0.1353827950561679),
            (5990.0, 0.1254076875624386), (6055.0, 0.1166151961160997),
            (6060.0, 0.1159464746590826), (6190.0, 0.1360088739957373),
            (6225.0, 0.1321805610552402), (6240.0, 0.130_652_593_916_186),
            (6280.0, 0.1269166651139219), (6300.0, 0.125_278_994_432_085),
            (6350.0, 0.1217407755629008), (6370.0, 0.1204814861580182),
            (6475.0, 0.1157617669455559), (6480.0, 0.1155635029959559),
            (6600.0, 0.1129563774031786), (6650.0, 0.1127541347933357),
        ];

        let s1 = SviSmile::calibrate(forward, expiry, &data).unwrap();
        let s2 = SviSmile::calibrate(forward, expiry, &data).unwrap();
        let s3 = SviSmile::calibrate(forward, expiry, &data).unwrap();
        assert_eq!(s1, s2, "calibrate() not deterministic (run 1 vs 2)");
        assert_eq!(s2, s3, "calibrate() not deterministic (run 2 vs 3)");
    }

    #[test]
    fn calibrate_spx_both_sided_skew() {
        // SPX Jan 13 2025 snapshot, Feb 21 expiry, both OTM puts + calls.
        // Steep put skew (47% at 70% moneyness → 15% ATM → 18% at 120% moneyness).
        // Should produce ATM vol ~14.5%, not blow up.
        let forward = 5876.5982;
        let expiry = 0.106849;
        #[rustfmt::skip]
        let data: Vec<(f64, f64)> = vec![
            (4100.0, 0.4732372167), (4350.0, 0.4185548802),
            (4625.0, 0.3586987129), (4875.0, 0.3062291404),
            (4975.0, 0.2861728415), (5070.0, 0.2677133003),
            (5150.0, 0.2533989022), (5240.0, 0.2377730229),
            (5325.0, 0.2237641910), (5420.0, 0.2090983236),
            (5495.0, 0.1978458240), (5545.0, 0.1909091037),
            (5600.0, 0.1832238219), (5655.0, 0.1758238490),
            (5705.0, 0.1691892272), (5760.0, 0.1619520673),
            (5810.0, 0.1554653403), (5865.0, 0.1485963933),
            (5920.0, 0.1417555115), (5970.0, 0.1358067719),
            (6025.0, 0.1300346440), (6080.0, 0.1250926343),
            (6130.0, 0.1218305015), (6185.0, 0.1195831160),
            (6235.0, 0.1187962506), (6300.0, 0.1191681275),
            (6390.0, 0.1218569546), (6475.0, 0.1256958312),
            (6570.0, 0.1324496829), (7000.0, 0.1835421893),
        ];

        let smile = SviSmile::calibrate(forward, expiry, &data)
            .expect("SPX both-sided calibration should succeed");

        let atm_vol = smile.vol(forward).unwrap().0;
        assert!(
            atm_vol > 0.10 && atm_vol < 0.25,
            "ATM vol {:.4} outside [10%, 25%] — calibration produced degenerate params",
            atm_vol
        );

        let rms = rms_vol_error(&smile, &data);
        assert!(rms < 0.01, "RMS vol error {rms:.4} exceeds 1% — poor fit");
    }

    #[test]
    fn vega_weighting_improves_atm_fit() {
        // Construct data with deep OTM noise that would pull an unweighted fit
        // away from ATM. Vega weighting should keep ATM vol close to truth.
        let true_smile = SviSmile::new(100.0, 0.5, 0.02, 0.3, -0.3, 0.0, 0.15).unwrap();
        let mut data = synthetic_market_data(
            &true_smile,
            &[80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
        );
        // Perturb deep OTM wings with +10% vol noise
        data[0].1 += 0.10;
        data[1].1 += 0.08;
        data[7].1 += 0.06;
        data[8].1 += 0.10;

        let calibrated = SviSmile::calibrate(100.0, 0.5, &data).unwrap();
        let true_atm = true_smile.vol(100.0).unwrap().0;
        let fit_atm = calibrated.vol(100.0).unwrap().0;
        let atm_err = (fit_atm - true_atm).abs();
        assert!(
            atm_err < 0.02,
            "ATM error {atm_err:.4} too large — vega weighting should keep ATM fit tight"
        );
    }

    #[test]
    fn calibrate_one_sided_data_skips_atm_sanity() {
        // All strikes above forward — no ATM bracket → w_atm=None → sanity check skipped.
        // Calibration should still succeed (or fail for grid reasons, not ATM sanity).
        let forward = 100.0;
        let expiry = 1.0;
        let data: Vec<(f64, f64)> = (0..10)
            .map(|i| (110.0 + 5.0 * i as f64, 0.20 + 0.005 * i as f64))
            .collect();
        match SviSmile::calibrate(forward, expiry, &data) {
            Ok(smile) => {
                let vol = smile.vol(130.0).unwrap().0;
                assert!(vol.is_finite() && vol > 0.0);
            }
            Err(VolSurfError::CalibrationError { message, .. }) => {
                assert!(
                    !message.contains("ATM total variance"),
                    "one-sided data should not trigger ATM sanity check: {message}"
                );
            }
            Err(e) => panic!("unexpected: {e}"),
        }
    }

    #[test]
    fn calibrate_rejects_degenerate_atm_overshoot() {
        // Craft data where optimal (m,sigma) would produce w_fitted(0) >> median w.
        // Very wide strike range with tight near-ATM cluster and one distant outlier
        // to lure the fit into overshooting ATM.
        let forward = 100.0;
        let expiry = 1.0;
        let mut data: Vec<(f64, f64)> = vec![
            (95.0, 0.22),
            (97.0, 0.21),
            (99.0, 0.20),
            (100.0, 0.20),
            (101.0, 0.20),
        ];
        // Add a cluster of very low-vol deep OTM points
        for i in 0..6 {
            data.push((150.0 + 10.0 * i as f64, 0.03));
        }

        match SviSmile::calibrate(forward, expiry, &data) {
            Ok(smile) => {
                // If it calibrates, ATM should still be reasonable (sanity check passed)
                let atm = smile.vol(forward).unwrap().0;
                assert!(
                    atm < 0.80,
                    "degenerate ATM {atm:.4} should be caught by sanity check"
                );
            }
            Err(VolSurfError::CalibrationError { message, .. }) => {
                // Expected: either ATM sanity or grid/Lee rejection
                assert!(
                    message.contains("ATM total variance")
                        || message.contains("Roger Lee")
                        || message.contains("grid search"),
                    "unexpected rejection: {message}"
                );
            }
            Err(e) => panic!("unexpected: {e}"),
        }
    }

    #[test]
    fn roger_lee_bound_prevents_calibration_to_steep_wings() {
        // Data with extremely steep wings that would require b*(1+|rho|) > 2 to fit.
        let forward = 100.0;
        let expiry = 1.0;
        let data: Vec<(f64, f64)> = vec![
            (70.0, 0.90),
            (80.0, 0.70),
            (90.0, 0.50),
            (95.0, 0.35),
            (100.0, 0.20),
            (105.0, 0.35),
            (110.0, 0.50),
            (120.0, 0.70),
            (130.0, 0.90),
        ];

        match SviSmile::calibrate(forward, expiry, &data) {
            Ok(smile) => {
                let lee = smile.b * (1.0 + smile.rho.abs());
                assert!(
                    lee <= 2.0,
                    "Roger Lee violated in calibrated params: b*(1+|rho|) = {lee}"
                );
            }
            Err(VolSurfError::CalibrationError { .. }) => {
                // Also acceptable — rejected before constructing invalid params
            }
            Err(e) => panic!("unexpected: {e}"),
        }
    }

    #[test]
    fn roger_lee_boundary_exact_in_calibration() {
        // Verify calibrate() never returns params exceeding the bound,
        // even for data that fits well with moderate b.
        let original = SviSmile::new(100.0, 1.0, 0.04, 0.8, -0.5, 0.0, 0.1).unwrap();
        let strikes: Vec<f64> = (0..20).map(|i| 60.0 + 4.0 * i as f64).collect();
        let data = synthetic_market_data(&original, &strikes);

        let calibrated = SviSmile::calibrate(100.0, 1.0, &data).unwrap();
        let lee = calibrated.b * (1.0 + calibrated.rho.abs());
        assert!(
            lee <= 2.0,
            "Roger Lee bound violated: b={}, rho={}, b*(1+|rho|)={}",
            calibrated.b,
            calibrated.rho,
            lee
        );
    }
}
