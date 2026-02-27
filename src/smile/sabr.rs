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
use crate::smile::SmileSection;
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
use crate::types::Vol;
use crate::validate::{validate_non_negative, validate_positive};

/// SABR volatility smile (Hagan et al., 2002).
///
/// Models the forward price as a CEV process with stochastic volatility,
/// parameterized by `(α, β, ρ, ν)`. The Hagan closed-form approximation
/// maps these parameters to Black implied volatility:
///
/// ```text
/// σ_B(K) ≈ (α / denom) · (z / x(z)) · [1 + correction · T]
/// ```
///
/// where `z = (ν/α) · (FK)^((1−β)/2) · ln(F/K)` and `x(z)` involves
/// the inverse sinh function.
///
/// # Parameters
///
/// | Parameter | Range | Meaning |
/// |-----------|-------|---------|
/// | `α` | `> 0` | ATM vol scale |
/// | `β` | `[0, 1]` | CEV exponent (backbone shape) |
/// | `ρ` | `(−1, 1)` | Spot-vol correlation (skew) |
/// | `ν` | `≥ 0` | Vol-of-vol (smile curvature) |
///
/// Common choices: `β = 0.5` (equity), `β = 0` (normal), `β = 1` (lognormal).
///
/// # Approximation limits
///
/// The Hagan formula is a first-order expansion in *T*. The correction factor
/// `1 + T·(…)` can become negative for long expiries combined with high |ρ|
/// and ν (e.g. *T* > 10 with |ρ| > 0.9 and ν > 1). When this happens the
/// correction is clamped to a small positive floor, producing a near-zero
/// but valid implied vol.
/// Results in that regime should be treated with caution.
///
/// # References
///
/// - Hagan, P. et al., "Managing Smile Risk", Wilmott Magazine, Jan 2002
///
/// # Examples
///
/// ```
/// use volsurf::smile::{SabrSmile, SmileSection};
///
/// let smile = SabrSmile::new(100.0, 1.0, 0.3, 1.0, -0.3, 0.4)?;
/// let vol = smile.vol(100.0)?; // ATM vol
/// assert!(vol.0 > 0.0);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SabrSmileRaw", into = "SabrSmileRaw")]
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

#[derive(Serialize, Deserialize)]
struct SabrSmileRaw {
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
}

impl TryFrom<SabrSmileRaw> for SabrSmile {
    type Error = VolSurfError;
    fn try_from(raw: SabrSmileRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.forward,
            raw.expiry,
            raw.alpha,
            raw.beta,
            raw.rho,
            raw.nu,
        )
    }
}

impl From<SabrSmile> for SabrSmileRaw {
    fn from(s: SabrSmile) -> Self {
        Self {
            forward: s.forward,
            expiry: s.expiry,
            alpha: s.alpha,
            beta: s.beta,
            rho: s.rho,
            nu: s.nu,
        }
    }
}

impl SabrSmile {
    /// Create a SABR smile from calibrated parameters.
    ///
    /// # Arguments
    ///
    /// * `forward` — Forward price, must be positive
    /// * `expiry` — Time to expiry in years, must be positive
    /// * `alpha` — ATM vol scale, must be positive
    /// * `beta` — CEV exponent, must be in \[0, 1\]
    /// * `rho` — Spot-vol correlation, must be in (−1, 1)
    /// * `nu` — Vol-of-vol, must be non-negative
    ///
    /// # Errors
    ///
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is out of range.
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

    /// ATM vol scale parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// CEV exponent (0 = normal, 1 = lognormal).
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Spot-vol correlation.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Vol-of-vol.
    pub fn nu(&self) -> f64 {
        self.nu
    }

    // Hagan (2002) Eq. (2.17a) — unified formula with Taylor z/x(z) for small z.
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
        let z = if nu == 0.0 {
            0.0
        } else {
            (nu / alpha) * fk_mid * ln_fk
        };
        let z_ratio = if z.abs() < 1e-6 {
            // Taylor: z/x(z) ≈ 1 − ρz/2 + (2−3ρ²)/12 · z²
            1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) / 12.0 * z * z
        } else {
            let disc = (1.0 - 2.0 * rho * z + z * z).sqrt();
            let xz = ((disc + z - rho) / (1.0 - rho)).ln();
            z / xz
        };

        // Time-dependent correction factor — first-order in T, can go negative
        // for extreme (ρ, ν, T) combinations. Clamp to ε so the formula degrades
        // to a small positive vol rather than producing NumericalError.
        let fk_omb = fk.powf(omb); // (FK)^(1-β)
        let correction = (1.0
            + t * (omb_sq / 24.0 * alpha * alpha / fk_omb
                + 0.25 * rho * beta * nu * alpha / fk_mid
                + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu))
            .max(1e-10);

        (alpha / denom) * z_ratio * correction
    }

    /// Calibrate SABR parameters from market (strike, vol) observations.
    ///
    /// Beta is fixed by the user (industry convention) and is not calibrated.
    /// Alpha is solved analytically from the ATM vol for each (ρ, ν) candidate.
    /// Rho and nu are optimized via Nelder-Mead in transformed space
    /// (`ρ = tanh(x)`, `ν = exp(y)`).
    ///
    /// # Arguments
    /// * `forward` — Forward price, must be positive
    /// * `expiry` — Time to expiry in years, must be positive
    /// * `beta` — CEV exponent, fixed, must be in \[0, 1\]
    /// * `market_vols` — Slice of `(strike, implied_vol)` pairs (minimum 4)
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for bad inputs,
    /// [`VolSurfError::CalibrationError`] if the optimizer fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::smile::{SabrSmile, SmileSection};
    ///
    /// let market_vols = vec![
    ///     (80.0, 0.28), (90.0, 0.24), (100.0, 0.20),
    ///     (110.0, 0.22), (120.0, 0.26),
    /// ];
    /// let smile = SabrSmile::calibrate(100.0, 1.0, 0.5, &market_vols)?;
    /// let vol = smile.vol(100.0)?;
    /// assert!((vol.0 - 0.20).abs() < 0.01);
    /// # Ok::<(), volsurf::VolSurfError>(())
    /// ```
    pub fn calibrate(
        forward: f64,
        expiry: f64,
        beta: f64,
        market_vols: &[(f64, f64)],
    ) -> error::Result<Self> {
        #[cfg(feature = "logging")]
        tracing::debug!(
            forward,
            expiry,
            beta,
            n_quotes = market_vols.len(),
            "SABR calibration started"
        );

        const MIN_POINTS: usize = 4;
        const GRID_N: usize = 15;
        const NM_MAX_ITER: usize = 300;
        const NM_DIAMETER_TOL: f64 = 1e-8;
        const NM_FVALUE_TOL: f64 = 1e-12;
        const ALPHA_MAX_ITER: usize = 50;

        // Input validation
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        if !(0.0..=1.0).contains(&beta) || !beta.is_finite() {
            return Err(VolSurfError::InvalidInput {
                message: format!("beta must be in [0, 1], got {beta}"),
            });
        }
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

        // Interpolate ATM vol from market data
        let sigma_atm = interpolate_atm_vol(forward, market_vols);
        let f_beta = forward.powf(1.0 - beta);

        // Alpha solver: Newton iteration on the ATM cubic
        //
        // ATM Hagan formula (K = F):
        //   σ_ATM = (α / F^(1-β)) * [1 + T·(A·α² + B·α + C)]
        //
        // where:
        //   A = (1-β)² / (24 · F^(2(1-β)))
        //   B = ρ·β·ν / (4 · F^(1-β))
        //   C = (2 - 3ρ²) / 24 · ν²
        //
        // Rearranged as f(α) = 0:
        //   f(α) = T·A·α³ + T·B·α² + (1 + T·C)·α − σ_ATM·F^(1-β) = 0
        let solve_alpha = |rho: f64, nu: f64| -> Option<f64> {
            let omb = 1.0 - beta;
            let a_coeff = omb * omb / (24.0 * f_beta * f_beta);
            let b_coeff = rho * beta * nu / (4.0 * f_beta);
            let c_coeff = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
            let target = sigma_atm * f_beta;
            let t = expiry;

            // f(α) = t·A·α³ + t·B·α² + (1 + t·C)·α − target
            // f'(α) = 3·t·A·α² + 2·t·B·α + (1 + t·C)
            let mut alpha = target; // initial guess: zeroth-order approximation
            for _ in 0..ALPHA_MAX_ITER {
                let a2 = alpha * alpha;
                let f_val =
                    t * a_coeff * a2 * alpha + t * b_coeff * a2 + (1.0 + t * c_coeff) * alpha
                        - target;
                let f_prime =
                    3.0 * t * a_coeff * a2 + 2.0 * t * b_coeff * alpha + (1.0 + t * c_coeff);
                if f_prime.abs() < 1e-30 {
                    return None;
                }
                let delta = f_val / f_prime;
                alpha -= delta;
                if alpha <= 0.0 {
                    return None;
                }
                if delta.abs() < 1e-14 * alpha {
                    break;
                }
            }
            if alpha > 0.0 && alpha.is_finite() {
                Some(alpha)
            } else {
                None
            }
        };

        // Objective function in transformed space
        // x → rho = tanh(x), y → nu = exp(y)
        let objective = |x: f64, y: f64| -> f64 {
            let rho = x.tanh();
            let nu = y.exp();
            if nu > 100.0 {
                return f64::MAX;
            }
            let alpha = match solve_alpha(rho, nu) {
                Some(a) => a,
                None => return f64::MAX,
            };
            // Build a temporary SABR smile and compute RSS
            let smile = match Self::new(forward, expiry, alpha, beta, rho, nu) {
                Ok(s) => s,
                Err(_) => return f64::MAX,
            };
            let mut rss = 0.0;
            for &(strike, market_vol) in market_vols {
                let model_vol = smile.hagan_implied_vol(strike);
                if !model_vol.is_finite() || model_vol <= 0.0 {
                    return f64::MAX;
                }
                let diff = model_vol - market_vol;
                rss += diff * diff;
            }
            rss
        };

        // Grid search over transformed (x, y) space
        let x_lo = -1.5_f64; // tanh(-1.5) ≈ -0.905
        let x_hi = 1.5_f64; // tanh(1.5) ≈ 0.905
        let y_lo = (-2.0_f64).max((0.01_f64).ln()); // nu ≥ 0.01
        let y_hi = (2.0_f64).ln(); // nu ≤ ~7.4

        let mut best_x = 0.0;
        let mut best_y = 0.0;
        let mut best_rss = f64::MAX;

        for ix in 0..GRID_N {
            let x = x_lo + (x_hi - x_lo) * (ix as f64) / ((GRID_N - 1) as f64);
            for iy in 0..GRID_N {
                let y = y_lo + (y_hi - y_lo) * (iy as f64) / ((GRID_N - 1) as f64);
                let rss = objective(x, y);
                if rss < best_rss {
                    best_rss = rss;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        if best_rss >= f64::MAX {
            return Err(VolSurfError::CalibrationError {
                message: "grid search found no valid starting point".into(),
                model: "SABR",
                rms_error: None,
            });
        }

        // Nelder-Mead 2D refinement
        let step_x = (x_hi - x_lo) / (GRID_N as f64) * 0.5;
        let step_y = (y_hi - y_lo) / (GRID_N as f64) * 0.5;

        let nm_config = crate::optim::NelderMeadConfig {
            max_iter: NM_MAX_ITER,
            diameter_tol: NM_DIAMETER_TOL,
            fvalue_tol: NM_FVALUE_TOL,
        };
        let nm_result =
            crate::optim::nelder_mead_2d(objective, best_x, best_y, step_x, step_y, &nm_config);

        // Recover final parameters
        let rho = nm_result.x.tanh();
        let nu = nm_result.y.exp();
        let alpha = solve_alpha(rho, nu).ok_or_else(|| VolSurfError::CalibrationError {
            message: "alpha solve failed at optimal (rho, nu)".into(),
            model: "SABR",
            rms_error: None,
        })?;

        let rms = (nm_result.fval / market_vols.len() as f64).sqrt();

        #[cfg(feature = "logging")]
        tracing::debug!(
            alpha,
            beta,
            rho,
            nu,
            rms_implied_vol = rms,
            "SABR calibration complete"
        );

        Self::new(forward, expiry, alpha, beta, rho, nu).map_err(|e| {
            VolSurfError::CalibrationError {
                message: format!("calibrated params invalid: {e}"),
                model: "SABR",
                rms_error: Some(rms),
            }
        })
    }
}

fn interpolate_atm_vol(forward: f64, market_vols: &[(f64, f64)]) -> f64 {
    // Sort by strike distance to forward
    let mut sorted: Vec<(f64, f64)> = market_vols.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Find bracketing strikes
    let right_idx = sorted.partition_point(|&(k, _)| k < forward);
    if right_idx == 0 {
        // Forward is below all strikes — use the lowest
        return sorted[0].1;
    }
    if right_idx >= sorted.len() {
        // Forward is above all strikes — use the highest
        return sorted[sorted.len() - 1].1;
    }
    // Linear interpolation between bracketing strikes
    let (k_lo, v_lo) = sorted[right_idx - 1];
    let (k_hi, v_hi) = sorted[right_idx];
    let alpha = (forward - k_lo) / (k_hi - k_lo);
    (1.0 - alpha) * v_lo + alpha * v_hi
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

    #[test]
    fn smile_section_forward_and_expiry() {
        let s = make_smile();
        assert_eq!(SmileSection::forward(&s), F);
        assert_eq!(SmileSection::expiry(&s), T);
    }

    #[test]
    fn serde_round_trip() {
        let s = make_smile();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SabrSmile = serde_json::from_str(&json).unwrap();
        assert_eq!(SmileSection::forward(&s), SmileSection::forward(&s2));
        assert_eq!(SmileSection::expiry(&s), SmileSection::expiry(&s2));
        assert_eq!(s.alpha(), s2.alpha());
        assert_eq!(s.beta(), s2.beta());
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.nu(), s2.nu());
    }

    #[test]
    fn serde_rejects_negative_forward() {
        let json = r#"{"forward":-100.0,"expiry":1.0,"alpha":0.2,"beta":0.5,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_expiry() {
        let json = r#"{"forward":100.0,"expiry":0.0,"alpha":0.2,"beta":0.5,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_alpha() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":-0.1,"beta":0.5,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_alpha() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.0,"beta":0.5,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_beta_out_of_range() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":1.5,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_beta() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":-0.1,"rho":-0.3,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_plus_one() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":0.5,"rho":1.0,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_at_minus_one() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":0.5,"rho":-1.0,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_rho_out_of_range() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":0.5,"rho":1.5,"nu":0.4}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_nu() {
        let json = r#"{"forward":100.0,"expiry":1.0,"alpha":0.2,"beta":0.5,"rho":-0.3,"nu":-0.1}"#;
        assert!(serde_json::from_str::<SabrSmile>(json).is_err());
    }

    // ========================================================================
    // Hagan formula tests (T02)
    // ========================================================================

    // Use realistic equity-like params: ATM vol ≈ 20%
    const EQ_ALPHA: f64 = 2.0;

    fn make_equity_smile() -> SabrSmile {
        SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, NU).unwrap()
    }

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

    #[test]
    fn vol_beta_one_lognormal_sabr() {
        // beta=1: lognormal SABR, ATM vol ≈ alpha
        let alpha = 0.20;
        let s = SabrSmile::new(F, T, alpha, 1.0, RHO, NU).unwrap();
        let v = s.vol(F).unwrap().0;
        // σ_ATM = α * [1 + T*(¼ρνα + (2-3ρ²)/24*ν²)]
        let expected = alpha
            * (1.0 + T * (0.25 * RHO * NU * alpha + (2.0 - 3.0 * RHO * RHO) / 24.0 * NU * NU));
        assert!(
            (v - expected).abs() < 1e-14,
            "Lognormal ATM: expected {expected}, got {v}"
        );
    }

    #[test]
    fn vol_nu_zero_cev_limit() {
        // nu=0: z/x(z) = 1, no vol-of-vol terms
        let s = SabrSmile::new(F, T, EQ_ALPHA, BETA, RHO, 0.0).unwrap();
        let v_atm = s.vol(F).unwrap().0;
        let omb = 1.0 - BETA;
        let f_omb = F.powf(omb);
        // Only the (1-β)²/24 * α²/F^(2(1-β)) term survives
        let expected =
            EQ_ALPHA / f_omb * (1.0 + T * omb * omb / 24.0 * EQ_ALPHA * EQ_ALPHA / (f_omb * f_omb));
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

    #[test]
    fn vol_rejects_zero_strike() {
        let s = make_equity_smile();
        assert!(matches!(s.vol(0.0), Err(VolSurfError::InvalidInput { .. })));
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

    #[test]
    fn vol_taylor_branch_continuity() {
        let smile = make_smile();
        let k_inside = F * (-5e-8_f64).exp();
        let k_outside = F * (-5e-5_f64).exp();
        let v_inside = smile.vol(k_inside).unwrap().0;
        let v_outside = smile.vol(k_outside).unwrap().0;
        let v_atm = smile.vol(F).unwrap().0;
        assert!(v_inside.is_finite());
        assert!(
            (v_inside - v_atm).abs() < 1e-6,
            "Taylor branch should be near ATM vol"
        );
        assert!(
            (v_inside - v_outside).abs() < 1e-4,
            "Taylor and exact branches should be continuous"
        );
    }

    #[test]
    fn vol_nu_zero_otm_strikes() {
        let smile = SabrSmile::new(F, T, ALPHA, BETA, 0.0, 0.0).unwrap();
        for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let v = smile.vol(k).unwrap().0;
            assert!(
                v > 0.01 && v < 1.0,
                "nu=0 vol at K={k} should be in (0.01, 1.0), got {v}"
            );
        }
    }

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

    // Hagan correction factor goes negative for extreme (ρ, ν, T) — issue #50.
    // The correction clamp ensures vol() returns positive rather than NumericalError.
    #[test]
    fn vol_negative_correction_clamped_t20() {
        let s = SabrSmile::new(100.0, 20.0, 0.3, 0.5, -0.95, 1.5).unwrap();
        for &k in &[80.0, 100.0, 120.0] {
            let v = s.vol(k);
            assert!(v.is_ok(), "T=20, K={k}: should not return NumericalError");
            let vol = v.unwrap().0;
            assert!(vol > 0.0, "T=20, K={k}: vol must be positive");
            assert!(
                vol < 0.01,
                "T=20, K={k}: clamped vol should be near-zero, got {vol}"
            );
        }
    }

    #[test]
    fn vol_negative_correction_clamped_t10() {
        let s = SabrSmile::new(100.0, 10.0, 0.3, 0.5, -0.95, 1.5).unwrap();
        let v = s.vol(100.0).unwrap().0;
        assert!(v > 0.0, "T=10, ATM vol must be positive, got {v}");
        assert!(v.is_finite(), "T=10, ATM vol must be finite");
    }

    #[test]
    fn vol_correction_boundary_degrades_monotonically() {
        // T=8 is in the unclamped regime, T=12 is near the sign-change boundary.
        // Vol should degrade monotonically — no discontinuity at the clamp.
        let s8 = SabrSmile::new(100.0, 8.0, 0.3, 0.5, -0.95, 1.5).unwrap();
        let s12 = SabrSmile::new(100.0, 12.0, 0.3, 0.5, -0.95, 1.5).unwrap();
        let v8 = s8.vol(100.0).unwrap().0;
        let v12 = s12.vol(100.0).unwrap().0;
        assert!(
            v8 > v12,
            "vol should decrease as T increases past the boundary"
        );
        assert!(v12 > 0.0, "T=12 vol should still be positive");
    }

    #[test]
    fn vol_negative_correction_clamped_beta_one() {
        // beta=1 (lognormal): omb=0 kills two correction terms, only (2-3ρ²)/24·ν²
        // remains. Clamp should still fire for extreme ρ and long T.
        let s = SabrSmile::new(100.0, 20.0, 0.20, 1.0, -0.95, 1.5).unwrap();
        let v = s.vol(100.0).unwrap().0;
        assert!(v > 0.0, "beta=1 clamped vol must be positive");
        assert!(v < 0.01, "beta=1 clamped vol should be near-zero, got {v}");
    }

    #[test]
    fn arb_free_clamped_regime() {
        // In the clamped regime the smile is nearly flat at ~0, so density ≈ 0
        // everywhere. The arb scan should complete without crashing.
        let s = SabrSmile::new(100.0, 20.0, 0.3, 0.5, -0.95, 1.5).unwrap();
        let report = s.is_arbitrage_free();
        assert!(
            report.is_ok(),
            "arb scan should not error in clamped regime"
        );
    }

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
        assert!(
            report.is_free,
            "β=1 with moderate params should be arb-free"
        );
    }

    #[test]
    fn arb_free_beta_zero() {
        let s = SabrSmile::new(F, T, 10.0, 0.0, -0.2, 0.30).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(
            report.is_free,
            "β=0 with moderate params should be arb-free"
        );
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

    #[test]
    fn sabr_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SabrSmile>();
    }

    #[test]
    fn sabr_as_trait_object() {
        let s = make_equity_smile();
        let boxed: Box<dyn SmileSection> = Box::new(s);
        let v = boxed.vol(F).unwrap();
        assert!(v.0 > 0.0);
        let report = boxed.is_arbitrage_free().unwrap();
        assert!(report.is_free);
    }

    // ========================================================================
    // Calibration tests (T04)
    // ========================================================================

    /// Generate synthetic market data from a known SabrSmile.
    fn sabr_synthetic_data(smile: &SabrSmile, strikes: &[f64]) -> Vec<(f64, f64)> {
        strikes
            .iter()
            .map(|&k| (k, smile.vol(k).unwrap().0))
            .collect()
    }

    /// Compute RMS vol error between calibrated and original across strikes.
    fn vol_rms(original: &SabrSmile, calibrated: &SabrSmile, strikes: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for &k in strikes {
            let v_orig = original.vol(k).unwrap().0;
            let v_cal = calibrated.vol(k).unwrap().0;
            sum_sq += (v_orig - v_cal).powi(2);
        }
        (sum_sq / strikes.len() as f64).sqrt()
    }

    #[test]
    fn calibrate_round_trip_equity() {
        // beta = 0.5 (equity convention)
        let original = SabrSmile::new(F, T, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, 0.5, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "equity round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_round_trip_rates() {
        // beta = 0.0 (rates convention, normal SABR)
        let original = SabrSmile::new(F, T, 10.0, 0.0, -0.2, 0.3).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, 0.0, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "rates round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_round_trip_lognormal() {
        // beta = 1.0 (lognormal SABR)
        let original = SabrSmile::new(F, T, 0.20, 1.0, -0.25, 0.3).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, 1.0, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(
            rms < 0.001,
            "lognormal round-trip RMS {rms} should be < 0.001"
        );
    }

    #[test]
    fn calibrate_minimum_four_points() {
        let original = SabrSmile::new(F, T, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes = [85.0, 95.0, 105.0, 115.0];
        let data = sabr_synthetic_data(&original, &strikes);
        let result = SabrSmile::calibrate(F, T, 0.5, &data);
        assert!(result.is_ok(), "4 points should succeed: {result:?}");
    }

    #[test]
    fn calibrate_rejects_three_points() {
        let data = vec![(90.0, 0.20), (100.0, 0.18), (110.0, 0.22)];
        let result = SabrSmile::calibrate(F, T, 0.5, &data);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_zero_points() {
        let result = SabrSmile::calibrate(F, T, 0.5, &[]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn calibrate_rejects_invalid_forward() {
        let data = vec![(90.0, 0.2), (95.0, 0.19), (100.0, 0.18), (105.0, 0.19)];
        assert!(SabrSmile::calibrate(0.0, T, 0.5, &data).is_err());
        assert!(SabrSmile::calibrate(-1.0, T, 0.5, &data).is_err());
    }

    #[test]
    fn calibrate_rejects_invalid_expiry() {
        let data = vec![(90.0, 0.2), (95.0, 0.19), (100.0, 0.18), (105.0, 0.19)];
        assert!(SabrSmile::calibrate(F, 0.0, 0.5, &data).is_err());
        assert!(SabrSmile::calibrate(F, -1.0, 0.5, &data).is_err());
    }

    #[test]
    fn calibrate_rejects_invalid_beta() {
        let data = vec![(90.0, 0.2), (95.0, 0.19), (100.0, 0.18), (105.0, 0.19)];
        assert!(SabrSmile::calibrate(F, T, -0.1, &data).is_err());
        assert!(SabrSmile::calibrate(F, T, 1.5, &data).is_err());
    }

    #[test]
    fn calibrate_rejects_invalid_market_data() {
        // Negative vol
        let data = vec![(90.0, 0.2), (95.0, -0.1), (100.0, 0.18), (105.0, 0.19)];
        assert!(SabrSmile::calibrate(F, T, 0.5, &data).is_err());
        // Zero strike
        let data = vec![(0.0, 0.2), (95.0, 0.19), (100.0, 0.18), (105.0, 0.19)];
        assert!(SabrSmile::calibrate(F, T, 0.5, &data).is_err());
    }

    #[test]
    fn calibrate_grid_search_fails_overflow_alpha() {
        // ATM vol near f64 overflow: sigma_atm * F^(1-beta) overflows, making
        // the Newton target = Inf. Alpha diverges for all (rho, nu) grid points,
        // so the grid search finds no valid starting point.
        let data = vec![
            (90.0, 1e150),
            (95.0, 1e150),
            (100.0, 1e150),
            (105.0, 1e150),
            (110.0, 1e150),
        ];
        let result = SabrSmile::calibrate(100.0, 1.0, 0.5, &data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, VolSurfError::CalibrationError { model: "SABR", .. }),
            "expected SABR CalibrationError, got: {err}"
        );
    }

    #[test]
    fn calibration_error_format_sabr_alpha_solve() {
        let err = VolSurfError::CalibrationError {
            message: "alpha solve failed at optimal (rho, nu)".into(),
            model: "SABR",
            rms_error: None,
        };
        assert!(err.to_string().contains("alpha solve failed"));
    }

    #[test]
    fn calibrate_params_pass_validation() {
        // Calibrated params should always be valid (pass SabrSmile::new)
        let original = SabrSmile::new(F, T, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..10).map(|i| 75.0 + 5.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let cal = SabrSmile::calibrate(F, T, 0.5, &data).unwrap();
        assert!(cal.alpha() > 0.0);
        assert!((0.0..=1.0).contains(&cal.beta()));
        assert!(cal.rho().abs() < 1.0);
        assert!(cal.nu() >= 0.0);
    }

    #[test]
    fn calibrate_positive_rho_recoverable() {
        // Test with positive rho (non-standard, but valid)
        let original = SabrSmile::new(F, T, 2.0, 0.5, 0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, 0.5, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(
            rms < 0.001,
            "positive rho round-trip RMS {rms} should be < 0.001"
        );
    }

    #[test]
    fn calibrate_zero_rho_recoverable() {
        // rho = 0 (symmetric smile for beta=1)
        let original = SabrSmile::new(F, T, 0.20, 1.0, 0.0, 0.3).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, 1.0, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(
            rms < 0.001,
            "zero rho round-trip RMS {rms} should be < 0.001"
        );
    }

    #[test]
    fn calibrate_different_forward() {
        // Non-100 forward
        let fwd = 50.0;
        let original = SabrSmile::new(fwd, T, 1.5, 0.5, -0.2, 0.3).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 35.0 + 2.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(fwd, T, 0.5, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(rms < 0.001, "F=50 round-trip RMS {rms} should be < 0.001");
    }

    #[test]
    fn calibrate_short_expiry() {
        let original = SabrSmile::new(F, 0.1, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..10).map(|i| 80.0 + 4.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, 0.1, 0.5, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(
            rms < 0.001,
            "short expiry round-trip RMS {rms} should be < 0.001"
        );
    }

    #[test]
    fn calibrate_round_trip_non_uniform_strikes() {
        let original = make_smile();
        let strikes = vec![
            60.0, 70.0, 80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0,
            115.0, 120.0, 130.0, 140.0,
        ];
        let data = sabr_synthetic_data(&original, &strikes);
        let calibrated = SabrSmile::calibrate(F, T, BETA, &data).unwrap();
        let rms = vol_rms(&original, &calibrated, &strikes);
        assert!(
            rms < 0.001,
            "non-uniform round-trip RMS {rms} should be < 0.001"
        );
    }

    #[test]
    fn calibrate_performance() {
        // Calibration should complete in < 1ms for 10 market points.
        // We just verify it doesn't take unreasonably long (no assertion on exact time
        // since test environments vary, but the test itself timing out would indicate a problem).
        let original = SabrSmile::new(F, T, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..10).map(|i| 75.0 + 5.0 * i as f64).collect();
        let data = sabr_synthetic_data(&original, &strikes);
        let start = std::time::Instant::now();
        let _cal = SabrSmile::calibrate(F, T, 0.5, &data).unwrap();
        let elapsed = start.elapsed();
        // Generous bound — the target is <1ms but test environments vary
        assert!(
            elapsed.as_millis() < 100,
            "calibration took {}ms, should be fast",
            elapsed.as_millis()
        );
    }

    // ========================================================================
    // Academic validation tests (T05)
    //
    // Independent reference implementation of Hagan (2002) Eq. (2.17a)
    // for cross-validation of the production code. Every test cites the
    // source formula and parameter set.
    // ========================================================================

    /// Independent implementation of Hagan (2002) Eq. (2.17a) for cross-validation.
    ///
    /// This is a standalone transcription of the SABR implied vol formula,
    /// deliberately written differently from `SabrSmile::hagan_implied_vol()`
    /// to catch implementation errors via independent computation.
    ///
    /// Reference: Hagan, P. et al., "Managing Smile Risk", Wilmott Magazine,
    /// January 2002, Equation (2.17a).
    fn hagan_reference(f: f64, k: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
        let one_minus_beta = 1.0 - beta;
        let fk_product = f * k;
        let log_ratio = (f / k).ln();

        // (FK)^((1-β)/2)
        let fk_half_power = fk_product.powf(one_minus_beta * 0.5);

        // Denominator D from Eq. (2.17a)
        let omb2 = one_minus_beta * one_minus_beta;
        let log2 = log_ratio * log_ratio;
        let denom = fk_half_power * (1.0 + omb2 / 24.0 * log2 + omb2 * omb2 / 1920.0 * log2 * log2);

        // z and z/x(z) ratio
        let z = if nu.abs() < 1e-30 {
            0.0
        } else {
            nu / alpha * fk_half_power * log_ratio
        };

        let z_over_xz = if z.abs() < 1e-6 {
            // Taylor series: 1 - ρz/2 + (2-3ρ²)/12 · z²
            1.0 + z * (-0.5 * rho + z * (2.0 - 3.0 * rho * rho) / 12.0)
        } else {
            let sqrt_term = (1.0 - 2.0 * rho * z + z * z).sqrt();
            let x_of_z = ((sqrt_term + z - rho) / (1.0 - rho)).ln();
            z / x_of_z
        };

        // (FK)^(1-β)
        let fk_full_power = fk_product.powf(one_minus_beta);

        // Time-dependent correction factor from Eq. (2.17a)
        let term1 = omb2 / 24.0 * alpha * alpha / fk_full_power;
        let term2 = 0.25 * rho * beta * nu * alpha / fk_half_power;
        let term3 = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
        let correction = 1.0 + t * (term1 + term2 + term3);

        alpha / denom * z_over_xz * correction
    }

    // Hagan (2002) Eq. (2.17a): F=100, T=1, α=2.0, β=0.5, ρ=−0.3, ν=0.4

    /// Hagan (2002) Eq. (2.17a), equity params, K=60 (deep ITM put).
    #[test]
    fn hagan_equity_k60() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 60.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(60.0).unwrap().0;
        let eps = expected * 1e-12;
        assert!(
            (actual - expected).abs() < eps,
            "K=60: expected {expected:.15e}, got {actual:.15e}, err={:.2e}",
            (actual - expected).abs()
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=70.
    #[test]
    fn hagan_equity_k70() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 70.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(70.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=70: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=80.
    #[test]
    fn hagan_equity_k80() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 80.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(80.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=80: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=90.
    #[test]
    fn hagan_equity_k90() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 90.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(90.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=90: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=95.
    #[test]
    fn hagan_equity_k95() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 95.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(95.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=95: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=100 (ATM).
    #[test]
    fn hagan_equity_atm() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        // ATM: analytical formula (no z/x(z) needed, z=0 → ratio=1)
        let f = 100.0_f64;
        let omb = 0.5;
        let f_omb = f.powf(omb);
        let expected = 2.0 / f_omb
            * (1.0
                + 1.0
                    * (omb * omb / 24.0 * 4.0 / (f_omb * f_omb)
                        + 0.25 * (-0.3) * 0.5 * 0.4 * 2.0 / f_omb
                        + (2.0 - 3.0 * 0.09) / 24.0 * 0.16));
        let actual = s.vol(100.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "ATM: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=105.
    #[test]
    fn hagan_equity_k105() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 105.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(105.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=105: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=110.
    #[test]
    fn hagan_equity_k110() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 110.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(110.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=110: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=120.
    #[test]
    fn hagan_equity_k120() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 120.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(120.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=120: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=130.
    #[test]
    fn hagan_equity_k130() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 130.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(130.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=130: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), equity params, K=140.
    #[test]
    fn hagan_equity_k140() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let expected = hagan_reference(100.0, 140.0, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual = s.vol(140.0).unwrap().0;
        assert!(
            (actual - expected).abs() < expected * 1e-12,
            "K=140: expected {expected:.15e}, got {actual:.15e}"
        );
    }

    // Hagan (2002) Eq. (2.17a): F=0.05, T=2, α=0.01, β=0, ρ=−0.2, ν=0.3

    /// Hagan (2002) Eq. (2.17a), normal SABR (β=0), full strike range.
    /// F=0.05 (5% rate), T=2y, α=0.01, ρ=−0.2, ν=0.3.
    #[test]
    fn hagan_normal_sabr_strike_range() {
        let f = 0.05;
        let t = 2.0;
        let alpha = 0.01;
        let rho = -0.2;
        let nu = 0.3;
        let s = SabrSmile::new(f, t, alpha, 0.0, rho, nu).unwrap();

        for &k in &[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09] {
            let expected = hagan_reference(f, k, t, alpha, 0.0, rho, nu);
            let actual = s.vol(k).unwrap().0;
            let eps = expected.abs() * 1e-12;
            assert!(
                (actual - expected).abs() < eps.max(1e-15),
                "β=0, K={k}: expected {expected:.15e}, got {actual:.15e}, err={:.2e}",
                (actual - expected).abs()
            );
        }
    }

    // Hagan (2002) Eq. (2.17a): F=100, T=0.5, α=0.20, β=1, ρ=−0.5, ν=0.6

    /// Hagan (2002) Eq. (2.17a), lognormal SABR (β=1), full strike range.
    /// F=100, T=0.5y, α=0.20, ρ=−0.5, ν=0.6.
    #[test]
    fn hagan_lognormal_sabr_strike_range() {
        let f = 100.0;
        let t = 0.5;
        let alpha = 0.20;
        let rho = -0.5;
        let nu = 0.6;
        let s = SabrSmile::new(f, t, alpha, 1.0, rho, nu).unwrap();

        for &k in &[60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0] {
            let expected = hagan_reference(f, k, t, alpha, 1.0, rho, nu);
            let actual = s.vol(k).unwrap().0;
            let eps = expected * 1e-12;
            assert!(
                (actual - expected).abs() < eps,
                "β=1, K={k}: expected {expected:.15e}, got {actual:.15e}, err={:.2e}",
                (actual - expected).abs()
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), CEV limit (ν=1e-10).
    /// As ν → 0, SABR reduces to CEV: σ_B = α·F^(β-1).
    /// With small ν, z → 0, z/x(z) → 1, and the ν² correction vanishes.
    #[test]
    fn hagan_cev_limit_nu_near_zero() {
        let nu_tiny = 1e-10;
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, nu_tiny).unwrap();
        // Cross-validate against independent reference
        for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let expected = hagan_reference(100.0, k, 1.0, 2.0, 0.5, -0.3, nu_tiny);
            let actual = s.vol(k).unwrap().0;
            let eps = expected * 1e-12;
            assert!(
                (actual - expected).abs() < eps.max(1e-15),
                "CEV, K={k}: expected {expected:.15e}, got {actual:.15e}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), exact ν=0.
    /// z=0 forces z/x(z) Taylor path; all ν terms vanish.
    #[test]
    fn hagan_cev_exact_nu_zero() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.0).unwrap();
        for &k in &[80.0, 100.0, 120.0] {
            let expected = hagan_reference(100.0, k, 1.0, 2.0, 0.5, -0.3, 0.0);
            let actual = s.vol(k).unwrap().0;
            assert!(
                (actual - expected).abs() < expected * 1e-12,
                "ν=0, K={k}: expected {expected:.15e}, got {actual:.15e}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), T=0.01 (1-week expiry).
    /// Short expiry: correction factor ≈ 1.
    #[test]
    fn hagan_short_expiry_one_week() {
        let t = 0.01;
        let s = SabrSmile::new(100.0, t, 2.0, 0.5, -0.3, 0.4).unwrap();
        for &k in &[90.0, 95.0, 100.0, 105.0, 110.0] {
            let expected = hagan_reference(100.0, k, t, 2.0, 0.5, -0.3, 0.4);
            let actual = s.vol(k).unwrap().0;
            let eps = expected * 1e-12;
            assert!(
                (actual - expected).abs() < eps,
                "T=0.01, K={k}: expected {expected:.15e}, got {actual:.15e}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), T=10 (10-year expiry).
    /// Long expiry: correction factor is large.
    #[test]
    fn hagan_long_expiry_ten_year() {
        let t = 10.0;
        let s = SabrSmile::new(100.0, t, 2.0, 0.5, -0.3, 0.4).unwrap();
        for &k in &[70.0, 85.0, 100.0, 115.0, 130.0] {
            let expected = hagan_reference(100.0, k, t, 2.0, 0.5, -0.3, 0.4);
            let actual = s.vol(k).unwrap().0;
            let eps = expected * 1e-12;
            assert!(
                (actual - expected).abs() < eps,
                "T=10, K={k}: expected {expected:.15e}, got {actual:.15e}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), ATM continuity.
    /// Verifies K → F from both sides produces continuous vol.
    /// The unified code path (no ATM branch) ensures this by construction.
    #[test]
    fn hagan_atm_continuity_approach() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let vol_atm = s.vol(100.0).unwrap().0;

        // Approach from below and above
        for &delta in &[1e-3, 1e-5, 1e-7, 1e-9] {
            let v_below = s.vol(100.0 - delta).unwrap().0;
            let v_above = s.vol(100.0 + delta).unwrap().0;
            assert!(
                (v_below - vol_atm).abs() < delta * 0.1,
                "δ={delta}: vol(F-δ)={v_below} should approach vol(F)={vol_atm}"
            );
            assert!(
                (v_above - vol_atm).abs() < delta * 0.1,
                "δ={delta}: vol(F+δ)={v_above} should approach vol(F)={vol_atm}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), deep OTM strikes.
    /// K = 0.1F and K = 5F — formula remains valid for large log-moneyness.
    #[test]
    fn hagan_deep_otm_strikes() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();

        // Deep OTM put
        let k_low = 10.0; // K/F = 0.1
        let expected_low = hagan_reference(100.0, k_low, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual_low = s.vol(k_low).unwrap().0;
        assert!(
            (actual_low - expected_low).abs() < expected_low * 1e-12,
            "K=10: expected {expected_low:.15e}, got {actual_low:.15e}"
        );

        // Deep OTM call
        let k_high = 500.0; // K/F = 5
        let expected_high = hagan_reference(100.0, k_high, 1.0, 2.0, 0.5, -0.3, 0.4);
        let actual_high = s.vol(k_high).unwrap().0;
        assert!(
            (actual_high - expected_high).abs() < expected_high * 1e-12,
            "K=500: expected {expected_high:.15e}, got {actual_high:.15e}"
        );
    }

    /// Hagan (2002) Eq. (2.17a), second equity set with different params.
    /// F=50, T=0.25, α=1.5, β=0.5, ρ=−0.15, ν=0.25.
    #[test]
    fn hagan_equity_set2_strike_range() {
        let f = 50.0;
        let t = 0.25;
        let alpha = 1.5;
        let rho = -0.15;
        let nu = 0.25;
        let s = SabrSmile::new(f, t, alpha, 0.5, rho, nu).unwrap();

        for &k in &[35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0] {
            let expected = hagan_reference(f, k, t, alpha, 0.5, rho, nu);
            let actual = s.vol(k).unwrap().0;
            let eps = expected * 1e-12;
            assert!(
                (actual - expected).abs() < eps,
                "Set2, K={k}: expected {expected:.15e}, got {actual:.15e}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), Taylor/exact branch boundary.
    /// Verifies no discontinuity at |z| = 1e-6 where we switch between
    /// Taylor expansion and exact z/x(z) computation.
    #[test]
    fn hagan_taylor_exact_boundary_agreement() {
        // With α=2, β=0.5, ν=0.4, F=100: z = (ν/α) · (FK)^0.25 · ln(F/K)
        // For z ≈ 1e-6, need ln(F/K) ≈ 1e-6 · α/(ν · (FK)^0.25) ≈ 1e-6 · 2/(0.4·10) ≈ 5e-7
        // K ≈ F · exp(-5e-7) ≈ 99.99995
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();

        // Sample densely around the boundary
        let k_base = 100.0 * (-5e-7_f64).exp();
        let offsets = [-1e-5, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 1e-5];
        let vols: Vec<f64> = offsets
            .iter()
            .map(|&dk| s.vol(k_base + dk).unwrap().0)
            .collect();

        // All vols should be within 1e-10 of each other (smooth transition)
        for i in 1..vols.len() {
            assert!(
                (vols[i] - vols[i - 1]).abs() < 1e-8,
                "Discontinuity at Taylor boundary: vol[{i}]={:.15e}, vol[{}]={:.15e}",
                vols[i],
                i - 1,
                vols[i - 1]
            );
        }
    }

    /// Hagan (2002), Section 2: negative ρ produces downward-sloping skew.
    /// With β < 1, the combination of CEV backbone and negative ρ gives
    /// vol(K < F) > vol(K > F).
    #[test]
    fn hagan_skew_direction_negative_rho() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.5, 0.4).unwrap();
        let v_low = s.vol(80.0).unwrap().0;
        let v_atm = s.vol(100.0).unwrap().0;
        let v_high = s.vol(120.0).unwrap().0;
        assert!(
            v_low > v_atm,
            "ρ<0: vol(K<F)={v_low} should > vol(ATM)={v_atm}"
        );
        assert!(
            v_atm > v_high,
            "ρ<0: vol(ATM)={v_atm} should > vol(K>F)={v_high}"
        );
    }

    /// Hagan (2002), Section 2: positive ρ produces upward-sloping skew.
    #[test]
    fn hagan_skew_direction_positive_rho() {
        let s = SabrSmile::new(100.0, 1.0, 2.0, 0.5, 0.5, 0.4).unwrap();
        let v_low = s.vol(80.0).unwrap().0;
        let v_high = s.vol(120.0).unwrap().0;
        // Positive rho: OTM calls have higher vol than OTM puts (reversed skew)
        assert!(
            v_high > v_low,
            "ρ>0: vol(K>F)={v_high} should > vol(K<F)={v_low}"
        );
    }

    /// Calibration round-trip with ~50bps deterministic noise.
    /// Hagan (2002) parameters, equity-like. Tests robustness of the
    /// optimizer to noisy market data.
    #[test]
    fn calibrate_round_trip_noisy_equity() {
        let original = SabrSmile::new(100.0, 1.0, 2.0, 0.5, -0.3, 0.4).unwrap();
        let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();
        let clean_data = sabr_synthetic_data(&original, &strikes);

        // Deterministic noise (~50bps = 0.005 vol points)
        let noise = [
            0.003, -0.002, 0.005, -0.001, 0.004, -0.003, 0.002, -0.004, 0.001, -0.005, 0.003,
            -0.002, 0.004, -0.001, 0.002,
        ];
        let noisy_data: Vec<(f64, f64)> = clean_data
            .iter()
            .zip(noise.iter())
            .map(|(&(k, v), &n)| (k, v + n))
            .collect();

        let calibrated = SabrSmile::calibrate(100.0, 1.0, 0.5, &noisy_data).unwrap();

        // RMS against noisy data (fitted noise is absorbed)
        let rms: f64 = (noisy_data
            .iter()
            .map(|&(k, v)| (calibrated.vol(k).unwrap().0 - v).powi(2))
            .sum::<f64>()
            / noisy_data.len() as f64)
            .sqrt();
        assert!(
            rms < 0.01,
            "noisy calibration RMS {rms:.6e} should be < 0.01"
        );
    }

    /// Calibration round-trip with noise, lognormal (β=1).
    #[test]
    fn calibrate_round_trip_noisy_lognormal() {
        let original = SabrSmile::new(100.0, 1.0, 0.20, 1.0, -0.25, 0.3).unwrap();
        let strikes: Vec<f64> = (0..12).map(|i| 75.0 + 5.0 * i as f64).collect();
        let clean_data = sabr_synthetic_data(&original, &strikes);

        let noise = [
            0.002, -0.003, 0.001, -0.002, 0.004, -0.001, 0.003, -0.004, 0.002, -0.003, 0.001,
            -0.002,
        ];
        let noisy_data: Vec<(f64, f64)> = clean_data
            .iter()
            .zip(noise.iter())
            .map(|(&(k, v), &n)| (k, v + n))
            .collect();

        let calibrated = SabrSmile::calibrate(100.0, 1.0, 1.0, &noisy_data).unwrap();
        let rms: f64 = (noisy_data
            .iter()
            .map(|&(k, v)| (calibrated.vol(k).unwrap().0 - v).powi(2))
            .sum::<f64>()
            / noisy_data.len() as f64)
            .sqrt();
        assert!(
            rms < 0.01,
            "noisy lognormal calibration RMS {rms:.6e} should be < 0.01"
        );
    }

    /// Hagan (2002) Eq. (2.17a), β=1 pure lognormal (ν=0, ρ=0).
    /// With β=1, ν=0, ρ=0: σ_B(K) = α for all K (constant vol, Black model).
    #[test]
    fn hagan_pure_lognormal_constant_vol() {
        let alpha = 0.25;
        let s = SabrSmile::new(100.0, 1.0, alpha, 1.0, 0.0, 0.0).unwrap();
        for &k in &[50.0, 75.0, 100.0, 125.0, 150.0] {
            let actual = s.vol(k).unwrap().0;
            assert!(
                (actual - alpha).abs() < 1e-10,
                "Pure lognormal K={k}: expected α={alpha}, got {actual}"
            );
        }
    }

    /// Hagan (2002) Eq. (2.17a), β=0.5 with ρ=0 and ν=0 (pure CEV).
    /// Cross-validate ATM vol = α / F^(1-β) for the CEV backbone.
    #[test]
    fn hagan_cev_backbone_atm() {
        let alpha = 2.0;
        let beta = 0.5;
        let s = SabrSmile::new(100.0, 1.0, alpha, beta, 0.0, 0.0).unwrap();
        let expected_atm = alpha / 100.0_f64.powf(1.0 - beta);
        let actual = s.vol(100.0).unwrap().0;
        // Only correction term is (1-β)²/24 * α²/F^(2(1-β))
        let correction = 1.0 + 0.25 / 24.0 * alpha * alpha / (100.0_f64.powf(1.0 - beta)).powi(2);
        let expected_corrected = expected_atm * correction;
        assert!(
            (actual - expected_corrected).abs() < 1e-14,
            "CEV ATM: expected {expected_corrected:.15e}, got {actual:.15e}"
        );
    }
}
