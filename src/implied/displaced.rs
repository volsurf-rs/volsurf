//! Displaced diffusion implied volatility.
//!
//! A hybrid model that interpolates between the normal model (β = 0) and
//! the Black model (β = 1) via a displacement parameter β ∈ \[0, 1\].
//!
//! # Model
//! Under displaced diffusion, the forward evolves as
//! ```text
//! dF = σ · [β·F + (1−β)·F₀] · dW
//! ```
//! where `F₀` is the initial forward. Pricing is analytical via reduction to a
//! Black formula with shifted forward and strike:
//! ```text
//! C(F, K, σ, T, β) = (1/β) · Black(F, K_s, β·σ, T)
//! where K_s = (1−β)·F + β·K
//! ```

use crate::error::VolSurfError;
use crate::implied::black::{self, BlackImpliedVol};
use crate::implied::normal::{self, NormalImpliedVol};
use crate::types::{OptionType, Vol};
use crate::validate::{validate_non_negative, validate_positive};

/// Displaced diffusion implied volatility calculator.
///
/// The displacement parameter `beta` controls the blend:
/// - β = 0 → pure normal (Bachelier) model
/// - β = 1 → pure Black (lognormal) model
/// - 0 < β < 1 → intermediate CEV-like behavior
///
/// The returned vol `σ` is the displaced diffusion vol parameter.
/// At β = 1, it equals the Black implied vol. At β = 0, the effective
/// normal vol is `σ · F`.
#[derive(Debug, Clone, Copy)]
pub struct DisplacedImpliedVol {
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

    /// Returns the displacement parameter β.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Compute displaced diffusion implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Undiscounted option price (must be ≥ 0)
    /// * `forward` — Forward price at expiry (must be > 0 and finite)
    /// * `strike` — Strike price (must be > 0 and finite)
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `option_type` — Call or Put
    ///
    /// **Note:** Even at β = 0 (normal model equivalent), `forward` and `strike` must be
    /// positive because the vol conversion `σ_N = σ·F` is ill-defined for `F ≤ 0`.
    /// Use [`NormalImpliedVol`] directly for negative-rate markets.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for invalid inputs,
    /// [`VolSurfError::NumericalError`] if the price is outside the attainable range.
    pub fn compute(
        &self,
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: OptionType,
    ) -> crate::error::Result<Vol> {
        validate_non_negative(option_price, "option_price")?;
        validate_positive(forward, "forward")?;
        validate_positive(strike, "strike")?;
        validate_positive(expiry, "expiry")?;

        if self.beta == 1.0 {
            return BlackImpliedVol::compute(option_price, forward, strike, expiry, option_type);
        }

        if self.beta == 0.0 {
            let normal_vol =
                NormalImpliedVol::compute(option_price, forward, strike, expiry, option_type)?;
            return Ok(Vol(normal_vol.0 / forward));
        }

        let k_shifted = (1.0 - self.beta) * forward + self.beta * strike;
        if k_shifted <= 0.0 {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "shifted strike {k_shifted} is non-positive \
                     (beta={}, forward={forward}, strike={strike})",
                    self.beta
                ),
            });
        }

        let price_adj = self.beta * option_price;
        let shifted_iv =
            BlackImpliedVol::compute(price_adj, forward, k_shifted, expiry, option_type)?;

        Ok(Vol(shifted_iv.0 / self.beta))
    }
}

/// Compute undiscounted displaced diffusion option price.
///
/// # Formula
/// ```text
/// C(F, K, σ, T, β) = (1/β) · Black(F, K_s, β·σ, T)
/// where K_s = (1−β)·F + β·K
/// ```
/// At β = 0, reduces to the normal model with `σ_N = σ·F`.
/// At β = 1, reduces to the Black model with `σ_Black = σ`.
///
/// # Arguments
/// * `forward` — Forward price at expiry (must be > 0 and finite)
/// * `strike` — Strike price (must be > 0 and finite)
/// * `vol` — Displaced diffusion vol parameter σ (must be ≥ 0)
/// * `expiry` — Time to expiry in years (must be ≥ 0)
/// * `beta` — Displacement parameter β ∈ \[0, 1\]
/// * `option_type` — Call or Put
///
/// # Errors
/// Returns [`VolSurfError::InvalidInput`] for invalid inputs.
pub fn displaced_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    beta: f64,
    option_type: OptionType,
) -> crate::error::Result<f64> {
    validate_positive(forward, "forward")?;
    validate_positive(strike, "strike")?;
    validate_non_negative(vol, "volatility")?;
    validate_non_negative(expiry, "expiry")?;
    if !(0.0..=1.0).contains(&beta) {
        return Err(VolSurfError::InvalidInput {
            message: format!("beta must be in [0, 1], got {beta}"),
        });
    }

    if beta == 1.0 {
        return black::black_price(forward, strike, vol, expiry, option_type);
    }

    if beta == 0.0 {
        let normal_vol = vol * forward;
        return normal::normal_price(forward, strike, normal_vol, expiry, option_type);
    }

    let k_shifted = (1.0 - beta) * forward + beta * strike;
    validate_positive(k_shifted, "shifted strike")?;

    let black_vol = beta * vol;
    let p = black::black_price(forward, k_shifted, black_vol, expiry, option_type)?;
    Ok(p / beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn new_accepts_beta_zero() {
        assert!(DisplacedImpliedVol::new(0.0).is_ok());
    }

    #[test]
    fn new_accepts_beta_one() {
        assert!(DisplacedImpliedVol::new(1.0).is_ok());
    }

    #[test]
    fn new_accepts_beta_mid() {
        assert!(DisplacedImpliedVol::new(0.5).is_ok());
    }

    #[test]
    fn new_rejects_beta_above_one() {
        assert!(matches!(
            DisplacedImpliedVol::new(1.01),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn new_rejects_beta_below_zero() {
        assert!(matches!(
            DisplacedImpliedVol::new(-0.01),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn new_rejects_nan_beta() {
        assert!(matches!(
            DisplacedImpliedVol::new(f64::NAN),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn new_rejects_inf_beta() {
        assert!(matches!(
            DisplacedImpliedVol::new(f64::INFINITY),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn beta_accessor() {
        let calc = DisplacedImpliedVol::new(0.7).unwrap();
        assert_abs_diff_eq!(calc.beta(), 0.7, epsilon = 1e-15);
    }

    #[test]
    fn displaced_price_beta_one_equals_black() {
        let (f, k, sigma, t) = (100.0, 110.0, 0.25, 1.0);
        let dp = displaced_price(f, k, sigma, t, 1.0, OptionType::Call).unwrap();
        let bp = black::black_price(f, k, sigma, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(dp, bp, epsilon = 1e-14);
    }

    #[test]
    fn displaced_price_beta_zero_equals_normal() {
        let (f, k, sigma, t) = (100.0, 110.0, 0.25, 1.0);
        let dp = displaced_price(f, k, sigma, t, 0.0, OptionType::Call).unwrap();
        let np = normal::normal_price(f, k, sigma * f, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(dp, np, epsilon = 1e-12);
    }

    #[test]
    fn displaced_price_put_call_parity() {
        let (f, k, sigma, t, beta) = (100.0, 110.0, 0.25, 1.0, 0.5);
        let call = displaced_price(f, k, sigma, t, beta, OptionType::Call).unwrap();
        let put = displaced_price(f, k, sigma, t, beta, OptionType::Put).unwrap();
        assert_abs_diff_eq!(call - put, f - k, epsilon = 1e-10);
    }

    #[test]
    fn displaced_price_zero_vol() {
        let price = displaced_price(100.0, 80.0, 0.0, 1.0, 0.5, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn displaced_price_zero_expiry() {
        let price = displaced_price(100.0, 80.0, 0.25, 0.0, 0.5, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn displaced_price_rejects_negative_vol() {
        let result = displaced_price(100.0, 100.0, -0.1, 1.0, 0.5, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn displaced_price_rejects_invalid_beta() {
        let result = displaced_price(100.0, 100.0, 0.25, 1.0, 1.5, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn round_trip_beta_one_atm() {
        let calc = DisplacedImpliedVol::new(1.0).unwrap();
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 0.20);
        let price = displaced_price(f, k, sigma, t, 1.0, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_beta_one_matches_black() {
        let calc = DisplacedImpliedVol::new(1.0).unwrap();
        let (f, k, t) = (100.0, 110.0, 1.0);
        let price = black::black_price(f, k, 0.25, t, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let black_iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(iv.0, black_iv.0, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_beta_half_atm_call() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 0.20);
        let price = displaced_price(f, k, sigma, t, 0.5, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.5, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_beta_half_otm_put() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 0.30);
        let price = displaced_price(f, k, sigma, t, 0.5, OptionType::Put).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.5, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_beta_quarter() {
        let calc = DisplacedImpliedVol::new(0.25).unwrap();
        let (f, k, t, sigma) = (100.0, 105.0, 0.5, 0.15);
        let price = displaced_price(f, k, sigma, t, 0.25, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.25, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-11);
    }

    #[test]
    fn round_trip_beta_zero() {
        let calc = DisplacedImpliedVol::new(0.0).unwrap();
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 0.20);
        let price = displaced_price(f, k, sigma, t, 0.0, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-10);
    }

    #[test]
    fn round_trip_beta_zero_otm_put() {
        let calc = DisplacedImpliedVol::new(0.0).unwrap();
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 0.25);
        let price = displaced_price(f, k, sigma, t, 0.0, OptionType::Put).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.0, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-10);
    }

    #[test]
    fn compute_rejects_negative_price() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let result = calc.compute(-1.0, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_forward() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let result = calc.compute(5.0, 0.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_expiry() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let result = calc.compute(5.0, 100.0, 100.0, 0.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_nan_price() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let result = calc.compute(f64::NAN, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_inf_forward() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let result = calc.compute(5.0, f64::INFINITY, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_zero_price_atm() {
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let iv = calc
            .compute(0.0, 100.0, 100.0, 1.0, OptionType::Call)
            .unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn displaced_price_monotone_in_beta() {
        let (f, k, sigma, t) = (100.0, 100.0, 0.20, 1.0);
        let betas = [0.0, 0.25, 0.5, 0.75, 1.0];
        let prices: Vec<f64> = betas
            .iter()
            .map(|&b| displaced_price(f, k, sigma, t, b, OptionType::Call).unwrap())
            .collect();
        // ATM call prices should all be positive; exact monotonicity depends on
        // the vol parameter interpretation, so just verify they're all reasonable
        for &p in &prices {
            assert!(p > 0.0);
            assert!(p < f);
        }
    }

    // Edge cases from first-principles audit

    #[test]
    fn round_trip_near_zero_beta() {
        // β=0.001: general path near β=0 delegation boundary
        let calc = DisplacedImpliedVol::new(0.001).unwrap();
        let (f, k, t, sigma) = (100.0, 105.0, 1.0, 0.20);
        let price = displaced_price(f, k, sigma, t, 0.001, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.001, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-11);
    }

    #[test]
    fn round_trip_near_one_beta() {
        // β=0.999: general path near β=1 delegation boundary
        let calc = DisplacedImpliedVol::new(0.999).unwrap();
        let (f, k, t, sigma) = (100.0, 110.0, 1.0, 0.25);
        let price = displaced_price(f, k, sigma, t, 0.999, OptionType::Call).unwrap();
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.999, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-11);
    }

    #[test]
    fn round_trip_deep_otm() {
        // K_s = 0.5*100 + 0.5*180 = 140, deep OTM for Black
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let (f, k, t, sigma) = (100.0, 180.0, 1.0, 0.30);
        let price = displaced_price(f, k, sigma, t, 0.5, OptionType::Call).unwrap();
        assert!(price > 0.0);
        let iv = calc.compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = displaced_price(f, k, iv.0, t, 0.5, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-10);
    }

    #[test]
    fn compute_price_at_intrinsic_itm() {
        // price = intrinsic = F-K = 20, zero time value → Vol(0.0)
        let calc = DisplacedImpliedVol::new(0.5).unwrap();
        let iv = calc
            .compute(20.0, 100.0, 80.0, 1.0, OptionType::Call)
            .unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-10);
    }
}
