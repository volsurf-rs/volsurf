//! Bachelier (normal) implied volatility.
//!
//! Used in markets where the normal model is standard (short-dated FX, fixed income).
//! The Bachelier formula assumes arithmetic (Gaussian) returns rather than lognormal.
//!
//! # Formula
//! The undiscounted Bachelier call price is
//! ```text
//! C(F, K, T, σ_N) = (F − K)·Φ(d) + σ_N·√T·φ(d)
//! where d = (F − K) / (σ_N·√T)
//! ```
//! with `σ_N` in absolute (price) units per √year, `Φ` the standard normal CDF,
//! and `φ` the standard normal PDF.
//!
//! Wraps the [`implied_vol`] crate for near-machine-precision extraction.

use implied_vol::{DefaultSpecialFn, ImpliedNormalVolatility, PriceBachelier};

use crate::error::VolSurfError;
use crate::types::{OptionType, Vol};
use crate::validate::{validate_finite, validate_non_negative, validate_positive};

/// Bachelier (normal) implied volatility calculator.
///
/// Extracts implied volatility under the normal model:
/// `dF = σ_N · dW` (arithmetic Brownian motion).
///
/// The returned volatility `σ_N` is in **absolute price units** per √year,
/// unlike Black implied vol which is a dimensionless annualized proportion.
///
/// Uses Jäckel's algorithm from the [`implied_vol`] crate for
/// near-machine-precision extraction.
///
/// # References
/// - Bachelier, L. "Théorie de la spéculation" (1900)
/// - Jäckel, P. "Implied Normal Volatility" (2017)
#[derive(Debug)]
pub struct NormalImpliedVol;

impl NormalImpliedVol {
    /// Compute normal (Bachelier) implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Undiscounted option price (must be ≥ 0)
    /// * `forward` — Forward price at expiry (must be finite; may be negative for rates)
    /// * `strike` — Strike price (must be finite; may be negative for rates)
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `option_type` — Call or Put
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] for invalid inputs,
    /// [`VolSurfError::NumericalError`] if the price is outside the attainable range.
    pub fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: OptionType,
    ) -> crate::error::Result<Vol> {
        validate_non_negative(option_price, "option_price")?;
        validate_finite(forward, "forward")?;
        validate_finite(strike, "strike")?;
        validate_positive(expiry, "expiry")?;

        let is_call = matches!(option_type, OptionType::Call);

        let iv = ImpliedNormalVolatility::builder()
            .option_price(option_price)
            .forward(forward)
            .strike(strike)
            .expiry(expiry)
            .is_call(is_call)
            .build()
            .ok_or_else(|| VolSurfError::InvalidInput {
                message: "implied-vol rejected inputs as outside model domain".into(),
            })?;

        let sigma =
            iv.calculate::<DefaultSpecialFn>()
                .ok_or_else(|| VolSurfError::NumericalError {
                    message: "option price is outside the attainable range".into(),
                })?;

        Ok(Vol(sigma))
    }
}

/// Compute undiscounted Bachelier (normal model) option price.
///
/// # Formula
/// ```text
/// C(F, K, T, σ_N) = (F − K)·Φ(d) + σ_N·√T·φ(d)
/// P(F, K, T, σ_N) = (K − F)·Φ(−d) + σ_N·√T·φ(d)
/// where d = (F − K) / (σ_N·√T)
/// ```
///
/// # Arguments
/// * `forward` — Forward price at expiry (must be finite; may be negative for rates)
/// * `strike` — Strike price (must be finite; may be negative for rates)
/// * `vol` — Normal (Bachelier) volatility `σ_N` in price units/√year (must be ≥ 0)
/// * `expiry` — Time to expiry in years (must be ≥ 0)
/// * `option_type` — Call or Put
///
/// # Errors
/// Returns [`VolSurfError::InvalidInput`] for invalid inputs.
pub fn normal_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: OptionType,
) -> crate::error::Result<f64> {
    validate_finite(forward, "forward")?;
    validate_finite(strike, "strike")?;
    validate_non_negative(vol, "volatility")?;
    validate_non_negative(expiry, "expiry")?;

    let is_call = matches!(option_type, OptionType::Call);

    let price = PriceBachelier::builder()
        .forward(forward)
        .strike(strike)
        .volatility(vol)
        .expiry(expiry)
        .is_call(is_call)
        .build()
        .ok_or_else(|| VolSurfError::InvalidInput {
            message: "implied-vol rejected pricing inputs as outside model domain".into(),
        })?
        .calculate::<DefaultSpecialFn>();

    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normal_price_put_call_parity() {
        let (f, k, sigma, t) = (100.0, 110.0, 20.0, 1.0);
        let call = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let put = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(call - put, f - k, epsilon = 1e-10);
    }

    #[test]
    fn normal_price_zero_vol_itm_call() {
        let price = normal_price(100.0, 80.0, 0.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_zero_vol_otm_call() {
        let price = normal_price(100.0, 120.0, 0.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_zero_vol_itm_put() {
        let price = normal_price(100.0, 120.0, 0.0, 1.0, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_zero_vol_otm_put() {
        let price = normal_price(100.0, 80.0, 0.0, 1.0, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_zero_expiry() {
        let price = normal_price(100.0, 80.0, 20.0, 0.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_atm_call() {
        // ATM Bachelier: C = σ√T / √(2π)
        let (f, k, sigma, t) = (100.0, 100.0, 20.0, 1.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let expected = sigma * t.sqrt() / (2.0 * std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(price, expected, epsilon = 1e-12);
    }

    #[test]
    fn normal_price_rejects_negative_vol() {
        let result = normal_price(100.0, 100.0, -1.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn normal_price_rejects_negative_expiry() {
        let result = normal_price(100.0, 100.0, 20.0, -1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn normal_price_rejects_nan_forward() {
        let result = normal_price(f64::NAN, 100.0, 20.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn normal_price_rejects_inf_strike() {
        let result = normal_price(100.0, f64::INFINITY, 20.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn normal_price_negative_forward() {
        // Normal model supports negative forwards (rates markets)
        let call = normal_price(-1.0, -0.5, 0.5, 1.0, OptionType::Call).unwrap();
        let put = normal_price(-1.0, -0.5, 0.5, 1.0, OptionType::Put).unwrap();
        assert_abs_diff_eq!(call - put, -1.0 - (-0.5), epsilon = 1e-10);
    }

    #[test]
    fn round_trip_atm_call() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_atm_put() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_itm_call() {
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 15.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_otm_call() {
        let (f, k, t, sigma) = (100.0, 120.0, 1.0, 25.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_itm_put() {
        let (f, k, t, sigma) = (100.0, 120.0, 1.0, 25.0);
        let price = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_otm_put() {
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 15.0);
        let price = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_short_expiry() {
        let (f, k, t, sigma) = (100.0, 100.0, 0.01, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_long_expiry() {
        let (f, k, t, sigma) = (100.0, 100.0, 10.0, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_high_vol() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 80.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_negative_forward() {
        // Normal model supports negative forwards (rates markets)
        let (f, k, t, sigma) = (-0.5, -0.3, 1.0, 0.5);
        let price = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_deep_otm_call() {
        let (f, k, t, sigma) = (100.0, 160.0, 1.0, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        assert!(price > 0.0, "deep OTM normal price should be positive");
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn compute_rejects_negative_price() {
        let result = NormalImpliedVol::compute(-1.0, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_nan_price() {
        let result = NormalImpliedVol::compute(f64::NAN, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_expiry() {
        let result = NormalImpliedVol::compute(5.0, 100.0, 100.0, 0.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_negative_expiry() {
        let result = NormalImpliedVol::compute(5.0, 100.0, 100.0, -1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_nan_forward() {
        let result = NormalImpliedVol::compute(5.0, f64::NAN, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_inf_strike() {
        let result = NormalImpliedVol::compute(5.0, 100.0, f64::INFINITY, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_price_below_intrinsic() {
        // Call with F=100, K=80, intrinsic=20 — price=15 is below intrinsic
        let result = NormalImpliedVol::compute(15.0, 100.0, 80.0, 1.0, OptionType::Call);
        assert!(result.is_err());
    }

    #[test]
    fn compute_zero_price_otm() {
        let iv = NormalImpliedVol::compute(0.0, 100.0, 120.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn compute_zero_price_atm() {
        let iv = NormalImpliedVol::compute(0.0, 100.0, 100.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-10);
    }

    // Edge cases from Jäckel (2017) paper audit

    #[test]
    fn round_trip_deep_otm_underflow() {
        // d = (100-900)/20 = -40, beyond paper's -38.278 double precision limit
        let (f, k, t, sigma) = (100.0, 900.0, 1.0, 20.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        assert!(price < 1e-100, "deep OTM should underflow to near zero");
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn round_trip_small_sigma() {
        // d = (100-100.5)/0.01 = -50, past paper's -38.278 double precision limit.
        // Price underflows — verify self-consistent round-trip, not sigma recovery.
        let (f, k, t, sigma) = (100.0, 100.5, 1.0, 0.01);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_large_negative_rates() {
        let (f, k, t, sigma) = (-5.0, -3.0, 2.0, 2.0);
        let call = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        let put = normal_price(f, k, sigma, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(call - put, f - k, epsilon = 1e-10);

        let iv_call = NormalImpliedVol::compute(call, f, k, t, OptionType::Call).unwrap();
        let iv_put = NormalImpliedVol::compute(put, f, k, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(iv_call.0, sigma, epsilon = 1e-10);
        assert_abs_diff_eq!(iv_put.0, sigma, epsilon = 1e-10);
    }

    #[test]
    fn round_trip_very_high_vol() {
        // Normal model has no upper price bound (unlike Black)
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 1000.0);
        let price = normal_price(f, k, sigma, t, OptionType::Call).unwrap();
        assert!(price > 100.0, "very high vol should give very large price");
        let iv = NormalImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = normal_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-8);
    }

    #[test]
    fn compute_price_at_intrinsic_itm() {
        // Price exactly at intrinsic (zero time value) => Vol(0.0)
        let iv = NormalImpliedVol::compute(20.0, 100.0, 80.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(iv.0, 0.0, epsilon = 1e-10);
    }
}
