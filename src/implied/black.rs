//! Black (lognormal) implied volatility via Jäckel's "Let's Be Rational" algorithm.
//!
//! Wraps the [`implied_vol`] crate which achieves 3 ULP accuracy.

use implied_vol::{DefaultSpecialFn, ImpliedBlackVolatility, PriceBlackScholes};

use crate::error::VolSurfError;
use crate::types::{OptionType, Vol};

/// Black (lognormal) implied volatility calculator.
///
/// Uses Peter Jäckel's rational approximation algorithm (2013) for
/// near-machine-precision extraction of Black implied volatility.
///
/// # References
/// - Jäckel, P. "Let's Be Rational" (2013)
pub struct BlackImpliedVol;

impl BlackImpliedVol {
    /// Compute Black implied volatility from an option price.
    ///
    /// # Arguments
    /// * `option_price` — Undiscounted option price (must be >= 0)
    /// * `forward` — Forward price at expiry (must be > 0 and finite)
    /// * `strike` — Strike price (must be > 0 and finite)
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
        if option_price < 0.0 || option_price.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("option_price must be non-negative, got {option_price}"),
            });
        }
        if forward <= 0.0 || !forward.is_finite() {
            return Err(VolSurfError::InvalidInput {
                message: format!("forward must be positive and finite, got {forward}"),
            });
        }
        if strike <= 0.0 || !strike.is_finite() {
            return Err(VolSurfError::InvalidInput {
                message: format!("strike must be positive and finite, got {strike}"),
            });
        }
        if expiry <= 0.0 || expiry.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("expiry must be positive, got {expiry}"),
            });
        }

        let is_call = matches!(option_type, OptionType::Call);

        let iv = ImpliedBlackVolatility::builder()
            .option_price(option_price)
            .forward(forward)
            .strike(strike)
            .expiry(expiry)
            .is_call(is_call)
            .build()
            .ok_or_else(|| {
                VolSurfError::InvalidInput {
                    message: "implied-vol rejected inputs as outside model domain".into(),
                }
            })?;

        let sigma = iv.calculate::<DefaultSpecialFn>().ok_or_else(|| {
            VolSurfError::NumericalError {
                message: "option price is outside the attainable range".into(),
            }
        })?;

        Ok(Vol(sigma))
    }
}

/// Compute undiscounted Black-Scholes option price.
///
/// # Arguments
/// * `forward` — Forward price at expiry (must be > 0 and finite)
/// * `strike` — Strike price (must be > 0 and finite)
/// * `vol` — Black implied volatility σ (must be >= 0)
/// * `expiry` — Time to expiry in years (must be >= 0)
/// * `option_type` — Call or Put
///
/// # Errors
/// Returns [`VolSurfError::InvalidInput`] for invalid inputs.
pub fn black_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: OptionType,
) -> crate::error::Result<f64> {
    if forward <= 0.0 || !forward.is_finite() {
        return Err(VolSurfError::InvalidInput {
            message: format!("forward must be positive and finite, got {forward}"),
        });
    }
    if strike <= 0.0 || !strike.is_finite() {
        return Err(VolSurfError::InvalidInput {
            message: format!("strike must be positive and finite, got {strike}"),
        });
    }
    if vol < 0.0 || vol.is_nan() {
        return Err(VolSurfError::InvalidInput {
            message: format!("volatility must be non-negative, got {vol}"),
        });
    }
    if expiry < 0.0 || expiry.is_nan() {
        return Err(VolSurfError::InvalidInput {
            message: format!("expiry must be non-negative, got {expiry}"),
        });
    }

    let is_call = matches!(option_type, OptionType::Call);

    let price = PriceBlackScholes::builder()
        .forward(forward)
        .strike(strike)
        .volatility(vol)
        .expiry(expiry)
        .is_call(is_call)
        .build()
        .ok_or_else(|| {
            VolSurfError::InvalidInput {
                message: "implied-vol rejected pricing inputs as outside model domain".into(),
            }
        })?
        .calculate::<DefaultSpecialFn>();

    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Round-trip tests: price → IV → reprice ---

    #[test]
    fn round_trip_atm_call() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 0.20);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_atm_put() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 0.20);
        let price = black_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_itm_call() {
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 0.25);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_otm_call() {
        let (f, k, t, sigma) = (100.0, 120.0, 1.0, 0.30);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_deep_otm_call() {
        let (f, k, t, sigma) = (100.0, 200.0, 1.0, 0.20);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        assert!(price > 0.0, "deep OTM price should be positive");
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_high_vol() {
        let (f, k, t, sigma) = (100.0, 100.0, 1.0, 1.0);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_short_expiry() {
        let (f, k, t, sigma) = (100.0, 100.0, 0.01, 0.20);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_long_expiry() {
        let (f, k, t, sigma) = (100.0, 100.0, 10.0, 0.20);
        let price = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Call).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_otm_put() {
        let (f, k, t, sigma) = (100.0, 80.0, 1.0, 0.25);
        let price = black_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_itm_put() {
        let (f, k, t, sigma) = (100.0, 120.0, 1.0, 0.30);
        let price = black_price(f, k, sigma, t, OptionType::Put).unwrap();
        let iv = BlackImpliedVol::compute(price, f, k, t, OptionType::Put).unwrap();
        let reprice = black_price(f, k, iv.0, t, OptionType::Put).unwrap();
        assert_abs_diff_eq!(price, reprice, epsilon = 1e-12);
    }

    // --- black_price tests ---

    #[test]
    fn black_price_call_put_parity() {
        let (f, k, t, sigma) = (100.0, 110.0, 1.0, 0.25);
        let call = black_price(f, k, sigma, t, OptionType::Call).unwrap();
        let put = black_price(f, k, sigma, t, OptionType::Put).unwrap();
        // Call - Put = Forward - Strike (undiscounted put-call parity)
        assert_abs_diff_eq!(call - put, f - k, epsilon = 1e-10);
    }

    #[test]
    fn black_price_zero_vol_call() {
        // Zero vol: call = max(F - K, 0)
        let price = black_price(100.0, 80.0, 0.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn black_price_zero_vol_otm_call() {
        // Zero vol, OTM call: max(F - K, 0) = 0
        let price = black_price(100.0, 120.0, 0.0, 1.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn black_price_zero_expiry() {
        // Zero expiry: intrinsic value
        let price = black_price(100.0, 80.0, 0.20, 0.0, OptionType::Call).unwrap();
        assert_abs_diff_eq!(price, 20.0, epsilon = 1e-12);
    }

    // --- Validation error tests ---

    #[test]
    fn compute_rejects_negative_price() {
        let result = BlackImpliedVol::compute(-1.0, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_nan_price() {
        let result =
            BlackImpliedVol::compute(f64::NAN, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_forward() {
        let result = BlackImpliedVol::compute(5.0, 0.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_negative_forward() {
        let result = BlackImpliedVol::compute(5.0, -1.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_infinite_forward() {
        let result =
            BlackImpliedVol::compute(5.0, f64::INFINITY, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_strike() {
        let result = BlackImpliedVol::compute(5.0, 100.0, 0.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_zero_expiry() {
        let result = BlackImpliedVol::compute(5.0, 100.0, 100.0, 0.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_negative_expiry() {
        let result = BlackImpliedVol::compute(5.0, 100.0, 100.0, -1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn compute_rejects_price_above_forward() {
        // Call price cannot exceed forward price
        let result =
            BlackImpliedVol::compute(150.0, 100.0, 100.0, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::NumericalError { .. })));
    }

    #[test]
    fn black_price_rejects_negative_vol() {
        let result = black_price(100.0, 100.0, -0.1, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn black_price_rejects_negative_expiry() {
        let result = black_price(100.0, 100.0, 0.2, -1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn black_price_rejects_zero_forward() {
        let result = black_price(0.0, 100.0, 0.2, 1.0, OptionType::Call);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }
}
