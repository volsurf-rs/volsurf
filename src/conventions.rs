//! Market conventions for volatility surfaces.
//!
//! Defines sticky-strike vs sticky-delta conventions and moneyness
//! transformations. The convention choice affects how Greeks are computed
//! and how the surface evolves as spot moves.

use serde::{Deserialize, Serialize};

use crate::error;
use crate::validate::{validate_finite, validate_non_negative, validate_positive};

/// Stickiness convention for the volatility surface.
///
/// - **Sticky strike**: vol at a fixed strike stays constant as spot moves.
///   Most common in equity markets.
/// - **Sticky delta**: vol at a fixed moneyness (delta) stays constant.
///   Common for index options and some FX markets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StickyKind {
    /// Vol at fixed strike is constant. Standard for single-stock equities.
    StickyStrike,
    /// Vol at fixed moneyness is constant. Standard for index/FX.
    StickyDelta,
}

/// Convert a strike to log-moneyness: k = ln(K / F).
///
/// Returns `Err` if `strike` or `forward` is non-positive, NaN, or infinite.
pub fn log_moneyness(strike: f64, forward: f64) -> error::Result<f64> {
    validate_positive(strike, "strike")?;
    validate_positive(forward, "forward")?;
    Ok((strike / forward).ln())
}

/// Convert a strike to simple moneyness: m = K / F.
///
/// Returns `Err` if `strike` or `forward` is non-positive, NaN, or infinite.
pub fn moneyness(strike: f64, forward: f64) -> error::Result<f64> {
    validate_positive(strike, "strike")?;
    validate_positive(forward, "forward")?;
    Ok(strike / forward)
}

/// Compute forward price from spot: F = S · exp((r − q) · T).
///
/// Returns `Err` if `spot` is non-positive, `rate` or `dividend_yield` is
/// non-finite, or `expiry` is negative.
pub fn forward_price(spot: f64, rate: f64, dividend_yield: f64, expiry: f64) -> error::Result<f64> {
    validate_positive(spot, "spot")?;
    validate_finite(rate, "rate")?;
    validate_finite(dividend_yield, "dividend_yield")?;
    validate_non_negative(expiry, "expiry")?;
    let result = spot * ((rate - dividend_yield) * expiry).exp();
    if !result.is_finite() {
        return Err(crate::error::VolSurfError::NumericalError {
            message: format!(
                "forward price overflow: spot={spot}, rate={rate}, q={dividend_yield}, T={expiry}"
            ),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn forward_price_known_value() {
        // spot=100, rate=0.05, q=0, expiry=1.0 → F = 100 * e^0.05 ≈ 105.127
        let f = forward_price(100.0, 0.05, 0.0, 1.0).unwrap();
        let expected = 100.0 * (0.05_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(f, 105.127109637602, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_zero_rate() {
        // Zero rate, zero yield → forward equals spot
        let f = forward_price(100.0, 0.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(f, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_negative_rate() {
        // Negative rate → forward < spot (inverted curve)
        let f = forward_price(100.0, -0.02, 0.0, 1.0).unwrap();
        let expected = 100.0 * (-0.02_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert!(f < 100.0, "negative rate should produce forward < spot");
    }

    #[test]
    fn forward_price_zero_expiry() {
        // Zero expiry → forward equals spot regardless of rate
        let f = forward_price(100.0, 0.05, 0.0, 0.0).unwrap();
        assert_abs_diff_eq!(f, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_scales_with_expiry() {
        // Doubling expiry should compound the rate effect
        let f1 = forward_price(100.0, 0.05, 0.0, 1.0).unwrap();
        let f2 = forward_price(100.0, 0.05, 0.0, 2.0).unwrap();
        let expected_ratio = (0.05_f64).exp();
        assert_abs_diff_eq!(f2 / f1, expected_ratio, epsilon = 1e-10);
    }

    #[test]
    fn log_moneyness_atm() {
        // ATM: strike == forward → k = ln(1) = 0
        let k = log_moneyness(100.0, 100.0).unwrap();
        assert_abs_diff_eq!(k, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn log_moneyness_itm_call() {
        // Strike < forward → k < 0
        let k = log_moneyness(95.0, 100.0).unwrap();
        let expected = (95.0_f64 / 100.0).ln();
        assert_abs_diff_eq!(k, expected, epsilon = 1e-10);
        assert!(k < 0.0, "ITM call should have negative log-moneyness");
    }

    #[test]
    fn log_moneyness_otm_call() {
        // Strike > forward → k > 0
        let k = log_moneyness(110.0, 100.0).unwrap();
        let expected = (110.0_f64 / 100.0).ln();
        assert_abs_diff_eq!(k, expected, epsilon = 1e-10);
        assert!(k > 0.0, "OTM call should have positive log-moneyness");
    }

    #[test]
    fn log_moneyness_symmetry() {
        // k(K/F) = -k(F/K)
        let k1 = log_moneyness(120.0, 100.0).unwrap();
        let k2 = log_moneyness(100.0, 120.0).unwrap();
        assert_abs_diff_eq!(k1, -k2, epsilon = 1e-10);
    }

    #[test]
    fn moneyness_atm() {
        // ATM: strike/forward = 1.0
        let m = moneyness(100.0, 100.0).unwrap();
        assert_abs_diff_eq!(m, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn moneyness_itm_call() {
        // Strike < forward → m < 1.0
        let m = moneyness(80.0, 100.0).unwrap();
        assert_abs_diff_eq!(m, 0.8, epsilon = 1e-10);
        assert!(m < 1.0, "ITM call should have moneyness < 1");
    }

    #[test]
    fn moneyness_otm_call() {
        // Strike > forward → m > 1.0
        let m = moneyness(120.0, 100.0).unwrap();
        assert_abs_diff_eq!(m, 1.2, epsilon = 1e-10);
        assert!(m > 1.0, "OTM call should have moneyness > 1");
    }

    #[test]
    fn moneyness_is_inverse_of_log() {
        // m = exp(k)
        let strike = 110.0;
        let forward = 100.0;
        let k = log_moneyness(strike, forward).unwrap();
        let m = moneyness(strike, forward).unwrap();
        assert_abs_diff_eq!(m, k.exp(), epsilon = 1e-10);
    }

    #[test]
    fn moneyness_consistency() {
        // moneyness(K, F) * moneyness(F, K) = 1
        let m1 = moneyness(120.0, 100.0).unwrap();
        let m2 = moneyness(100.0, 120.0).unwrap();
        assert_abs_diff_eq!(m1 * m2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn log_moneyness_zero_forward_returns_err() {
        assert!(log_moneyness(100.0, 0.0).is_err());
    }

    #[test]
    fn moneyness_zero_forward_returns_err() {
        assert!(moneyness(100.0, 0.0).is_err());
    }

    #[test]
    fn log_moneyness_zero_strike_returns_err() {
        assert!(log_moneyness(0.0, 100.0).is_err());
    }

    #[test]
    fn moneyness_zero_strike_returns_err() {
        assert!(moneyness(0.0, 100.0).is_err());
    }

    #[test]
    fn log_moneyness_nan_returns_err() {
        assert!(log_moneyness(f64::NAN, 100.0).is_err());
        assert!(log_moneyness(100.0, f64::NAN).is_err());
    }

    #[test]
    fn moneyness_nan_returns_err() {
        assert!(moneyness(f64::NAN, 100.0).is_err());
        assert!(moneyness(100.0, f64::NAN).is_err());
    }

    #[test]
    fn log_moneyness_inf_returns_err() {
        assert!(log_moneyness(f64::INFINITY, 100.0).is_err());
        assert!(log_moneyness(100.0, f64::INFINITY).is_err());
        assert!(log_moneyness(f64::NEG_INFINITY, 100.0).is_err());
        assert!(log_moneyness(100.0, f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn moneyness_inf_returns_err() {
        assert!(moneyness(f64::INFINITY, 100.0).is_err());
        assert!(moneyness(100.0, f64::INFINITY).is_err());
        assert!(moneyness(f64::NEG_INFINITY, 100.0).is_err());
        assert!(moneyness(100.0, f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn forward_price_non_finite_returns_err() {
        assert!(forward_price(f64::NAN, 0.05, 0.0, 1.0).is_err());
        assert!(forward_price(100.0, f64::NAN, 0.0, 1.0).is_err());
        assert!(forward_price(100.0, 0.05, f64::NAN, 1.0).is_err());
        assert!(forward_price(100.0, 0.05, 0.0, f64::NAN).is_err());
        assert!(forward_price(f64::INFINITY, 0.05, 0.0, 1.0).is_err());
        assert!(forward_price(100.0, f64::INFINITY, 0.0, 1.0).is_err());
        assert!(forward_price(100.0, 0.05, f64::NEG_INFINITY, 1.0).is_err());
        assert!(forward_price(100.0, 0.05, 0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn forward_price_overflow_returns_numerical_error() {
        use crate::error::VolSurfError;
        // exp(50*20) = exp(1000) = Inf
        let err = forward_price(100.0, 50.0, 0.0, 20.0).unwrap_err();
        assert!(matches!(err, VolSurfError::NumericalError { .. }));
    }

    #[test]
    fn forward_price_large_but_valid_succeeds() {
        // exp(500) ≈ 1.4e217 — large but finite
        let f = forward_price(1.0, 1.0, 0.0, 500.0).unwrap();
        assert!(f.is_finite());
        assert!(f > 1e200);
    }

    #[test]
    fn forward_price_negative_expiry_returns_err() {
        assert!(forward_price(100.0, 0.05, 0.0, -1.0).is_err());
    }

    #[test]
    fn forward_price_zero_spot_returns_err() {
        assert!(forward_price(0.0, 0.05, 0.0, 1.0).is_err());
    }

    #[test]
    fn log_moneyness_negative_strike_returns_err() {
        assert!(log_moneyness(-50.0, 100.0).is_err());
    }

    #[test]
    fn log_moneyness_negative_forward_returns_err() {
        assert!(log_moneyness(100.0, -50.0).is_err());
    }

    #[test]
    fn moneyness_negative_strike_returns_err() {
        assert!(moneyness(-50.0, 100.0).is_err());
    }

    #[test]
    fn moneyness_negative_forward_returns_err() {
        assert!(moneyness(100.0, -50.0).is_err());
    }

    #[test]
    fn forward_price_negative_spot_returns_err() {
        assert!(forward_price(-100.0, 0.05, 0.0, 1.0).is_err());
    }

    #[test]
    fn log_moneyness_error_is_invalid_input() {
        use crate::error::VolSurfError;
        let err = log_moneyness(100.0, 0.0).unwrap_err();
        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
    }

    #[test]
    fn moneyness_error_is_invalid_input() {
        use crate::error::VolSurfError;
        let err = moneyness(f64::NAN, 100.0).unwrap_err();
        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
    }

    #[test]
    fn forward_price_error_is_invalid_input() {
        use crate::error::VolSurfError;
        let err = forward_price(100.0, 0.05, 0.0, -1.0).unwrap_err();
        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
    }

    #[test]
    fn log_moneyness_large_finite_values_succeed() {
        let k = log_moneyness(1e300, 1e300).unwrap();
        assert_abs_diff_eq!(k, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn moneyness_large_finite_values_succeed() {
        let m = moneyness(1e300, 1e300).unwrap();
        assert_abs_diff_eq!(m, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn moneyness_extreme_ratio_overflows() {
        // Both inputs valid (positive, finite), but ratio overflows to Inf.
        // Documents current behavior: input guards don't catch output overflow.
        let m = moneyness(100.0, f64::MIN_POSITIVE).unwrap();
        assert!(m.is_infinite());
    }

    #[test]
    fn forward_price_positive_dividend_yield() {
        // r=0.05, q=0.02 → F = 100 * exp(0.03) ≈ 103.045
        let f = forward_price(100.0, 0.05, 0.02, 1.0).unwrap();
        let expected = 100.0 * (0.03_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert!(f < forward_price(100.0, 0.05, 0.0, 1.0).unwrap());
    }

    #[test]
    fn forward_price_yield_equals_rate() {
        // r = q → cost of carry is zero → F = S
        let f = forward_price(100.0, 0.05, 0.05, 1.0).unwrap();
        assert_abs_diff_eq!(f, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_yield_exceeds_rate() {
        // q > r → forward < spot (high-dividend stock)
        let f = forward_price(100.0, 0.03, 0.05, 1.0).unwrap();
        let expected = 100.0 * (-0.02_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert!(f < 100.0);
    }

    #[test]
    fn forward_price_negative_dividend_yield() {
        // Negative q (e.g. convenience yield on commodity) → higher forward
        let f = forward_price(100.0, 0.05, -0.02, 1.0).unwrap();
        let expected = 100.0 * (0.07_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert!(f > forward_price(100.0, 0.05, 0.0, 1.0).unwrap());
    }

    // Gap #5: StickyKind enum

    #[test]
    fn sticky_kind_debug_display() {
        assert_eq!(format!("{:?}", StickyKind::StickyStrike), "StickyStrike");
        assert_eq!(format!("{:?}", StickyKind::StickyDelta), "StickyDelta");
    }

    #[test]
    fn sticky_kind_copy_and_eq() {
        let a = StickyKind::StickyStrike;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = StickyKind::StickyDelta;
        assert_ne!(a, c);
    }

    #[test]
    fn sticky_kind_serde_round_trip() {
        for kind in [StickyKind::StickyStrike, StickyKind::StickyDelta] {
            let json = serde_json::to_string(&kind).unwrap();
            let kind2: StickyKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, kind2);
        }
    }

    #[test]
    fn sticky_kind_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StickyKind::StickyStrike);
        set.insert(StickyKind::StickyDelta);
        set.insert(StickyKind::StickyStrike); // duplicate
        assert_eq!(set.len(), 2);
    }
}
