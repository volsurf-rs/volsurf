//! Market conventions for volatility surfaces.
//!
//! Defines sticky-strike vs sticky-delta conventions and moneyness
//! transformations. The convention choice affects how Greeks are computed
//! and how the surface evolves as spot moves.

use serde::{Deserialize, Serialize};

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
pub fn log_moneyness(strike: f64, forward: f64) -> f64 {
    (strike / forward).ln()
}

/// Convert a strike to simple moneyness: m = K / F.
pub fn moneyness(strike: f64, forward: f64) -> f64 {
    strike / forward
}

/// Compute forward price from spot: F = S · exp(r · T).
pub fn forward_price(spot: f64, rate: f64, expiry: f64) -> f64 {
    spot * (rate * expiry).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- forward_price() tests ---

    #[test]
    fn forward_price_known_value() {
        // spot=100, rate=0.05, expiry=1.0 → F = 100 * e^0.05 ≈ 105.127
        let f = forward_price(100.0, 0.05, 1.0);
        let expected = 100.0 * (0.05_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(f, 105.127109637602, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_zero_rate() {
        // Zero rate → forward equals spot
        let f = forward_price(100.0, 0.0, 1.0);
        assert_abs_diff_eq!(f, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_negative_rate() {
        // Negative rate → forward < spot (inverted curve)
        let f = forward_price(100.0, -0.02, 1.0);
        let expected = 100.0 * (-0.02_f64).exp();
        assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        assert!(f < 100.0, "negative rate should produce forward < spot");
    }

    #[test]
    fn forward_price_zero_expiry() {
        // Zero expiry → forward equals spot regardless of rate
        let f = forward_price(100.0, 0.05, 0.0);
        assert_abs_diff_eq!(f, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_price_scales_with_expiry() {
        // Doubling expiry should compound the rate effect
        let f1 = forward_price(100.0, 0.05, 1.0);
        let f2 = forward_price(100.0, 0.05, 2.0);
        let expected_ratio = (0.05_f64).exp();
        assert_abs_diff_eq!(f2 / f1, expected_ratio, epsilon = 1e-10);
    }

    // --- log_moneyness() tests ---

    #[test]
    fn log_moneyness_atm() {
        // ATM: strike == forward → k = ln(1) = 0
        let k = log_moneyness(100.0, 100.0);
        assert_abs_diff_eq!(k, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn log_moneyness_itm_call() {
        // Strike < forward → k < 0
        let k = log_moneyness(95.0, 100.0);
        let expected = (95.0_f64 / 100.0).ln();
        assert_abs_diff_eq!(k, expected, epsilon = 1e-10);
        assert!(k < 0.0, "ITM call should have negative log-moneyness");
    }

    #[test]
    fn log_moneyness_otm_call() {
        // Strike > forward → k > 0
        let k = log_moneyness(110.0, 100.0);
        let expected = (110.0_f64 / 100.0).ln();
        assert_abs_diff_eq!(k, expected, epsilon = 1e-10);
        assert!(k > 0.0, "OTM call should have positive log-moneyness");
    }

    #[test]
    fn log_moneyness_symmetry() {
        // k(K/F) = -k(F/K)
        let k1 = log_moneyness(120.0, 100.0);
        let k2 = log_moneyness(100.0, 120.0);
        assert_abs_diff_eq!(k1, -k2, epsilon = 1e-10);
    }

    // --- moneyness() tests ---

    #[test]
    fn moneyness_atm() {
        // ATM: strike/forward = 1.0
        let m = moneyness(100.0, 100.0);
        assert_abs_diff_eq!(m, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn moneyness_itm_call() {
        // Strike < forward → m < 1.0
        let m = moneyness(80.0, 100.0);
        assert_abs_diff_eq!(m, 0.8, epsilon = 1e-10);
        assert!(m < 1.0, "ITM call should have moneyness < 1");
    }

    #[test]
    fn moneyness_otm_call() {
        // Strike > forward → m > 1.0
        let m = moneyness(120.0, 100.0);
        assert_abs_diff_eq!(m, 1.2, epsilon = 1e-10);
        assert!(m > 1.0, "OTM call should have moneyness > 1");
    }

    #[test]
    fn moneyness_is_inverse_of_log() {
        // m = exp(k)
        let strike = 110.0;
        let forward = 100.0;
        let k = log_moneyness(strike, forward);
        let m = moneyness(strike, forward);
        assert_abs_diff_eq!(m, k.exp(), epsilon = 1e-10);
    }

    #[test]
    fn moneyness_consistency() {
        // moneyness(K, F) * moneyness(F, K) = 1
        let m1 = moneyness(120.0, 100.0);
        let m2 = moneyness(100.0, 120.0);
        assert_abs_diff_eq!(m1 * m2, 1.0, epsilon = 1e-10);
    }

    // --- Gap #3: log_moneyness/moneyness with zero forward ---

    #[test]
    fn log_moneyness_zero_forward_returns_inf() {
        let k = log_moneyness(100.0, 0.0);
        assert!(
            k.is_infinite(),
            "log_moneyness(100, 0) should be Inf, got {k}"
        );
        assert!(k.is_sign_positive());
    }

    #[test]
    fn moneyness_zero_forward_returns_inf() {
        let m = moneyness(100.0, 0.0);
        assert!(m.is_infinite(), "moneyness(100, 0) should be Inf, got {m}");
    }

    #[test]
    fn log_moneyness_zero_strike_returns_neg_inf() {
        let k = log_moneyness(0.0, 100.0);
        assert!(
            k.is_infinite() && k.is_sign_negative(),
            "log_moneyness(0, 100) should be -Inf, got {k}"
        );
    }

    #[test]
    fn moneyness_zero_strike_returns_zero() {
        let m = moneyness(0.0, 100.0);
        assert_abs_diff_eq!(m, 0.0, epsilon = 1e-15);
    }

    // --- Gap #4: NaN and Inf inputs ---

    #[test]
    fn log_moneyness_nan_strike_returns_nan() {
        let k = log_moneyness(f64::NAN, 100.0);
        assert!(k.is_nan(), "log_moneyness(NaN, 100) should be NaN, got {k}");
    }

    #[test]
    fn log_moneyness_nan_forward_returns_nan() {
        let k = log_moneyness(100.0, f64::NAN);
        assert!(k.is_nan(), "log_moneyness(100, NaN) should be NaN, got {k}");
    }

    #[test]
    fn moneyness_nan_inputs_return_nan() {
        assert!(moneyness(f64::NAN, 100.0).is_nan());
        assert!(moneyness(100.0, f64::NAN).is_nan());
    }

    #[test]
    fn log_moneyness_inf_strike_returns_inf() {
        let k = log_moneyness(f64::INFINITY, 100.0);
        assert!(k.is_infinite() && k.is_sign_positive());
    }

    #[test]
    fn moneyness_inf_strike_returns_inf() {
        let m = moneyness(f64::INFINITY, 100.0);
        assert!(m.is_infinite());
    }

    #[test]
    fn forward_price_nan_inputs() {
        assert!(forward_price(f64::NAN, 0.05, 1.0).is_nan());
        assert!(forward_price(100.0, f64::NAN, 1.0).is_nan());
        assert!(forward_price(100.0, 0.05, f64::NAN).is_nan());
    }

    // --- Gap #5: StickyKind enum ---

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
