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
}
