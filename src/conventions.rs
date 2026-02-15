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
