//! Arbitrage detection for volatility smiles.
//!
//! Butterfly arbitrage occurs when the risk-neutral density implied by option
//! prices becomes negative, violating the no-arbitrage condition.
//!
//! # References
//! - Breeden, D.T. & Litzenberger, R.H. "Prices of State-Contingent Claims
//!   Implicit in Option Prices" (1978)

use serde::{Deserialize, Serialize};

/// Report on arbitrage-freeness of a smile or surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageReport {
    /// Whether the smile/surface is free of detected arbitrage.
    pub is_free: bool,
    /// Butterfly spread violations (negative density regions).
    pub butterfly_violations: Vec<ButterflyViolation>,
}

impl ArbitrageReport {
    /// Create a report indicating no arbitrage was found.
    pub fn clean() -> Self {
        Self {
            is_free: true,
            butterfly_violations: Vec::new(),
        }
    }
}

/// A butterfly spread arbitrage violation at a specific strike.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ButterflyViolation {
    /// Strike where the violation occurs.
    pub strike: f64,
    /// Risk-neutral density value (negative indicates violation).
    pub density: f64,
    /// Absolute magnitude of the violation.
    pub magnitude: f64,
}
