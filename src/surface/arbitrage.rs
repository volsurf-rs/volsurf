//! Surface-level arbitrage diagnostics.
//!
//! Extends the per-smile butterfly checks with cross-tenor calendar spread
//! checks: total variance must be non-decreasing in time at every strike.

use crate::smile::ArbitrageReport;
use serde::{Deserialize, Serialize};

/// Comprehensive diagnostics for a volatility surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceDiagnostics {
    /// Per-tenor butterfly arbitrage reports.
    pub smile_reports: Vec<ArbitrageReport>,
    /// Calendar spread violations (variance decreasing in time).
    pub calendar_violations: Vec<CalendarViolation>,
    /// Whether the entire surface is arbitrage-free.
    pub is_free: bool,
}

/// A calendar spread arbitrage violation at a specific (tenor, strike) point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarViolation {
    /// Strike where the violation occurs.
    pub strike: f64,
    /// Shorter tenor T₁.
    pub tenor_short: f64,
    /// Longer tenor T₂ > T₁.
    pub tenor_long: f64,
    /// Variance at shorter tenor (should be smaller).
    pub variance_short: f64,
    /// Variance at longer tenor (should be larger).
    pub variance_long: f64,
}
