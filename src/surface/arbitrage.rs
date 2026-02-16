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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};

    #[test]
    fn empty_diagnostics_is_free() {
        let diag = SurfaceDiagnostics {
            smile_reports: vec![],
            calendar_violations: vec![],
            is_free: true,
        };
        assert!(diag.is_free);
        assert!(diag.smile_reports.is_empty());
        assert!(diag.calendar_violations.is_empty());
    }

    #[test]
    fn diagnostics_with_butterfly_violations_not_free() {
        let report = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![ButterflyViolation {
                strike: 90.0,
                density: -0.001,
                magnitude: 0.001,
            }],
        };
        let diag = SurfaceDiagnostics {
            smile_reports: vec![report],
            calendar_violations: vec![],
            is_free: false,
        };
        assert!(!diag.is_free);
        assert_eq!(diag.smile_reports.len(), 1);
        assert!(!diag.smile_reports[0].is_free);
    }

    #[test]
    fn diagnostics_with_calendar_violations_not_free() {
        let violation = CalendarViolation {
            strike: 100.0,
            tenor_short: 0.25,
            tenor_long: 0.50,
            variance_short: 0.06,
            variance_long: 0.05,
        };
        let clean_report = ArbitrageReport {
            is_free: true,
            butterfly_violations: vec![],
        };
        let diag = SurfaceDiagnostics {
            smile_reports: vec![clean_report],
            calendar_violations: vec![violation],
            is_free: false,
        };
        assert!(!diag.is_free);
        assert_eq!(diag.calendar_violations.len(), 1);
        assert_eq!(diag.calendar_violations[0].strike, 100.0);
        assert!(diag.calendar_violations[0].variance_short > diag.calendar_violations[0].variance_long);
    }

    #[test]
    fn diagnostics_mixed_violations_not_free() {
        let butterfly_report = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![ButterflyViolation {
                strike: 85.0,
                density: -0.002,
                magnitude: 0.002,
            }],
        };
        let cal_violation = CalendarViolation {
            strike: 110.0,
            tenor_short: 0.50,
            tenor_long: 1.00,
            variance_short: 0.10,
            variance_long: 0.08,
        };
        let diag = SurfaceDiagnostics {
            smile_reports: vec![butterfly_report],
            calendar_violations: vec![cal_violation],
            is_free: false,
        };
        assert!(!diag.is_free);
        assert!(!diag.smile_reports.is_empty());
        assert!(!diag.calendar_violations.is_empty());
    }

    #[test]
    fn surface_diagnostics_serde_round_trip() {
        let diag = SurfaceDiagnostics {
            smile_reports: vec![
                ArbitrageReport {
                    is_free: true,
                    butterfly_violations: vec![],
                },
                ArbitrageReport {
                    is_free: false,
                    butterfly_violations: vec![ButterflyViolation {
                        strike: 95.0,
                        density: -0.0005,
                        magnitude: 0.0005,
                    }],
                },
            ],
            calendar_violations: vec![CalendarViolation {
                strike: 105.0,
                tenor_short: 0.25,
                tenor_long: 0.50,
                variance_short: 0.07,
                variance_long: 0.065,
            }],
            is_free: false,
        };

        let json = serde_json::to_string(&diag).unwrap();
        let roundtrip: SurfaceDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(roundtrip.is_free, diag.is_free);
        assert_eq!(roundtrip.smile_reports.len(), diag.smile_reports.len());
        assert_eq!(roundtrip.calendar_violations.len(), diag.calendar_violations.len());
        assert_eq!(roundtrip.calendar_violations[0].strike, 105.0);
        assert_eq!(roundtrip.smile_reports[1].butterfly_violations[0].strike, 95.0);
    }

    #[test]
    fn calendar_violation_serde_round_trip() {
        let violation = CalendarViolation {
            strike: 100.0,
            tenor_short: 0.25,
            tenor_long: 0.50,
            variance_short: 0.06,
            variance_long: 0.05,
        };
        let json = serde_json::to_string(&violation).unwrap();
        let roundtrip: CalendarViolation = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.strike, violation.strike);
        assert_eq!(roundtrip.tenor_short, violation.tenor_short);
        assert_eq!(roundtrip.tenor_long, violation.tenor_long);
        assert_eq!(roundtrip.variance_short, violation.variance_short);
        assert_eq!(roundtrip.variance_long, violation.variance_long);
    }
}
