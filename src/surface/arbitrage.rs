//! Surface-level arbitrage diagnostics.
//!
//! Extends the per-smile butterfly checks with cross-tenor calendar spread
//! checks: total variance must be non-decreasing in time at every strike.

use crate::error;
use crate::smile::ArbitrageReport;
use crate::surface::interp::strike_grid;
use crate::surface::{CALENDAR_ARB_TOL, CALENDAR_CHECK_GRID_SIZE};
use serde::{Deserialize, Serialize};

/// Comprehensive diagnostics for a volatility surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceDiagnostics {
    /// Per-tenor butterfly arbitrage reports.
    pub smile_reports: Vec<ArbitrageReport>,
    /// Calendar spread violations (variance decreasing in time).
    pub calendar_violations: Vec<CalendarViolation>,
}

impl SurfaceDiagnostics {
    /// Whether the entire surface is free of detected arbitrage.
    pub fn is_free(&self) -> bool {
        self.smile_reports.iter().all(|r| r.is_free()) && self.calendar_violations.is_empty()
    }
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

/// Run common per-smile and adjacent-tenor surface diagnostics.
pub(crate) fn surface_diagnostics<R, V>(
    tenors: &[f64],
    forwards: &[f64],
    report_at: R,
    variance_at: V,
) -> error::Result<SurfaceDiagnostics>
where
    R: Fn(usize) -> error::Result<ArbitrageReport>,
    V: Fn(usize, f64) -> error::Result<f64>,
{
    let smile_reports = (0..tenors.len())
        .map(report_at)
        .collect::<error::Result<Vec<_>>>()?;

    let mut calendar_violations = Vec::new();
    for i in 0..tenors.len().saturating_sub(1) {
        let forward = 0.5 * (forwards[i] + forwards[i + 1]);
        for strike in strike_grid(forward, CALENDAR_CHECK_GRID_SIZE) {
            let variance_short = variance_at(i, strike)?;
            let variance_long = variance_at(i + 1, strike)?;
            if variance_long < variance_short - CALENDAR_ARB_TOL {
                calendar_violations.push(CalendarViolation {
                    strike,
                    tenor_short: tenors[i],
                    tenor_long: tenors[i + 1],
                    variance_short,
                    variance_long,
                });
            }
        }
    }

    Ok(SurfaceDiagnostics {
        smile_reports,
        calendar_violations,
    })
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
        };
        assert!(diag.is_free());
        assert!(diag.smile_reports.is_empty());
        assert!(diag.calendar_violations.is_empty());
    }

    #[test]
    fn diagnostics_with_butterfly_violations_not_free() {
        let report = ArbitrageReport {
            expiry: 1.0,
            butterfly_violations: vec![ButterflyViolation {
                strike: 90.0,
                density: -0.001,
                magnitude: 0.001,
            }],
        };
        let diag = SurfaceDiagnostics {
            smile_reports: vec![report],
            calendar_violations: vec![],
        };
        assert!(!diag.is_free());
        assert_eq!(diag.smile_reports.len(), 1);
        assert!(!diag.smile_reports[0].is_free());
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
            expiry: 1.0,
            butterfly_violations: vec![],
        };
        let diag = SurfaceDiagnostics {
            smile_reports: vec![clean_report],
            calendar_violations: vec![violation],
        };
        assert!(!diag.is_free());
        assert_eq!(diag.calendar_violations.len(), 1);
        assert_eq!(diag.calendar_violations[0].strike, 100.0);
        assert!(
            diag.calendar_violations[0].variance_short > diag.calendar_violations[0].variance_long
        );
    }

    #[test]
    fn diagnostics_mixed_violations_not_free() {
        let butterfly_report = ArbitrageReport {
            expiry: 1.0,
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
        };
        assert!(!diag.is_free());
        assert!(!diag.smile_reports.is_empty());
        assert!(!diag.calendar_violations.is_empty());
    }

    #[test]
    fn surface_diagnostics_serde_round_trip() {
        let diag = SurfaceDiagnostics {
            smile_reports: vec![
                ArbitrageReport {
                    expiry: 0.5,
                    butterfly_violations: vec![],
                },
                ArbitrageReport {
                    expiry: 1.0,
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
        };

        let json = serde_json::to_string(&diag).unwrap();
        let roundtrip: SurfaceDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(roundtrip.is_free(), diag.is_free());
        assert_eq!(roundtrip.smile_reports.len(), diag.smile_reports.len());
        assert_eq!(
            roundtrip.calendar_violations.len(),
            diag.calendar_violations.len()
        );
        assert_eq!(roundtrip.calendar_violations[0].strike, 105.0);
        assert_eq!(
            roundtrip.smile_reports[1].butterfly_violations[0].strike,
            95.0
        );
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
