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

    /// Merge two reports, combining all violations.
    ///
    /// The merged report is arbitrage-free only if both source reports are free.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::smile::{ArbitrageReport, ButterflyViolation};
    ///
    /// let clean = ArbitrageReport::clean();
    /// let violated = ArbitrageReport {
    ///     is_free: false,
    ///     butterfly_violations: vec![ButterflyViolation {
    ///         strike: 80.0, density: -0.001, magnitude: 0.001,
    ///     }],
    /// };
    /// let merged = clean.merge(&violated);
    /// assert!(!merged.is_free);
    /// assert_eq!(merged.butterfly_violations.len(), 1);
    /// ```
    pub fn merge(&self, other: &ArbitrageReport) -> ArbitrageReport {
        let mut violations = self.butterfly_violations.clone();
        violations.extend(other.butterfly_violations.iter().cloned());
        ArbitrageReport {
            is_free: self.is_free && other.is_free,
            butterfly_violations: violations,
        }
    }

    /// Return the worst (largest magnitude) butterfly violation, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::smile::{ArbitrageReport, ButterflyViolation};
    ///
    /// let report = ArbitrageReport {
    ///     is_free: false,
    ///     butterfly_violations: vec![
    ///         ButterflyViolation { strike: 80.0, density: -0.001, magnitude: 0.001 },
    ///         ButterflyViolation { strike: 90.0, density: -0.005, magnitude: 0.005 },
    ///     ],
    /// };
    /// let worst = report.worst_violation().unwrap();
    /// assert!((worst.magnitude - 0.005).abs() < 1e-12);
    /// ```
    pub fn worst_violation(&self) -> Option<&ButterflyViolation> {
        self.butterfly_violations
            .iter()
            .max_by(|a, b| {
                a.magnitude
                    .partial_cmp(&b.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::SmileSection;

    fn make_violation(strike: f64, density: f64) -> ButterflyViolation {
        ButterflyViolation {
            strike,
            density,
            magnitude: density.abs(),
        }
    }

    fn violated_report() -> ArbitrageReport {
        ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![
                make_violation(80.0, -0.001),
                make_violation(85.0, -0.005),
            ],
        }
    }

    // ========== merge() ==========

    #[test]
    fn merge_two_clean_reports() {
        let a = ArbitrageReport::clean();
        let b = ArbitrageReport::clean();
        let merged = a.merge(&b);
        assert!(merged.is_free);
        assert!(merged.butterfly_violations.is_empty());
    }

    #[test]
    fn merge_clean_and_violated() {
        let clean = ArbitrageReport::clean();
        let bad = violated_report();
        let merged = clean.merge(&bad);
        assert!(!merged.is_free);
        assert_eq!(merged.butterfly_violations.len(), 2);
    }

    #[test]
    fn merge_violated_and_clean() {
        let bad = violated_report();
        let clean = ArbitrageReport::clean();
        let merged = bad.merge(&clean);
        assert!(!merged.is_free);
        assert_eq!(merged.butterfly_violations.len(), 2);
    }

    #[test]
    fn merge_two_violated_combines_violations() {
        let a = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![make_violation(80.0, -0.001)],
        };
        let b = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![make_violation(90.0, -0.003), make_violation(95.0, -0.002)],
        };
        let merged = a.merge(&b);
        assert!(!merged.is_free);
        assert_eq!(merged.butterfly_violations.len(), 3);
    }

    #[test]
    fn merge_preserves_violation_data() {
        let a = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![make_violation(80.0, -0.007)],
        };
        let b = ArbitrageReport::clean();
        let merged = a.merge(&b);
        assert_eq!(merged.butterfly_violations.len(), 1);
        let v = &merged.butterfly_violations[0];
        assert_eq!(v.strike, 80.0);
        assert_eq!(v.density, -0.007);
        assert_eq!(v.magnitude, 0.007);
    }

    // ========== worst_violation() ==========

    #[test]
    fn worst_violation_clean_report_returns_none() {
        let clean = ArbitrageReport::clean();
        assert!(clean.worst_violation().is_none());
    }

    #[test]
    fn worst_violation_single_violation() {
        let report = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![make_violation(80.0, -0.003)],
        };
        let worst = report.worst_violation().unwrap();
        assert_eq!(worst.strike, 80.0);
        assert_eq!(worst.magnitude, 0.003);
    }

    #[test]
    fn worst_violation_picks_largest_magnitude() {
        let report = ArbitrageReport {
            is_free: false,
            butterfly_violations: vec![
                make_violation(80.0, -0.001),
                make_violation(85.0, -0.010),
                make_violation(90.0, -0.005),
            ],
        };
        let worst = report.worst_violation().unwrap();
        assert_eq!(worst.strike, 85.0);
        assert_eq!(worst.magnitude, 0.010);
    }

    // ========== SABR butterfly detection ==========

    #[test]
    fn sabr_extreme_nu_detects_violations() {
        // Large nu produces wild smile curvature that creates negative density.
        use crate::smile::SabrSmile;
        let sabr = SabrSmile::new(100.0, 1.0, 0.3, 0.5, -0.5, 2.0).unwrap();
        let report = sabr.is_arbitrage_free().unwrap();
        assert!(!report.is_free, "extreme nu should produce butterfly violations");
        assert!(!report.butterfly_violations.is_empty());
        let worst = report.worst_violation().unwrap();
        assert!(worst.magnitude > 0.0);
    }

    #[test]
    fn sabr_conservative_params_clean() {
        use crate::smile::SabrSmile;
        let sabr = SabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.3).unwrap();
        let report = sabr.is_arbitrage_free().unwrap();
        assert!(report.is_free, "conservative SABR should be arb-free");
        assert!(report.worst_violation().is_none());
    }

    // ========== SSVI butterfly detection ==========

    #[test]
    fn ssvi_extreme_params_detects_violations() {
        // eta*(1+|rho|) = 3*(1+0.95) = 5.85 >> 2 => violations expected.
        use crate::surface::SsviSlice;
        let slice = SsviSlice::new(100.0, 1.0, -0.95, 3.0, 0.5, 0.16).unwrap();
        let report = slice.is_arbitrage_free().unwrap();
        assert!(!report.is_free, "extreme SSVI params should detect violations");
        let worst = report.worst_violation().unwrap();
        assert!(worst.magnitude > 0.0);
    }

    #[test]
    fn ssvi_conservative_params_clean() {
        // eta*(1+|rho|) = 0.5*(1+0.3) = 0.65 < 2 => clean.
        use crate::surface::SsviSlice;
        let slice = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap();
        let report = slice.is_arbitrage_free().unwrap();
        assert!(report.is_free, "conservative SSVI should be arb-free");
        assert!(report.worst_violation().is_none());
    }
}
