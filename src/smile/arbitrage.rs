//! Arbitrage detection for volatility smiles.
//!
//! Butterfly arbitrage occurs when the risk-neutral density implied by option
//! prices becomes negative, violating the no-arbitrage condition.
//!
//! # References
//! - Breeden, D.T. & Litzenberger, R.H. "Prices of State-Contingent Claims
//!   Implicit in Option Prices" (1978)

use serde::{Deserialize, Serialize};

use crate::error;
use crate::smile::{ArbitrageScanConfig, BUTTERFLY_G_TOL, DENSITY_NEG_TOL};

/// Report on arbitrage-freeness of a smile at a specific expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageReport {
    /// Expiry (time to maturity) of the smile this report covers.
    pub expiry: f64,
    /// Butterfly spread violations (negative density regions).
    pub butterfly_violations: Vec<ButterflyViolation>,
}

impl ArbitrageReport {
    /// Create a report indicating no arbitrage was found.
    pub fn clean(expiry: f64) -> Self {
        Self {
            expiry,
            butterfly_violations: Vec::new(),
        }
    }

    /// Whether this smile is free of detected butterfly arbitrage.
    pub fn is_free(&self) -> bool {
        self.butterfly_violations.is_empty()
    }

    /// Return the worst (largest magnitude) butterfly violation, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::smile::{ArbitrageReport, ButterflyViolation};
    ///
    /// let report = ArbitrageReport {
    ///     expiry: 1.0,
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
            .max_by(|a, b| a.magnitude.total_cmp(&b.magnitude))
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

/// Gatheral's g-function for a total-variance smile and its derivatives.
pub(crate) fn gatheral_g(k: f64, w: f64, wp: f64, wpp: f64) -> f64 {
    if w <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let term1 = 1.0 - k * wp / (2.0 * w);
    term1 * term1 - wp * wp / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
}

/// Convert Gatheral's g-function value to risk-neutral density.
pub(crate) fn density_from_g(strike: f64, k: f64, w: f64, g: f64) -> f64 {
    let sqrt_w = w.sqrt();
    let d2 = -k / sqrt_w - sqrt_w / 2.0;
    let n_d2 = (-d2 * d2 / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    g * n_d2 / (strike * sqrt_w)
}

/// Scan a log-moneyness grid for negative risk-neutral density.
pub(crate) fn scan_density<F>(
    expiry: f64,
    forward: f64,
    config: &ArbitrageScanConfig,
    density: F,
) -> error::Result<ArbitrageReport>
where
    F: Fn(f64) -> error::Result<f64>,
{
    config.validate()?;
    let mut violations = Vec::new();
    for i in 0..config.n_points {
        let k =
            config.k_min + (config.k_max - config.k_min) * i as f64 / (config.n_points - 1) as f64;
        let strike = forward * k.exp();
        let d = match density(strike) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if d < -DENSITY_NEG_TOL {
            violations.push(ButterflyViolation {
                strike,
                density: d,
                magnitude: d.abs(),
            });
        }
    }
    Ok(ArbitrageReport {
        expiry,
        butterfly_violations: violations,
    })
}

/// Scan a log-moneyness grid for negative Gatheral g-function values.
pub(crate) fn scan_g<G, D>(
    expiry: f64,
    forward: f64,
    config: &ArbitrageScanConfig,
    g_at_k: G,
    density: D,
) -> error::Result<ArbitrageReport>
where
    G: Fn(f64) -> f64,
    D: Fn(f64) -> error::Result<f64>,
{
    config.validate()?;
    let mut violations = Vec::new();
    for i in 0..config.n_points {
        let k =
            config.k_min + (config.k_max - config.k_min) * i as f64 / (config.n_points - 1) as f64;
        if g_at_k(k) < -BUTTERFLY_G_TOL {
            let strike = forward * k.exp();
            let d = match density(strike) {
                Ok(d) => d,
                Err(_) => continue,
            };
            violations.push(ButterflyViolation {
                strike,
                density: d,
                magnitude: d.abs(),
            });
        }
    }
    Ok(ArbitrageReport {
        expiry,
        butterfly_violations: violations,
    })
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

    #[test]
    fn clean_report_is_free() {
        let report = ArbitrageReport::clean(1.0);
        assert!(report.is_free());
        assert!(report.butterfly_violations.is_empty());
        assert_eq!(report.expiry, 1.0);
    }

    #[test]
    fn is_free_computed_from_violations() {
        let clean = ArbitrageReport {
            expiry: 1.0,
            butterfly_violations: vec![],
        };
        assert!(clean.is_free());

        let violated = ArbitrageReport {
            expiry: 1.0,
            butterfly_violations: vec![make_violation(80.0, -0.001)],
        };
        assert!(!violated.is_free());
    }

    #[test]
    fn worst_violation_clean_report_returns_none() {
        let clean = ArbitrageReport::clean(1.0);
        assert!(clean.worst_violation().is_none());
    }

    #[test]
    fn worst_violation_single_violation() {
        let report = ArbitrageReport {
            expiry: 1.0,
            butterfly_violations: vec![make_violation(80.0, -0.003)],
        };
        let worst = report.worst_violation().unwrap();
        assert_eq!(worst.strike, 80.0);
        assert_eq!(worst.magnitude, 0.003);
    }

    #[test]
    fn worst_violation_picks_largest_magnitude() {
        let report = ArbitrageReport {
            expiry: 1.0,
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

    #[test]
    fn worst_violation_tied_magnitudes() {
        let report = ArbitrageReport {
            expiry: 1.0,
            butterfly_violations: vec![
                ButterflyViolation {
                    strike: 90.0,
                    density: -0.01,
                    magnitude: 0.005,
                },
                ButterflyViolation {
                    strike: 110.0,
                    density: -0.02,
                    magnitude: 0.005,
                },
            ],
        };
        let worst = report.worst_violation().unwrap();
        assert!((worst.magnitude - 0.005).abs() < 1e-15);
        assert_eq!(worst.strike, 110.0);
    }

    #[test]
    fn sabr_extreme_nu_detects_violations() {
        use crate::smile::SabrSmile;
        let sabr = SabrSmile::new(100.0, 1.0, 0.3, 0.5, -0.5, 2.0).unwrap();
        let report = sabr.is_arbitrage_free().unwrap();
        assert!(
            !report.is_free(),
            "extreme nu should produce butterfly violations"
        );
        assert!(!report.butterfly_violations.is_empty());
        assert!(report.worst_violation().unwrap().magnitude > 0.0);
    }

    #[test]
    fn sabr_conservative_params_clean() {
        use crate::smile::SabrSmile;
        let sabr = SabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.3).unwrap();
        let report = sabr.is_arbitrage_free().unwrap();
        assert!(report.is_free(), "conservative SABR should be arb-free");
        assert!(report.worst_violation().is_none());
    }

    #[test]
    fn ssvi_extreme_params_detects_violations() {
        use crate::surface::SsviSlice;
        let slice = SsviSlice::new(100.0, 1.0, -0.95, 3.0, 0.5, 0.16).unwrap();
        let report = slice.is_arbitrage_free().unwrap();
        assert!(
            !report.is_free(),
            "extreme SSVI params should detect violations"
        );
        assert!(report.worst_violation().unwrap().magnitude > 0.0);
    }

    #[test]
    fn ssvi_conservative_params_clean() {
        use crate::surface::SsviSlice;
        let slice = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap();
        let report = slice.is_arbitrage_free().unwrap();
        assert!(report.is_free(), "conservative SSVI should be arb-free");
        assert!(report.worst_violation().is_none());
    }

    // ========== ArbitrageScanConfig ==========

    #[test]
    fn svi_default_config_matches_hardcoded() {
        use crate::smile::{ArbitrageScanConfig, SviSmile};
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.5, 0.0, 0.3).unwrap();
        let default_report = svi.is_arbitrage_free().unwrap();
        let config_report = svi
            .is_arbitrage_free_with(&ArbitrageScanConfig::svi_default())
            .unwrap();
        assert_eq!(default_report.is_free(), config_report.is_free());
        assert_eq!(
            default_report.butterfly_violations.len(),
            config_report.butterfly_violations.len()
        );
    }

    #[test]
    fn sabr_default_config_matches_hardcoded() {
        use crate::smile::{ArbitrageScanConfig, SabrSmile};
        let sabr = SabrSmile::new(100.0, 1.0, 0.3, 0.5, -0.5, 2.0).unwrap();
        let default_report = sabr.is_arbitrage_free().unwrap();
        let config_report = sabr
            .is_arbitrage_free_with(&ArbitrageScanConfig::sabr_default())
            .unwrap();
        assert_eq!(default_report.is_free(), config_report.is_free());
        assert_eq!(
            default_report.butterfly_violations.len(),
            config_report.butterfly_violations.len()
        );
    }

    #[test]
    fn higher_n_points_finds_more_violations() {
        use crate::smile::{ArbitrageScanConfig, SviSmile};
        // Params that produce wing violations
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.4, -0.9, 0.1, 0.2).unwrap();
        let coarse = svi
            .is_arbitrage_free_with(&ArbitrageScanConfig {
                n_points: 20,
                k_min: -3.0,
                k_max: 3.0,
            })
            .unwrap();
        let fine = svi
            .is_arbitrage_free_with(&ArbitrageScanConfig {
                n_points: 500,
                k_min: -3.0,
                k_max: 3.0,
            })
            .unwrap();
        assert!(
            fine.butterfly_violations.len() >= coarse.butterfly_violations.len(),
            "finer grid should find at least as many violations"
        );
    }

    #[test]
    fn narrow_range_misses_wing_violations() {
        use crate::smile::{ArbitrageScanConfig, SabrSmile};
        let sabr = SabrSmile::new(100.0, 1.0, 0.3, 0.5, -0.5, 2.0).unwrap();
        let wide = sabr
            .is_arbitrage_free_with(&ArbitrageScanConfig::sabr_default())
            .unwrap();
        let narrow = sabr
            .is_arbitrage_free_with(&ArbitrageScanConfig {
                n_points: 200,
                k_min: -0.5,
                k_max: 0.5,
            })
            .unwrap();
        assert!(
            narrow.butterfly_violations.len() <= wide.butterfly_violations.len(),
            "narrow range should find fewer violations"
        );
    }

    #[test]
    fn config_rejects_n_points_below_two() {
        use crate::smile::{ArbitrageScanConfig, SviSmile};
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.5, 0.0, 0.3).unwrap();
        let config = ArbitrageScanConfig {
            n_points: 1,
            k_min: -3.0,
            k_max: 3.0,
        };
        assert!(svi.is_arbitrage_free_with(&config).is_err());
    }

    #[test]
    fn config_rejects_inverted_range() {
        use crate::smile::{ArbitrageScanConfig, SviSmile};
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.5, 0.0, 0.3).unwrap();
        let config = ArbitrageScanConfig {
            n_points: 200,
            k_min: 3.0,
            k_max: -3.0,
        };
        assert!(svi.is_arbitrage_free_with(&config).is_err());
    }

    #[test]
    fn config_rejects_nan() {
        use crate::smile::{ArbitrageScanConfig, SviSmile};
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.5, 0.0, 0.3).unwrap();
        let config = ArbitrageScanConfig {
            n_points: 200,
            k_min: f64::NAN,
            k_max: 3.0,
        };
        assert!(svi.is_arbitrage_free_with(&config).is_err());
    }
}
