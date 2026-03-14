//! Calibration configuration types shared across smile models.

use serde::{Deserialize, Serialize};

/// Pre-calibration data filter for strike/vol cleaning.
///
/// Applied before any model-specific logic. Filters `(strike, vol)` pairs
/// by log-moneyness and minimum vol threshold.
///
/// # References
/// - Zeliade (2009) §3: restrict calibration to liquid strikes
/// - Corbetta et al. (2019): filter options below 2 ticks
/// - Ferhati (2020): volume concentrated in |k| < 0.1 across 23 equity indices
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct DataFilter {
    /// Exclude strikes where |ln(K/F)| exceeds this threshold.
    pub max_log_moneyness: Option<f64>,
    /// Exclude quotes with implied vol below this floor.
    pub min_vol: Option<f64>,
    /// Toggle the vol-cliff heuristic (>50% consecutive drop detection).
    /// `None` uses the model default: `true` for SVI, `false` for SABR.
    pub vol_cliff_filter: Option<bool>,
}

/// Weighting scheme for calibration objective function.
///
/// Controls how market points are weighted in the least-squares fit.
/// Each model has a literature-backed default that `ModelDefault` resolves to.
///
/// # References
/// - Zeliade (2009): vega weighting for SVI quasi-explicit decomposition
/// - Hagan et al. (2002): uniform weighting for ATM-anchored SABR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WeightingScheme {
    /// Model default: SVI → Vega, SABR → Uniform.
    #[default]
    ModelDefault,
    /// Vega weighting via n(d₁). Emphasizes liquid ATM region.
    Vega,
    /// Equal weight on all points.
    Uniform,
}

/// Filter `(strike, vol)` pairs by log-moneyness and minimum vol.
///
/// Unconditionally excludes points with non-finite vol (NaN/Inf) or
/// non-finite log-moneyness (NaN/Inf strike or forward). Does not apply
/// the vol-cliff heuristic — that is handled by model-specific calibration
/// code based on [`DataFilter::vol_cliff_filter`].
pub fn apply_filter(
    market_vols: &[(f64, f64)],
    forward: f64,
    filter: &DataFilter,
) -> Vec<(f64, f64)> {
    market_vols
        .iter()
        .copied()
        .filter(|&(strike, vol)| {
            if !vol.is_finite() {
                return false;
            }
            let lm = (strike / forward).ln();
            if !lm.is_finite() {
                return false;
            }
            if filter
                .max_log_moneyness
                .is_some_and(|max_k| lm.abs() > max_k)
            {
                return false;
            }
            if filter.min_vol.is_some_and(|min_v| vol < min_v) {
                return false;
            }
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_filter_passes_everything() {
        let data = vec![(90.0, 0.25), (100.0, 0.20), (110.0, 0.22)];
        let result = apply_filter(&data, 100.0, &DataFilter::default());
        assert_eq!(result, data);
    }

    #[test]
    fn max_log_moneyness_excludes_far_wings() {
        let data = vec![
            (50.0, 0.40),  // |k| = 0.69
            (90.0, 0.25),  // |k| = 0.105
            (100.0, 0.20), // |k| = 0.0
            (110.0, 0.22), // |k| = 0.095
            (200.0, 0.35), // |k| = 0.69
        ];
        let filter = DataFilter {
            max_log_moneyness: Some(0.5),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert_eq!(result, vec![(90.0, 0.25), (100.0, 0.20), (110.0, 0.22)]);
    }

    #[test]
    fn min_vol_excludes_cabinet_quotes() {
        let data = vec![
            (90.0, 0.25),
            (95.0, 0.005), // cabinet
            (100.0, 0.20),
            (150.0, 0.008), // cabinet
        ];
        let filter = DataFilter {
            min_vol: Some(0.01),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert_eq!(result, vec![(90.0, 0.25), (100.0, 0.20)]);
    }

    #[test]
    fn combined_filter() {
        let data = vec![
            (50.0, 0.40),   // far wing, passes min_vol but not moneyness
            (95.0, 0.005),  // near ATM but cabinet vol
            (100.0, 0.20),  // passes both
            (105.0, 0.22),  // passes both
            (200.0, 0.008), // far wing + cabinet
        ];
        let filter = DataFilter {
            max_log_moneyness: Some(0.2),
            min_vol: Some(0.01),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert_eq!(result, vec![(100.0, 0.20), (105.0, 0.22)]);
    }

    #[test]
    fn empty_input() {
        let result = apply_filter(&[], 100.0, &DataFilter::default());
        assert!(result.is_empty());
    }

    #[test]
    fn all_filtered_out() {
        let data = vec![(50.0, 0.40), (200.0, 0.35)];
        let filter = DataFilter {
            max_log_moneyness: Some(0.1),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert!(result.is_empty());
    }

    #[test]
    fn nan_strike_excluded() {
        let data = vec![(f64::NAN, 0.20), (100.0, 0.20)];
        let result = apply_filter(&data, 100.0, &DataFilter::default());
        assert_eq!(result, vec![(100.0, 0.20)]);
    }

    #[test]
    fn nan_vol_excluded_unconditionally() {
        let data = vec![(100.0, f64::NAN), (100.0, 0.20)];
        let result = apply_filter(&data, 100.0, &DataFilter::default());
        assert_eq!(result, vec![(100.0, 0.20)]);
    }

    #[test]
    fn inf_vol_excluded_unconditionally() {
        let data = vec![(100.0, f64::INFINITY), (100.0, 0.20)];
        let result = apply_filter(&data, 100.0, &DataFilter::default());
        assert_eq!(result, vec![(100.0, 0.20)]);
    }

    #[test]
    fn nan_vol_excluded_with_min_vol() {
        let data = vec![(100.0, f64::NAN), (100.0, 0.20)];
        let filter = DataFilter {
            min_vol: Some(0.01),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert_eq!(result, vec![(100.0, 0.20)]);
    }

    #[test]
    fn zero_forward_excludes_all() {
        let data = vec![(100.0, 0.20), (110.0, 0.22)];
        let filter = DataFilter {
            max_log_moneyness: Some(0.5),
            ..Default::default()
        };
        let result = apply_filter(&data, 0.0, &filter);
        assert!(result.is_empty());
    }

    #[test]
    fn vol_cliff_filter_not_applied() {
        let data = vec![(90.0, 0.25), (100.0, 0.02), (110.0, 0.22)];
        let filter = DataFilter {
            vol_cliff_filter: Some(true),
            ..Default::default()
        };
        let result = apply_filter(&data, 100.0, &filter);
        assert_eq!(
            result.len(),
            3,
            "apply_filter must not act on vol_cliff_filter"
        );
    }

    #[test]
    fn serde_round_trip_data_filter() {
        let filter = DataFilter {
            max_log_moneyness: Some(0.5),
            min_vol: Some(0.01),
            vol_cliff_filter: Some(false),
        };
        let json = serde_json::to_string(&filter).unwrap();
        let roundtrip: DataFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(filter, roundtrip);
    }

    #[test]
    fn serde_round_trip_weighting_scheme() {
        for scheme in [
            WeightingScheme::ModelDefault,
            WeightingScheme::Vega,
            WeightingScheme::Uniform,
        ] {
            let json = serde_json::to_string(&scheme).unwrap();
            let roundtrip: WeightingScheme = serde_json::from_str(&json).unwrap();
            assert_eq!(scheme, roundtrip);
        }
    }

    #[test]
    fn defaults() {
        let filter = DataFilter::default();
        assert_eq!(filter.max_log_moneyness, None);
        assert_eq!(filter.min_vol, None);
        assert_eq!(filter.vol_cliff_filter, None);
        assert_eq!(WeightingScheme::default(), WeightingScheme::ModelDefault);
    }

    #[test]
    fn preserves_input_order() {
        let data = vec![(110.0, 0.22), (90.0, 0.25), (100.0, 0.20)];
        let result = apply_filter(&data, 100.0, &DataFilter::default());
        assert_eq!(result, data);
    }
}
