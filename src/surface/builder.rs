//! Ergonomic builder API for volatility surface construction.
//!
//! ```
//! use volsurf::surface::{SurfaceBuilder, VolSurface};
//!
//! let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
//! let vols = vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28];
//!
//! let surface = SurfaceBuilder::new()
//!     .spot(100.0)
//!     .rate(0.05)
//!     .add_tenor(0.25, &strikes, &vols)
//!     .add_tenor(1.00, &strikes, &vols)
//!     .build()
//!     .unwrap();
//!
//! let vol = surface.black_vol(0.5, 100.0).unwrap();
//! assert!(vol.0 > 0.0);
//! ```

use crate::conventions;
use crate::error::VolSurfError;
use crate::smile::{SmileSection, SplineSmile, SviSmile};
use crate::surface::piecewise::PiecewiseSurface;
use crate::validate::{validate_finite, validate_positive};

/// Smile model to use when calibrating each tenor.
///
/// Different models have different trade-offs:
/// - [`Svi`](SmileModel::Svi) fits a parametric 5-parameter curve (minimum 5 strikes)
/// - [`CubicSpline`](SmileModel::CubicSpline) interpolates variance directly (minimum 3 strikes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmileModel {
    /// SVI parametric model (Gatheral 2004). Requires ≥ 5 strikes per tenor.
    #[default]
    Svi,
    /// Cubic spline on total variance. Requires ≥ 3 strikes per tenor.
    CubicSpline,
}

/// Builder for constructing volatility surfaces from market data.
///
/// Accumulates spot price, risk-free rate, and per-tenor (strikes, vols)
/// data, then calibrates smiles and assembles a [`PiecewiseSurface`].
pub struct SurfaceBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    model: SmileModel,
    tenor_data: Vec<TenorData>,
}

/// Market data for a single tenor.
struct TenorData {
    expiry: f64,
    strikes: Vec<f64>,
    vols: Vec<f64>,
}

impl SurfaceBuilder {
    /// Create a new surface builder with default settings (SVI model).
    pub fn new() -> Self {
        Self {
            spot: None,
            rate: None,
            model: SmileModel::default(),
            tenor_data: Vec::new(),
        }
    }

    /// Set the smile model used for per-tenor calibration.
    ///
    /// Default is [`SmileModel::Svi`].
    pub fn model(mut self, model: SmileModel) -> Self {
        self.model = model;
        self
    }

    /// Set the spot price.
    pub fn spot(mut self, spot: f64) -> Self {
        self.spot = Some(spot);
        self
    }

    /// Set the risk-free rate.
    pub fn rate(mut self, rate: f64) -> Self {
        self.rate = Some(rate);
        self
    }

    /// Add market data for a tenor.
    ///
    /// `strikes` and `vols` must have the same length.
    pub fn add_tenor(mut self, expiry: f64, strikes: &[f64], vols: &[f64]) -> Self {
        self.tenor_data.push(TenorData {
            expiry,
            strikes: strikes.to_vec(),
            vols: vols.to_vec(),
        });
        self
    }

    /// Build the volatility surface.
    ///
    /// Computes forward prices, calibrates an SVI smile per tenor, sorts
    /// by expiry, and assembles a [`PiecewiseSurface`].
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if required fields are missing
    /// or tenor data is invalid. Returns [`VolSurfError::CalibrationError`]
    /// if SVI calibration fails for any tenor.
    pub fn build(self) -> crate::error::Result<PiecewiseSurface> {
        #[cfg(feature = "logging")]
        tracing::debug!(
            n_tenors = self.tenor_data.len(),
            model = ?self.model,
            "surface build started"
        );

        // --- Validate required fields ---
        let spot = self.spot.ok_or_else(|| {
            VolSurfError::InvalidInput {
                message: "spot price is required".into(),
            }
        })?;
        let rate = self.rate.ok_or_else(|| {
            VolSurfError::InvalidInput {
                message: "risk-free rate is required".into(),
            }
        })?;

        validate_positive(spot, "spot")?;
        validate_finite(rate, "rate")?;
        if self.tenor_data.is_empty() {
            return Err(VolSurfError::InvalidInput {
                message: "at least one tenor is required".into(),
            });
        }

        // --- Per-tenor validation, forward calculation, and calibration ---
        let mut tenor_smile_pairs: Vec<(f64, Box<dyn SmileSection>)> =
            Vec::with_capacity(self.tenor_data.len());

        let min_strikes = match self.model {
            SmileModel::Svi => 5,
            SmileModel::CubicSpline => 3,
        };

        for tenor in &self.tenor_data {
            if tenor.expiry <= 0.0 || !tenor.expiry.is_finite() {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "expiry must be positive and finite, got {}",
                        tenor.expiry
                    ),
                });
            }
            if tenor.strikes.len() != tenor.vols.len() {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "strikes ({}) and vols ({}) must have the same length for tenor {}",
                        tenor.strikes.len(),
                        tenor.vols.len(),
                        tenor.expiry
                    ),
                });
            }
            if tenor.strikes.len() < min_strikes {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "at least {min_strikes} strikes required per tenor (model: {:?}), got {} for tenor {}",
                        self.model,
                        tenor.strikes.len(),
                        tenor.expiry
                    ),
                });
            }

            let forward = conventions::forward_price(spot, rate, tenor.expiry);

            let smile: Box<dyn SmileSection> = match self.model {
                SmileModel::Svi => {
                    let market_vols: Vec<(f64, f64)> = tenor
                        .strikes
                        .iter()
                        .zip(tenor.vols.iter())
                        .map(|(&k, &v)| (k, v))
                        .collect();
                    let svi = SviSmile::calibrate(forward, tenor.expiry, &market_vols)?;
                    Box::new(svi)
                }
                SmileModel::CubicSpline => {
                    // Sort strikes and convert vols to total variances
                    let mut pairs: Vec<(f64, f64)> = tenor
                        .strikes
                        .iter()
                        .zip(tenor.vols.iter())
                        .map(|(&k, &v)| (k, v * v * tenor.expiry))
                        .collect();
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let (strikes, variances): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
                    let spline = SplineSmile::new(forward, tenor.expiry, strikes, variances)?;
                    Box::new(spline)
                }
            };

            tenor_smile_pairs.push((tenor.expiry, smile));
        }

        // --- Sort by tenor ---
        tenor_smile_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // --- Assemble PiecewiseSurface ---
        let (tenors, smiles): (Vec<f64>, Vec<Box<dyn SmileSection>>) =
            tenor_smile_pairs.into_iter().unzip();

        #[cfg(feature = "logging")]
        tracing::debug!(n_tenors = tenors.len(), "surface build complete");

        PiecewiseSurface::new(tenors, smiles)
    }
}

impl Default for SurfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::VolSurface;
    use approx::assert_abs_diff_eq;

    /// Market data: symmetric U-shaped smile with 7 strikes.
    fn sample_strikes() -> Vec<f64> {
        vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    }

    fn sample_vols() -> Vec<f64> {
        vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28]
    }

    // --- Happy path ---

    #[test]
    fn build_single_tenor_surface() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let vol = surface.black_vol(0.25, 100.0).unwrap();
        assert!(vol.0 > 0.0, "ATM vol should be positive");
        assert!(vol.0 < 1.0, "ATM vol should be reasonable");
    }

    #[test]
    fn build_multi_tenor_surface() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        // Query at stored tenor
        let vol_3m = surface.black_vol(0.25, 100.0).unwrap();
        assert!(vol_3m.0 > 0.0);

        // Query between tenors
        let vol_6m = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol_6m.0 > 0.0);

        // Query at stored tenor
        let vol_1y = surface.black_vol(1.0, 100.0).unwrap();
        assert!(vol_1y.0 > 0.0);
    }

    #[test]
    fn build_with_unsorted_tenors_sorts_them() {
        // Add 1Y before 3M — builder should sort
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let vol = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol.0 > 0.0);
    }

    #[test]
    fn resulting_surface_answers_queries() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        // Multiple strikes at multiple tenors
        for t in [0.25, 0.5, 1.0] {
            for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
                let vol = surface.black_vol(t, k).unwrap();
                assert!(vol.0 > 0.0, "vol({t}, {k}) should be positive");
                assert!(vol.0 < 2.0, "vol({t}, {k}) should be reasonable");
            }
        }
    }

    #[test]
    fn vol_and_variance_consistent_after_build() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let t = 0.5;
        let k = 100.0;
        let vol = surface.black_vol(t, k).unwrap();
        let var = surface.black_variance(t, k).unwrap();
        assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-12);
    }

    #[test]
    fn build_with_negative_rate() {
        // Negative rates (EUR, JPY) should work
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(-0.01)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build();
        assert!(surface.is_ok());
    }

    // --- Validation errors ---

    #[test]
    fn missing_spot_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn missing_rate_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn no_tenors_returns_invalid_input() {
        let result = SurfaceBuilder::new().spot(100.0).rate(0.05).build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn fewer_than_5_strikes_returns_invalid_input() {
        let strikes = vec![90.0, 95.0, 100.0, 105.0]; // only 4
        let vols = vec![0.22, 0.20, 0.20, 0.22];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn mismatched_strikes_vols_returns_invalid_input() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let vols = vec![0.20, 0.20, 0.20]; // wrong length
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn zero_spot_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(0.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn negative_spot_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(-100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn zero_expiry_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.0, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn nan_rate_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(f64::NAN)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- SmileModel selector ---

    #[test]
    fn default_model_is_svi() {
        let builder = SurfaceBuilder::new();
        assert_eq!(builder.model, SmileModel::Svi);
    }

    #[test]
    fn build_with_cubic_spline_model() {
        // CubicSpline only needs 3 strikes
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.24, 0.20, 0.24];
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .add_tenor(1.0, &strikes, &vols)
            .build()
            .unwrap();

        let vol = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol.0 > 0.0);
        assert!(vol.0 < 1.0);
    }

    #[test]
    fn cubic_spline_with_3_strikes_succeeds() {
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.22, 0.20, 0.22];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn cubic_spline_with_2_strikes_fails() {
        let strikes = vec![90.0, 110.0];
        let vols = vec![0.22, 0.22];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Gap #6: CubicSpline with NaN and duplicate strikes ---

    #[test]
    fn cubic_spline_with_nan_strike_returns_error() {
        let strikes = vec![90.0, f64::NAN, 110.0];
        let vols = vec![0.22, 0.20, 0.22];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        // SplineSmile::new rejects NaN strikes
        assert!(result.is_err(), "NaN strike should cause build to fail");
    }

    #[test]
    fn cubic_spline_with_duplicate_strikes_returns_error() {
        let strikes = vec![90.0, 100.0, 100.0, 110.0];
        let vols = vec![0.22, 0.20, 0.20, 0.22];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        // SplineSmile::new rejects non-strictly-increasing strikes
        assert!(result.is_err(), "duplicate strikes should cause build to fail");
    }

    #[test]
    fn cubic_spline_with_unsorted_strikes_succeeds() {
        // Builder sorts strikes internally for CubicSpline path
        let strikes = vec![110.0, 90.0, 100.0];
        let vols = vec![0.24, 0.24, 0.20];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(result.is_ok(), "unsorted strikes should be sorted by builder");
    }

    // --- Default trait ---

    #[test]
    fn default_is_same_as_new() {
        let builder = SurfaceBuilder::default();
        // Just verify it doesn't panic
        let result = builder.build();
        assert!(result.is_err()); // no spot/rate/tenors
    }
}
