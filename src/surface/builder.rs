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
use crate::smile::SmileSection;
use crate::smile::SviSmile;
use crate::surface::piecewise::PiecewiseSurface;
use crate::validate::{validate_finite, validate_positive};

/// Builder for constructing volatility surfaces from market data.
///
/// Accumulates spot price, risk-free rate, and per-tenor (strikes, vols)
/// data, then calibrates SVI smiles and assembles a [`PiecewiseSurface`].
pub struct SurfaceBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    tenor_data: Vec<TenorData>,
}

/// Market data for a single tenor.
struct TenorData {
    expiry: f64,
    strikes: Vec<f64>,
    vols: Vec<f64>,
}

impl SurfaceBuilder {
    /// Create a new surface builder.
    pub fn new() -> Self {
        Self {
            spot: None,
            rate: None,
            tenor_data: Vec::new(),
        }
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
            if tenor.strikes.len() < 5 {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "at least 5 strikes required per tenor, got {} for tenor {}",
                    tenor.strikes.len(),
                    tenor.expiry
                ),
                });
            }

            let forward = conventions::forward_price(spot, rate, tenor.expiry);

            let market_vols: Vec<(f64, f64)> = tenor
                .strikes
                .iter()
                .zip(tenor.vols.iter())
                .map(|(&k, &v)| (k, v))
                .collect();

            let smile = SviSmile::calibrate(forward, tenor.expiry, &market_vols)?;

            tenor_smile_pairs.push((tenor.expiry, Box::new(smile) as Box<dyn SmileSection>));
        }

        // --- Sort by tenor ---
        tenor_smile_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // --- Assemble PiecewiseSurface ---
        let (tenors, smiles): (Vec<f64>, Vec<Box<dyn SmileSection>>) =
            tenor_smile_pairs.into_iter().unzip();

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

    // --- Default trait ---

    #[test]
    fn default_is_same_as_new() {
        let builder = SurfaceBuilder::default();
        // Just verify it doesn't panic
        let result = builder.build();
        assert!(result.is_err()); // no spot/rate/tenors
    }
}
