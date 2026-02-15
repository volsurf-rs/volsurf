//! Ergonomic builder API for volatility surface construction.
//!
//! ```ignore
//! let surface = SurfaceBuilder::new()
//!     .spot(100.0)
//!     .rate(0.05)
//!     .add_tenor(0.25, &strikes_3m, &vols_3m)
//!     .add_tenor(0.50, &strikes_6m, &vols_6m)
//!     .build()?;
//! ```

use crate::error::VolSurfError;
use crate::surface::piecewise::PiecewiseSurface;

/// Builder for constructing volatility surfaces from market data.
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
    /// Calibrates a smile section per tenor and assembles the piecewise surface.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if required fields are missing,
    /// [`VolSurfError::CalibrationError`] if smile calibration fails.
    pub fn build(self) -> crate::error::Result<PiecewiseSurface> {
        let _ = (self.spot, self.rate, self.tenor_data);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl Default for SurfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
