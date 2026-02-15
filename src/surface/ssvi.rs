//! SSVI (Surface SVI) global parameterization.
//!
//! SSVI parameterizes the entire volatility surface (not just a single tenor)
//! as a function of total variance at-the-money and log-moneyness.
//!
//! # References
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;
use crate::types::{Variance, Vol};

/// SSVI volatility surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsviSurface {
    // Global SSVI parameters will be added during implementation.
    _placeholder: (),
}

impl SsviSurface {
    /// Calibrate an SSVI surface from market data.
    ///
    /// # Errors
    /// Returns [`VolSurfError::CalibrationError`] if calibration fails.
    pub fn calibrate(
        _market_data: &[Vec<(f64, f64)>],
        _tenors: &[f64],
        _forwards: &[f64],
    ) -> error::Result<Self> {
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl VolSurface for SsviSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> error::Result<Variance> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        let _ = expiry;
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}
