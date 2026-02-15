//! eSSVI (Extended SSVI) surface with cross-tenor no-arbitrage guarantees.
//!
//! Extends the SSVI parameterization with explicit conditions that guarantee
//! no calendar spread arbitrage between tenors.
//!
//! # References
//! - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)

#![allow(dead_code)] // Stub — not yet implemented (v0.3 scope)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;
use crate::types::{Variance, Vol};

/// Extended SSVI surface with calendar-spread no-arbitrage guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssviSurface {
    _placeholder: (),
}

impl EssviSurface {
    /// Calibrate an eSSVI surface from market data.
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

impl VolSurface for EssviSurface {
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
