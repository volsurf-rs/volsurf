//! eSSVI (Extended SSVI) surface with cross-tenor no-arbitrage guarantees.
//!
//! Extends the SSVI parameterization with explicit conditions that guarantee
//! no calendar spread arbitrage between tenors.
//!
//! # References
//! - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)

use crate::error::VolSurfError;
use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;

/// Extended SSVI surface with calendar-spread no-arbitrage guarantees.
#[derive(Debug, Clone)]
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
    ) -> crate::error::Result<Self> {
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl VolSurface for EssviSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> f64 {
        let _ = (expiry, strike);
        unimplemented!()
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> f64 {
        let _ = (expiry, strike);
        unimplemented!()
    }

    fn local_vol(&self, expiry: f64, strike: f64) -> f64 {
        let _ = (expiry, strike);
        unimplemented!()
    }

    fn smile_at(&self, _expiry: f64) -> &dyn SmileSection {
        unimplemented!()
    }

    fn diagnostics(&self) -> SurfaceDiagnostics {
        unimplemented!()
    }
}
