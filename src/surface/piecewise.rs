//! Piecewise surface: per-tenor SmileSections with cross-tenor interpolation.
//!
//! The most flexible surface representation. Each tenor has its own
//! independently calibrated [`SmileSection`], and cross-tenor queries
//! interpolate in variance space to maintain no-calendar-arbitrage.

use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;

/// Piecewise volatility surface composed of per-tenor smile sections.
pub struct PiecewiseSurface {
    /// Sorted tenors (time to expiry in years).
    tenors: Vec<f64>,
    /// One smile section per tenor.
    smiles: Vec<Box<dyn SmileSection>>,
}

impl PiecewiseSurface {
    /// Create a piecewise surface from a set of calibrated smiles.
    ///
    /// `tenors` and `smiles` must have the same length and tenors must be
    /// strictly increasing.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if lengths mismatch or tenors
    /// are not sorted.
    pub fn new(
        tenors: Vec<f64>,
        smiles: Vec<Box<dyn SmileSection>>,
    ) -> crate::error::Result<Self> {
        let _ = (&tenors, &smiles);
        Err(crate::VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl VolSurface for PiecewiseSurface {
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
