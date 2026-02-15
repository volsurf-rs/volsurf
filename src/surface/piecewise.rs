//! Piecewise surface: per-tenor SmileSections with cross-tenor interpolation.
//!
//! The most flexible surface representation. Each tenor has its own
//! independently calibrated [`SmileSection`], and cross-tenor queries
//! interpolate in variance space to maintain no-calendar-arbitrage.

use std::fmt;

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;
use crate::types::{Variance, Vol};

/// Piecewise volatility surface composed of per-tenor smile sections.
pub struct PiecewiseSurface {
    /// Sorted tenors (time to expiry in years).
    tenors: Vec<f64>,
    /// One smile section per tenor.
    smiles: Vec<Box<dyn SmileSection>>,
}

impl fmt::Debug for PiecewiseSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PiecewiseSurface")
            .field("tenors", &self.tenors)
            .field("num_smiles", &self.smiles.len())
            .finish()
    }
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
    ) -> error::Result<Self> {
        let _ = (&tenors, &smiles);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl VolSurface for PiecewiseSurface {
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

    fn smile_at(&self, _expiry: f64) -> error::Result<Box<dyn SmileSection>> {
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
