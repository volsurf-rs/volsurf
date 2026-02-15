//! Cubic spline interpolation of variance with monotonicity constraints.
//!
//! A non-parametric smile model that fits a cubic spline through observed
//! variance points. Useful when parametric models (SVI, SABR) fail to
//! capture exotic smile shapes.
//!
//! # References
//! - Fengler, M.R. "Arbitrage-free Smoothing of the Implied Volatility Surface" (2009)

use crate::error::VolSurfError;
use crate::smile::arbitrage::ArbitrageReport;
use crate::smile::SmileSection;

/// Cubic spline smile on variance with optional monotonicity constraints.
#[derive(Debug, Clone)]
pub struct SplineSmile {
    forward: f64,
    expiry: f64,
    strikes: Vec<f64>,
    variances: Vec<f64>,
}

impl SplineSmile {
    /// Create a spline smile from strike-variance pairs.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if fewer than 3 data points
    /// are provided or if strikes are not strictly increasing.
    pub fn new(
        forward: f64,
        expiry: f64,
        strikes: Vec<f64>,
        variances: Vec<f64>,
    ) -> crate::error::Result<Self> {
        let _ = (forward, expiry, &strikes, &variances);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}

impl SmileSection for SplineSmile {
    fn vol(&self, strike: f64) -> f64 {
        let _ = strike;
        unimplemented!()
    }

    fn density(&self, strike: f64) -> f64 {
        let _ = strike;
        unimplemented!()
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    fn is_arbitrage_free(&self) -> ArbitrageReport {
        unimplemented!()
    }
}
