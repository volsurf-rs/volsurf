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
use crate::smile::{SabrSmile, SmileSection, SplineSmile, SviSmile};
use crate::surface::piecewise::PiecewiseSurface;
use crate::validate::{validate_finite, validate_positive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Smile model to use when calibrating each tenor.
///
/// Different models have different trade-offs:
/// - [`Svi`](SmileModel::Svi) fits a parametric 5-parameter curve (minimum 5 strikes)
/// - [`CubicSpline`](SmileModel::CubicSpline) interpolates variance directly (minimum 3 strikes)
/// - [`Sabr`](SmileModel::Sabr) fits the SABR stochastic vol model (minimum 4 strikes)
///
/// # Examples
///
/// ```
/// use volsurf::surface::SmileModel;
///
/// let svi = SmileModel::Svi;            // default, 5+ strikes
/// let spline = SmileModel::CubicSpline; // 3+ strikes
/// let sabr = SmileModel::Sabr { beta: 0.5 }; // 4+ strikes, equity backbone
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SmileModel {
    /// SVI parametric model (Gatheral 2004). Requires ≥ 5 strikes per tenor.
    #[default]
    Svi,
    /// Cubic spline on total variance. Requires ≥ 3 strikes per tenor.
    CubicSpline,
    /// SABR stochastic volatility model (Hagan 2002). Requires ≥ 4 strikes per tenor.
    ///
    /// `beta` is the CEV exponent, fixed by the user (industry convention):
    /// - `beta = 0.0`: normal model (rates)
    /// - `beta = 0.5`: CIR-like (equities)
    /// - `beta = 1.0`: lognormal model
    Sabr {
        /// CEV exponent, must be in \[0, 1\].
        beta: f64,
    },
}

/// Builder for constructing volatility surfaces from market data.
///
/// Accumulates spot price, risk-free rate, and per-tenor (strikes, vols)
/// data, then calibrates smiles and assembles a [`PiecewiseSurface`].
///
/// # Examples
///
/// ```
/// use volsurf::surface::{SurfaceBuilder, SmileModel, VolSurface};
///
/// let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
/// let vols = vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28];
///
/// let surface = SurfaceBuilder::new()
///     .spot(100.0)
///     .rate(0.05)
///     .model(SmileModel::Sabr { beta: 0.5 })
///     .add_tenor(0.25, &strikes, &vols)
///     .add_tenor(1.00, &strikes, &vols)
///     .build()?;
///
/// let vol = surface.black_vol(0.5, 100.0)?;
/// assert!(vol.0 > 0.0);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
#[derive(Debug)]
pub struct SurfaceBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    model: SmileModel,
    tenor_data: Vec<TenorData>,
}

#[derive(Debug)]
struct TenorData {
    expiry: f64,
    strikes: Vec<f64>,
    vols: Vec<f64>,
    forward: Option<f64>,
}

impl SurfaceBuilder {
    /// Create a new surface builder with default settings (SVI model).
    pub fn new() -> Self {
        Self {
            spot: None,
            rate: None,
            dividend_yield: None,
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

    /// Set the continuous dividend yield q for forward calculation.
    ///
    /// Forward price becomes F = S · exp((r − q) · T). Default is 0.
    pub fn dividend_yield(mut self, q: f64) -> Self {
        self.dividend_yield = Some(q);
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
            forward: None,
        });
        self
    }

    /// Add market data for a tenor with an explicit forward price.
    ///
    /// Bypasses the built-in F = S · exp((r − q) · T) calculation for this
    /// tenor. Use for futures options where the forward is the futures price,
    /// or when you have a better forward estimate (e.g. from put-call parity).
    pub fn add_tenor_with_forward(
        mut self,
        expiry: f64,
        strikes: &[f64],
        vols: &[f64],
        forward: f64,
    ) -> Self {
        self.tenor_data.push(TenorData {
            expiry,
            strikes: strikes.to_vec(),
            vols: vols.to_vec(),
            forward: Some(forward),
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

        // Validate required fields
        let spot = self.spot.ok_or_else(|| VolSurfError::InvalidInput {
            message: "spot price is required".into(),
        })?;
        let rate = self.rate.ok_or_else(|| VolSurfError::InvalidInput {
            message: "risk-free rate is required".into(),
        })?;

        let q = self.dividend_yield.unwrap_or(0.0);

        validate_positive(spot, "spot")?;
        validate_finite(rate, "rate")?;
        validate_finite(q, "dividend_yield")?;
        if self.tenor_data.is_empty() {
            return Err(VolSurfError::InvalidInput {
                message: "at least one tenor is required".into(),
            });
        }

        let min_strikes = match self.model {
            SmileModel::Svi => 5,
            SmileModel::CubicSpline => 3,
            SmileModel::Sabr { .. } => 4,
        };

        if let SmileModel::Sabr { beta } = self.model
            && (!beta.is_finite() || !(0.0..=1.0).contains(&beta))
        {
            return Err(VolSurfError::InvalidInput {
                message: format!("SABR beta must be in [0, 1] and finite, got {beta}"),
            });
        }

        let model = self.model;
        let calibrate_tenor =
            |tenor: &TenorData| -> crate::error::Result<(f64, Box<dyn SmileSection>)> {
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
                            "at least {min_strikes} strikes required per tenor (model: {model:?}), got {} for tenor {}",
                            tenor.strikes.len(),
                            tenor.expiry
                        ),
                    });
                }

                if let Some(fwd) = tenor.forward {
                    validate_positive(fwd, "per-tenor forward")?;
                }
                let forward = tenor
                    .forward
                    .unwrap_or_else(|| conventions::forward_price(spot, rate, q, tenor.expiry));

                let smile: Box<dyn SmileSection> = match model {
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
                        let mut pairs: Vec<(f64, f64)> = tenor
                            .strikes
                            .iter()
                            .zip(tenor.vols.iter())
                            .map(|(&k, &v)| (k, v * v * tenor.expiry))
                            .collect();
                        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let (strikes, variances): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
                        let spline = SplineSmile::new(forward, tenor.expiry, strikes, variances)?;
                        Box::new(spline)
                    }
                    SmileModel::Sabr { beta } => {
                        let market_vols: Vec<(f64, f64)> = tenor
                            .strikes
                            .iter()
                            .zip(tenor.vols.iter())
                            .map(|(&k, &v)| (k, v))
                            .collect();
                        let sabr = SabrSmile::calibrate(forward, tenor.expiry, beta, &market_vols)?;
                        Box::new(sabr)
                    }
                };

                Ok((tenor.expiry, smile))
            };

        #[cfg(feature = "parallel")]
        let mut tenor_smile_pairs: Vec<(f64, Box<dyn SmileSection>)> = self
            .tenor_data
            .par_iter()
            .map(calibrate_tenor)
            .collect::<crate::error::Result<Vec<_>>>()?;
        #[cfg(not(feature = "parallel"))]
        let mut tenor_smile_pairs: Vec<(f64, Box<dyn SmileSection>)> = self
            .tenor_data
            .iter()
            .map(calibrate_tenor)
            .collect::<crate::error::Result<Vec<_>>>()?;

        // Sort by tenor
        tenor_smile_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Assemble PiecewiseSurface
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

    // Gap #6: CubicSpline with NaN and duplicate strikes

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
        assert!(
            result.is_err(),
            "duplicate strikes should cause build to fail"
        );
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
        assert!(
            result.is_ok(),
            "unsorted strikes should be sorted by builder"
        );
    }

    #[test]
    fn default_is_same_as_new() {
        let builder = SurfaceBuilder::default();
        // Just verify it doesn't panic
        let result = builder.build();
        assert!(result.is_err()); // no spot/rate/tenors
    }

    // Gap #7: Inf rate rejected, large negative rate accepted

    #[test]
    fn inf_rate_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(f64::INFINITY)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn neg_inf_rate_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(f64::NEG_INFINITY)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    // Gap #8: Mismatched lengths error message content

    #[test]
    fn mismatched_lengths_error_message_contains_counts() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let vols = vec![0.20, 0.20, 0.20]; // 3 vols for 5 strikes
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        match result {
            Err(VolSurfError::InvalidInput { message }) => {
                assert!(
                    message.contains("5") && message.contains("3"),
                    "error should mention both lengths: {message}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    // Gap #9: Negative strikes cause calibration failure

    #[test]
    fn negative_strikes_cause_calibration_error() {
        let strikes = vec![-80.0, -90.0, -100.0, -110.0, -120.0];
        let vols = vec![0.28, 0.24, 0.20, 0.24, 0.28];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(
            result.is_err(),
            "negative strikes should cause build to fail"
        );
    }

    #[test]
    fn zero_strike_in_data_causes_error() {
        let strikes = vec![0.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
        let vols = vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(result.is_err(), "zero strike should cause build to fail");
    }

    #[test]
    fn build_with_sabr_model() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let vol = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol.0 > 0.0, "ATM vol should be positive");
        assert!(vol.0 < 1.0, "ATM vol should be reasonable");
    }

    #[test]
    fn sabr_with_4_strikes_succeeds() {
        let strikes = vec![90.0, 95.0, 100.0, 110.0];
        let vols = vec![0.24, 0.22, 0.20, 0.24];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 })
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(result.is_ok(), "SABR should work with 4 strikes");
    }

    #[test]
    fn sabr_with_3_strikes_fails() {
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.24, 0.20, 0.24];
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 })
            .add_tenor(0.25, &strikes, &vols)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn sabr_invalid_beta_negative() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: -0.1 })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn sabr_invalid_beta_above_1() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 1.1 })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn sabr_invalid_beta_nan() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: f64::NAN })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn sabr_multi_tenor_cross_tenor_query() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        // Query between tenors
        let vol = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol.0 > 0.0);

        // Query at/near stored tenors
        for t in [0.25, 0.5, 1.0] {
            for k in [80.0, 100.0, 120.0] {
                let v = surface.black_vol(t, k).unwrap();
                assert!(
                    v.0 > 0.0 && v.0 < 2.0,
                    "vol({t}, {k}) = {} out of range",
                    v.0
                );
            }
        }
    }

    #[test]
    fn sabr_vol_and_variance_consistent() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let t = 0.25;
        let k = 100.0;
        let vol = surface.black_vol(t, k).unwrap();
        let var = surface.black_variance(t, k).unwrap();
        assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-12);
    }

    #[test]
    fn sabr_beta_zero_normal_model() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.0 })
            .add_tenor(0.5, &sample_strikes(), &sample_vols())
            .build();
        assert!(
            result.is_ok(),
            "beta=0 (normal SABR) should build successfully"
        );
    }

    #[test]
    fn sabr_beta_one_lognormal_model() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 1.0 })
            .add_tenor(0.5, &sample_strikes(), &sample_vols())
            .build();
        assert!(
            result.is_ok(),
            "beta=1 (lognormal SABR) should build successfully"
        );
    }

    #[test]
    fn build_with_dividend_yield() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.02)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let smile = surface.smile_at(1.0).unwrap();
        let expected_fwd = 100.0 * (0.03_f64).exp();
        assert_abs_diff_eq!(smile.forward(), expected_fwd, epsilon = 0.5);
    }

    #[test]
    fn dividend_yield_zero_matches_no_dividend_yield() {
        let surface_no_q = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();
        let surface_q0 = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(0.0)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let fwd_no_q = surface_no_q.smile_at(1.0).unwrap().forward();
        let fwd_q0 = surface_q0.smile_at(1.0).unwrap().forward();
        assert_abs_diff_eq!(fwd_no_q, fwd_q0, epsilon = 1e-12);
    }

    #[test]
    fn nan_dividend_yield_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(f64::NAN)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn inf_dividend_yield_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .dividend_yield(f64::INFINITY)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn add_tenor_with_forward_bypasses_forward_price() {
        let explicit_fwd = 50.0;
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), explicit_fwd)
            .build()
            .unwrap();

        let smile = surface.smile_at(1.0).unwrap();
        // Should use the explicit 50.0, not 100*exp(0.05) ≈ 105.13
        assert_abs_diff_eq!(smile.forward(), explicit_fwd, epsilon = 1e-6);
    }

    #[test]
    fn add_tenor_with_forward_zero_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), 0.0)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn add_tenor_with_forward_negative_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), -50.0)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn add_tenor_with_forward_nan_returns_invalid_input() {
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), f64::NAN)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn mixed_add_tenor_and_add_tenor_with_forward() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), 105.0)
            .build()
            .unwrap();

        let vol = surface.black_vol(0.5, 100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }

    #[test]
    fn build_ten_tenor_surface() {
        let strikes = sample_strikes();
        let vols = sample_vols();
        let mut builder = SurfaceBuilder::new().spot(100.0).rate(0.05);
        for i in 1..=10 {
            builder = builder.add_tenor(i as f64 * 0.25, &strikes, &vols);
        }
        let surface = builder.build().unwrap();

        for i in 1..=10 {
            let t = i as f64 * 0.25;
            let vol = surface.black_vol(t, 100.0).unwrap();
            assert!(vol.0 > 0.0 && vol.0 < 1.0, "bad vol {vol:?} at T={t}");
        }
        let interp = surface.black_vol(1.375, 100.0).unwrap();
        assert!(interp.0 > 0.0 && interp.0 < 1.0);
    }

    #[test]
    fn bad_tenor_among_good_tenors_returns_error() {
        let strikes = sample_strikes();
        let vols = sample_vols();
        let result = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &strikes, &vols)
            .add_tenor(0.5, &strikes, &vols)
            .add_tenor(-1.0, &strikes, &vols) // bad
            .add_tenor(1.0, &strikes, &vols)
            .build();
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn ten_tenor_cubic_spline_surface() {
        let strikes = sample_strikes();
        let vols = sample_vols();
        let mut builder = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline);
        for i in 1..=10 {
            builder = builder.add_tenor(i as f64 * 0.25, &strikes, &vols);
        }
        let surface = builder.build().unwrap();
        let vol = surface.black_vol(1.0, 100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }

    #[test]
    fn ten_tenor_sabr_surface() {
        let strikes = sample_strikes();
        let vols = sample_vols();
        let mut builder = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: 0.5 });
        for i in 1..=10 {
            builder = builder.add_tenor(i as f64 * 0.25, &strikes, &vols);
        }
        let surface = builder.build().unwrap();
        let vol = surface.black_vol(1.0, 100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }
}
