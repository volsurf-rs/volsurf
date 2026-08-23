//! Ergonomic builder API for volatility surface construction.
//!
//! ```
//! use volsurf::surface::{SurfaceBuilder, VolSurface};
//! use volsurf::types::{Strike, Tenor};
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
//! let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
//! assert!(vol.0 > 0.0);
//! ```

use serde::{Deserialize, Serialize};

use crate::calibration::{DataFilter, WeightingScheme};
use crate::conventions;
use crate::error::VolSurfError;
use crate::smile::{SabrSmile, SmileCalibrator, SmileSection, SplineSmile, SviSmile};
use crate::surface::piecewise::PiecewiseSurface;
use crate::validate::{validate_finite, validate_in_range, validate_positive};
use std::sync::Arc;

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
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "SmileModelRaw")]
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

#[derive(Deserialize)]
enum SmileModelRaw {
    Svi,
    CubicSpline,
    Sabr { beta: f64 },
}

impl TryFrom<SmileModelRaw> for SmileModel {
    type Error = String;
    fn try_from(raw: SmileModelRaw) -> std::result::Result<Self, String> {
        match raw {
            SmileModelRaw::Svi => Ok(Self::Svi),
            SmileModelRaw::CubicSpline => Ok(Self::CubicSpline),
            SmileModelRaw::Sabr { beta } => {
                let model = Self::Sabr { beta };
                model.validate().map_err(|e| match e {
                    VolSurfError::InvalidInput { message } => message,
                    other => other.to_string(),
                })?;
                Ok(model)
            }
        }
    }
}

impl SmileCalibrator for SmileModel {
    fn model_name(&self) -> &'static str {
        match self {
            Self::Svi => "SVI",
            Self::CubicSpline => "CubicSpline",
            Self::Sabr { .. } => "SABR",
        }
    }

    fn min_strikes(&self) -> usize {
        match self {
            Self::Svi => 5,
            Self::CubicSpline => 3,
            Self::Sabr { .. } => 4,
        }
    }

    fn validate(&self) -> crate::error::Result<()> {
        match *self {
            Self::Sabr { beta } => validate_in_range(beta, 0.0, 1.0, "SABR beta").map(|_| ()),
            Self::Svi | Self::CubicSpline => Ok(()),
        }
    }

    fn calibrate(
        &self,
        forward: f64,
        expiry: f64,
        market_vols: &[(f64, f64)],
        filter: DataFilter,
        weighting: WeightingScheme,
    ) -> crate::error::Result<Box<dyn SmileSection>> {
        match *self {
            Self::Svi => Ok(Box::new(SviSmile::calibrate_with_config(
                forward,
                expiry,
                market_vols,
                filter,
                weighting,
                None,
            )?)),

            Self::CubicSpline => Ok(Box::new(SplineSmile::calibrate_with_config(
                forward,
                expiry,
                market_vols,
                filter,
            )?)),

            Self::Sabr { beta } => Ok(Box::new(SabrSmile::calibrate_with_config(
                forward,
                expiry,
                beta,
                market_vols,
                filter,
                weighting,
                None,
            )?)),
        }
    }
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
/// use volsurf::types::{Strike, Tenor};
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
/// let vol = surface.black_vol(Tenor(0.5), Strike(100.0))?;
/// assert!(vol.0 > 0.0);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SurfaceBuilder {
    spot: Option<f64>,
    rate: Option<f64>,
    dividend_yield: Option<f64>,
    calibrator: Arc<dyn SmileCalibrator>,
    data_filter: Option<DataFilter>,
    weighting: Option<WeightingScheme>,
    tenor_data: Vec<TenorData>,
}

#[derive(Debug, Clone)]
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
            calibrator: Arc::new(SmileModel::default()),
            data_filter: None,
            weighting: None,
            tenor_data: Vec::new(),
        }
    }

    /// Set the smile model used for per-tenor calibration.
    ///
    /// Default is [`SmileModel::Svi`]. For a model of your own, see
    /// [`calibrator`](Self::calibrator).
    pub fn model(self, model: SmileModel) -> Self {
        self.calibrator(model)
    }

    /// Calibrate each tenor with an arbitrary [`SmileCalibrator`].
    ///
    /// The built-in [`SmileModel`] variants implement the trait, so
    /// [`model`](Self::model) is the same call with the enum; this one also
    /// accepts a model defined outside the crate.
    pub fn calibrator(mut self, calibrator: impl SmileCalibrator + 'static) -> Self {
        self.calibrator = Arc::new(calibrator);
        self
    }

    /// Set pre-calibration strike/vol filtering applied to each tenor.
    pub fn data_filter(mut self, filter: DataFilter) -> Self {
        self.data_filter = Some(filter);
        self
    }

    /// Set the weighting scheme for calibration objective functions.
    pub fn weighting(mut self, weighting: WeightingScheme) -> Self {
        self.weighting = Some(weighting);
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
    /// Computes forward prices, calibrates a smile per tenor, sorts by expiry,
    /// and assembles a [`PiecewiseSurface`]. The result is not serializable
    /// (trait-object storage); use [`SsviSurface`](super::SsviSurface) or
    /// [`EssviSurface`](super::EssviSurface) when
    /// persistence is needed.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if the calibrator's own
    /// parameters are invalid — checked before any tenor data, see
    /// [`SmileCalibrator::validate`] — if
    /// tenor data is invalid, or if `spot`/`rate` are missing while a tenor
    /// still needs its forward derived; tenors added via
    /// [`add_tenor_with_forward`](Self::add_tenor_with_forward) need neither.
    /// Returns [`VolSurfError::CalibrationError`] if calibration fails for any
    /// tenor.
    pub fn build(self) -> crate::error::Result<PiecewiseSurface> {
        #[cfg(feature = "logging")]
        tracing::debug!(
            n_tenors = self.tenor_data.len(),
            model = self.calibrator.model_name(),
            "surface build started"
        );

        // spot and rate are only needed to derive forwards, so they are
        // required per-tenor rather than here — see the forward resolution below.
        let spot = self.spot;
        let rate = self.rate;

        let q = self.dividend_yield.unwrap_or(0.0);

        validate_finite(q, "dividend_yield")?;
        if self.tenor_data.is_empty() {
            return Err(VolSurfError::InvalidInput {
                message: "at least one tenor is required".into(),
            });
        }

        let calibrator = self.calibrator.as_ref();
        calibrator.validate()?;
        let min_strikes = calibrator.min_strikes();
        let model_name = calibrator.model_name();
        let filter = self.data_filter.unwrap_or_default();
        let weighting = self.weighting.unwrap_or_default();
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
                            "at least {min_strikes} strikes required per tenor (model: {model_name}), got {} for tenor {}",
                            tenor.strikes.len(),
                            tenor.expiry
                        ),
                    });
                }

                let forward = match tenor.forward {
                    Some(fwd) => {
                        validate_positive(fwd, "per-tenor forward")?;
                        fwd
                    }
                    None => {
                        let spot = spot.ok_or_else(|| VolSurfError::InvalidInput {
                            message: "spot price is required".into(),
                        })?;
                        let rate = rate.ok_or_else(|| VolSurfError::InvalidInput {
                            message: "risk-free rate is required".into(),
                        })?;
                        validate_positive(spot, "spot")?;
                        validate_finite(rate, "rate")?;
                        conventions::forward_price(spot, rate, q, tenor.expiry)?
                    }
                };

                let market_vols: Vec<(f64, f64)> = tenor
                    .strikes
                    .iter()
                    .zip(&tenor.vols)
                    .map(|(&strike, &vol)| (strike, vol))
                    .collect();

                let smile =
                    calibrator.calibrate(forward, tenor.expiry, &market_vols, filter, weighting)?;

                Ok((tenor.expiry, smile))
            };

        #[cfg(feature = "parallel")]
        let tenors = self.tenor_data.par_iter();
        #[cfg(not(feature = "parallel"))]
        let tenors = self.tenor_data.iter();

        let mut tenor_smile_pairs: Vec<(f64, Box<dyn SmileSection>)> = tenors
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
    use crate::types::{Strike, Tenor};
    use approx::assert_abs_diff_eq;

    /// Market data: symmetric U-shaped smile with 7 strikes.
    fn sample_strikes() -> Vec<f64> {
        vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    }

    fn sample_vols() -> Vec<f64> {
        vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28]
    }

    /// A calibrator defined outside the `SmileModel` enum.
    #[derive(Debug)]
    struct ConstantVolSmileFit {
        vol: f64,
        min_strikes: usize,
    }

    impl SmileCalibrator for ConstantVolSmileFit {
        fn model_name(&self) -> &'static str {
            "ConstantVol"
        }

        fn min_strikes(&self) -> usize {
            self.min_strikes
        }

        fn calibrate(
            &self,
            forward: f64,
            expiry: f64,
            market_vols: &[(f64, f64)],
            _filter: DataFilter,
            _weighting: WeightingScheme,
        ) -> crate::error::Result<Box<dyn SmileSection>> {
            let mut strikes: Vec<f64> = market_vols.iter().map(|&(k, _)| k).collect();
            strikes.sort_by(f64::total_cmp);
            let variances = vec![self.vol * self.vol * expiry; strikes.len()];
            Ok(Box::new(SplineSmile::new(
                forward, expiry, strikes, variances,
            )?))
        }
    }

    #[test]
    fn build_with_a_calibrator_defined_outside_the_crate() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .calibrator(ConstantVolSmileFit {
                vol: 0.25,
                min_strikes: 3,
            })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        // The custom fit ignores the quotes and returns a flat 25% smile.
        for t in [0.25, 1.0] {
            for k in [90.0, 100.0, 110.0] {
                assert_abs_diff_eq!(
                    surface.black_vol(Tenor(t), Strike(k)).unwrap().0,
                    0.25,
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn custom_calibrator_min_strikes_is_enforced() {
        let err = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .calibrator(ConstantVolSmileFit {
                vol: 0.25,
                min_strikes: 99,
            })
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap_err();

        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
        let message = err.to_string();
        assert!(message.contains("at least 99 strikes"), "got {message}");
        assert!(message.contains("ConstantVol"), "got {message}");
    }

    #[test]
    fn model_and_calibrator_are_the_same_call_for_built_in_models() {
        let via_model = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::CubicSpline)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();
        let via_calibrator = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .calibrator(SmileModel::CubicSpline)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        assert_abs_diff_eq!(
            via_model.black_vol(Tenor(0.25), Strike(105.0)).unwrap().0,
            via_calibrator
                .black_vol(Tenor(0.25), Strike(105.0))
                .unwrap()
                .0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn build_single_tenor_surface() {
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor(0.25, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        let vol = surface.black_vol(Tenor(0.25), Strike(100.0)).unwrap();
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
        let vol_3m = surface.black_vol(Tenor(0.25), Strike(100.0)).unwrap();
        assert!(vol_3m.0 > 0.0);

        // Query between tenors
        let vol_6m = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
        assert!(vol_6m.0 > 0.0);

        // Query at stored tenor
        let vol_1y = surface.black_vol(Tenor(1.0), Strike(100.0)).unwrap();
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

        let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
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
                let vol = surface.black_vol(Tenor(t), Strike(k)).unwrap();
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
        let vol = surface.black_vol(Tenor(t), Strike(k)).unwrap();
        let var = surface.black_variance(Tenor(t), Strike(k)).unwrap();
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
    fn all_explicit_forwards_need_no_spot_or_rate() {
        // Futures options: the forward is the futures price, so there is no
        // carry to compute and no spot to quote.
        let result = SurfaceBuilder::new()
            .add_tenor_with_forward(0.25, &sample_strikes(), &sample_vols(), 101.0)
            .add_tenor_with_forward(1.0, &sample_strikes(), &sample_vols(), 105.0)
            .build();
        assert!(
            result.is_ok(),
            "explicit forwards should not need carry inputs"
        );
    }

    #[test]
    fn mixed_forwards_still_need_spot_for_the_derived_tenor() {
        let result = SurfaceBuilder::new()
            .add_tenor_with_forward(0.25, &sample_strikes(), &sample_vols(), 101.0)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build();
        match result {
            Err(VolSurfError::InvalidInput { message }) => {
                assert!(
                    message.contains("spot"),
                    "expected a spot error, got: {message}"
                );
            }
            other => panic!("expected InvalidInput about spot, got {other:?}"),
        }
    }

    #[test]
    fn mixed_forwards_build_once_carry_is_supplied() {
        // With carry available, the explicit forward still wins for its own
        // tenor; only the tenor without one derives from spot and rate.
        let surface = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .add_tenor_with_forward(0.25, &sample_strikes(), &sample_vols(), 101.0)
            .add_tenor(1.0, &sample_strikes(), &sample_vols())
            .build()
            .unwrap();

        assert_abs_diff_eq!(
            surface.smile_at(Tenor(0.25)).unwrap().forward(),
            101.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            surface.smile_at(Tenor(1.0)).unwrap().forward(),
            100.0 * 0.05_f64.exp(),
            epsilon = 1e-12
        );
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
        assert_eq!(builder.calibrator.model_name(), "SVI");
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

        let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
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
        // `calibrate_with_config` validates every quote before the filter runs, so
        // the NaN strike errors outright rather than being filtered out
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
        // `calibrate_with_config` sorts the quotes and catches the shared strike
        // itself, before the knots ever reach `new`
        assert!(
            result.is_err(),
            "duplicate strikes should cause build to fail"
        );
    }

    #[test]
    fn cubic_spline_with_unsorted_strikes_succeeds() {
        // `calibrate_with_config` sorts the quotes by strike before building the knots
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
            "unsorted strikes should be sorted during calibration"
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

        let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
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

    /// A bad beta is the model's own error, so it must surface before any
    /// per-tenor check — here, too few strikes for SABR's `min_strikes` of 4.
    #[test]
    fn sabr_invalid_beta_reported_before_tenor_checks() {
        let err = SurfaceBuilder::new()
            .spot(100.0)
            .rate(0.05)
            .model(SmileModel::Sabr { beta: f64::NAN })
            .add_tenor(0.25, &[95.0, 100.0], &[0.22, 0.20])
            .build()
            .unwrap_err();
        let VolSurfError::InvalidInput { message } = err else {
            panic!("expected InvalidInput, got {err:?}");
        };
        assert!(message.contains("SABR beta"), "got {message}");
    }

    /// Every SABR beta check — smile constructor, calibrator, `SmileCalibrator::validate`,
    /// and the serde path — must produce byte-identical wording.
    #[test]
    fn sabr_beta_message_is_identical_across_all_call_sites() {
        const EXPECTED: &str = "SABR beta must be in [0, 1], got 2";

        let from_new = SabrSmile::new(100.0, 1.0, 0.2, 2.0, -0.3, 0.4).unwrap_err();
        let from_calibrate = SabrSmile::calibrate_with_config(
            100.0,
            1.0,
            2.0,
            &[],
            DataFilter::default(),
            WeightingScheme::default(),
            None,
        )
        .unwrap_err();
        let from_validate = SmileModel::Sabr { beta: 2.0 }.validate().unwrap_err();
        let from_serde =
            serde_json::from_str::<SmileModel>(r#"{"Sabr":{"beta":2.0}}"#).unwrap_err();

        for err in [&from_new, &from_calibrate, &from_validate] {
            let VolSurfError::InvalidInput { message } = err else {
                panic!("expected InvalidInput, got {err:?}");
            };
            assert_eq!(message, EXPECTED);
        }
        assert!(from_serde.to_string().contains(EXPECTED), "{from_serde}");
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
        let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
        assert!(vol.0 > 0.0);

        // Query at/near stored tenors
        for t in [0.25, 0.5, 1.0] {
            for k in [80.0, 100.0, 120.0] {
                let v = surface.black_vol(Tenor(t), Strike(k)).unwrap();
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
        let vol = surface.black_vol(Tenor(t), Strike(k)).unwrap();
        let var = surface.black_variance(Tenor(t), Strike(k)).unwrap();
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

        let smile = surface.smile_at(Tenor(1.0)).unwrap();
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

        let fwd_no_q = surface_no_q.smile_at(Tenor(1.0)).unwrap().forward();
        let fwd_q0 = surface_q0.smile_at(Tenor(1.0)).unwrap().forward();
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

        let smile = surface.smile_at(Tenor(1.0)).unwrap();
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

        let vol = surface.black_vol(Tenor(0.5), Strike(100.0)).unwrap();
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
            let vol = surface.black_vol(Tenor(t), Strike(100.0)).unwrap();
            assert!(vol.0 > 0.0 && vol.0 < 1.0, "bad vol {vol:?} at T={t}");
        }
        let interp = surface.black_vol(Tenor(1.375), Strike(100.0)).unwrap();
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
        let vol = surface.black_vol(Tenor(1.0), Strike(100.0)).unwrap();
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
        let vol = surface.black_vol(Tenor(1.0), Strike(100.0)).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }

    #[test]
    fn smile_model_serde_round_trip() {
        for model in [
            SmileModel::Svi,
            SmileModel::CubicSpline,
            SmileModel::Sabr { beta: 0.5 },
        ] {
            let json = serde_json::to_string(&model).unwrap();
            let roundtrip: SmileModel = serde_json::from_str(&json).unwrap();
            assert_eq!(model, roundtrip);
        }
    }

    #[test]
    fn smile_model_sabr_json_shape() {
        let model = SmileModel::Sabr { beta: 0.5 };
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("Sabr"));
        assert!(json.contains("beta"));
    }

    #[test]
    fn smile_model_sabr_rejects_invalid_beta() {
        for bad in [r#"{"Sabr":{"beta":-0.1}}"#, r#"{"Sabr":{"beta":1.5}}"#] {
            assert!(serde_json::from_str::<SmileModel>(bad).is_err());
        }
    }

    #[test]
    fn smile_model_sabr_boundary_beta() {
        for beta in [0.0, 1.0] {
            let json = format!(r#"{{"Sabr":{{"beta":{beta}}}}}"#);
            let model: SmileModel = serde_json::from_str(&json).unwrap();
            assert_eq!(model, SmileModel::Sabr { beta });
        }
    }
}
