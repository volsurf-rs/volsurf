//! Single-tenor volatility smile models.
//!
//! A smile represents how implied volatility varies with strike at a fixed
//! expiry. All models implement the [`SmileSection`] trait.
//!
//! ## Models
//!
//! - [`SviSmile`] — SVI parameterization (Gatheral), 5 parameters
//! - [`SabrSmile`] — SABR stochastic vol model (Hagan et al.), 4 parameters
//! - [`SplineSmile`] — Cubic spline on variance, non-parametric

pub mod arbitrage;
pub mod sabr;
pub mod spline;
pub mod svi;

pub use arbitrage::{ArbitrageReport, ButterflyViolation};
pub use sabr::SabrSmile;
pub use spline::SplineSmile;
pub use svi::SviSmile;

pub(crate) const BUTTERFLY_G_TOL: f64 = 1e-10;
pub(crate) const DENSITY_NEG_TOL: f64 = 1e-8;

use crate::error;
use crate::implied::black::black_price;
use crate::types::{OptionType, Strike, Variance, Vol};
use crate::validate::validate_positive;

/// Grid configuration for butterfly arbitrage scanning.
///
/// Controls the number of sample points and log-moneyness range
/// k = ln(K/F) used when checking `is_arbitrage_free_with`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArbitrageScanConfig {
    /// Sample points across the range, at least 2.
    pub n_points: usize,
    /// Lower end of the scanned log-moneyness range.
    pub k_min: f64,
    /// Upper end of the scanned log-moneyness range, greater than `k_min`.
    pub k_max: f64,
}

impl Default for ArbitrageScanConfig {
    fn default() -> Self {
        Self::wide()
    }
}

impl ArbitrageScanConfig {
    /// 200 points over k ∈ \[−3, 3\], the default grid.
    ///
    /// Wide enough for models valid across the whole wing — SVI, SSVI and
    /// eSSVI scan an analytical g-function, which stays well-behaved out there.
    pub fn wide() -> Self {
        Self {
            n_points: 200,
            k_min: -3.0,
            k_max: 3.0,
        }
    }

    /// 200 points over k ∈ \[−2, 2\].
    ///
    /// For models whose own approximation breaks down before the deep wings do,
    /// which would otherwise report the breakdown as arbitrage. SABR uses this:
    /// the Hagan expansion loses accuracy past |k| ≈ 2.
    pub fn narrow() -> Self {
        Self {
            n_points: 200,
            k_min: -2.0,
            k_max: 2.0,
        }
    }

    pub(crate) fn validate(&self) -> error::Result<()> {
        if self.n_points < 2 {
            return Err(error::VolSurfError::InvalidInput {
                message: format!("n_points must be >= 2, got {}", self.n_points),
            });
        }
        if !self.k_min.is_finite() || !self.k_max.is_finite() {
            return Err(error::VolSurfError::InvalidInput {
                message: "k_min and k_max must be finite".to_string(),
            });
        }
        if self.k_min >= self.k_max {
            return Err(error::VolSurfError::InvalidInput {
                message: format!("k_min ({}) must be < k_max ({})", self.k_min, self.k_max),
            });
        }
        Ok(())
    }
}

/// A single-tenor volatility smile.
///
/// Represents the relationship between strike and implied volatility at a
/// fixed expiry. Every smile model in this crate implements this trait,
/// enabling polymorphic use in surface construction.
///
/// # Thread Safety
/// All implementations must be `Send + Sync` for use in concurrent pricing.
///
/// # Error Handling
/// Methods return `Result` so implementations can report numerical failures
/// (e.g., negative variance, NaN) rather than panicking.
///
/// # Default Methods
///
/// [`density()`](SmileSection::density) computes the risk-neutral density
/// q(K) = d²C/dK² via Breeden-Litzenberger (1978) finite differences with
/// relative step h = K × 10⁻⁴. Override this in models with analytical
/// density (e.g., SVI via the g-function) for better accuracy.
///
/// [`variance()`](SmileSection::variance) derives total variance from
/// `vol()` as σ²T. Override when direct variance computation is cheaper.
///
/// # Examples
///
/// ```
/// use volsurf::smile::{SabrSmile, SmileSection};
/// use volsurf::types::Strike;
///
/// // Create a concrete smile and use it through the trait interface
/// let smile = SabrSmile::new(100.0, 1.0, 0.3, 1.0, -0.3, 0.4)?;
///
/// let vol = smile.vol(Strike(100.0))?;
/// assert!(vol.0 > 0.0);
///
/// let var = smile.variance(Strike(100.0))?;
/// assert!((var.0 - vol.0 * vol.0 * smile.expiry()).abs() < 1e-12);
///
/// let report = smile.is_arbitrage_free()?;
/// assert!(report.is_free());
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
pub trait SmileSection: Send + Sync + std::fmt::Debug {
    /// Implied Black volatility σ at the given strike.
    fn vol(&self, strike: Strike) -> error::Result<Vol>;

    /// Total Black variance σ²T at the given strike.
    ///
    /// Default implementation derives from [`vol`](SmileSection::vol):
    /// `variance(K) = vol(K)² × expiry`.
    fn variance(&self, strike: Strike) -> error::Result<Variance> {
        let v = self.vol(strike)?;
        Ok(Variance(v.0 * v.0 * self.expiry()))
    }

    /// Risk-neutral probability density q(K) via Breeden-Litzenberger.
    ///
    /// Default implementation uses finite differences on Black call prices:
    /// `q(K) = d²C/dK²` with step `h = K × 10⁻⁴`.
    ///
    /// Models with analytical density (e.g., SVI via the g-function) should
    /// override this for better accuracy and performance.
    fn density(&self, strike: Strike) -> error::Result<f64> {
        validate_positive(strike.0, "strike")?;
        // Relative perturbation for central finite difference
        let h = strike.0 * 1e-4;
        let k_lo = strike.0 - h;
        let k_hi = strike.0 + h;

        let v_lo = self.vol(Strike(k_lo))?;
        let v_mid = self.vol(strike)?;
        let v_hi = self.vol(Strike(k_hi))?;

        let c_lo = black_price(self.forward(), k_lo, v_lo, self.expiry(), OptionType::Call)?;
        let c_mid = black_price(
            self.forward(),
            strike.0,
            v_mid,
            self.expiry(),
            OptionType::Call,
        )?;
        let c_hi = black_price(self.forward(), k_hi, v_hi, self.expiry(), OptionType::Call)?;

        // Breeden-Litzenberger: q(K) = d²C/dK² (undiscounted)
        Ok((c_lo - 2.0 * c_mid + c_hi) / (h * h))
    }

    /// Forward price F at this tenor.
    fn forward(&self) -> f64;

    /// Time to expiry T in years.
    fn expiry(&self) -> f64;

    /// Human-readable model name (e.g. "SVI", "SABR", "CubicSpline").
    fn model_name(&self) -> &'static str;

    /// The scan grid [`is_arbitrage_free`](SmileSection::is_arbitrage_free)
    /// runs on.
    ///
    /// Override in models that are only accurate over part of the wing, so a
    /// caller asking for the default check gets this model's own domain rather
    /// than the crate-wide [`wide`](ArbitrageScanConfig::wide) grid.
    fn default_scan_config(&self) -> ArbitrageScanConfig {
        ArbitrageScanConfig::wide()
    }

    /// Check whether this smile is free of butterfly arbitrage.
    ///
    /// Scans [`default_scan_config`](SmileSection::default_scan_config).
    /// Override that to change the grid; override
    /// [`is_arbitrage_free_with`](SmileSection::is_arbitrage_free_with) to
    /// change how the grid is scanned.
    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        self.is_arbitrage_free_with(self.default_scan_config())
    }

    /// Check butterfly arbitrage with custom scan grid configuration.
    ///
    /// Default implementation uses density-based detection (Breeden-Litzenberger)
    /// over `config.n_points` equally-spaced log-moneyness points in
    /// `[config.k_min, config.k_max]`. Models with analytical g-functions
    /// (SVI, SSVI) override this for better accuracy.
    ///
    /// A returned report covers the whole grid actually scanned: if the density
    /// cannot be evaluated at any point, this returns `Err` rather than a partial
    /// scan. Implementations may first narrow `config` to their own domain of
    /// validity — [`SplineSmile`](crate::smile::SplineSmile) clips to its knot
    /// range, since it flat-extrapolates beyond it — so a clean report is not on
    /// its own proof that every point of `config` was examined.
    fn is_arbitrage_free_with(
        &self,
        config: ArbitrageScanConfig,
    ) -> error::Result<ArbitrageReport> {
        arbitrage::scan_density(self.expiry(), self.forward(), config, |strike| {
            self.density(Strike(strike))
        })
    }
}

/// Fits a [`SmileSection`] to one tenor of market quotes.
///
/// Every model runs the same pipeline — validate, filter, resolve weighting,
/// optimize, reconstruct — and this trait is the contract they share. It is
/// also what [`SurfaceBuilder`](crate::surface::SurfaceBuilder) calibrates
/// through, so a model living outside this crate can be built into a surface
/// on the same footing as [`SmileModel`](crate::surface::SmileModel).
///
/// # Examples
///
/// ```
/// use volsurf::calibration::{DataFilter, WeightingScheme};
/// use volsurf::smile::{SmileCalibrator, SmileSection, SplineSmile};
/// use volsurf::surface::{SurfaceBuilder, VolSurface};
/// use volsurf::types::{Strike, Tenor};
///
/// /// Straight-through interpolation of the quotes, no fitting.
/// #[derive(Debug)]
/// struct RawSpline;
///
/// impl SmileCalibrator for RawSpline {
///     fn model_name(&self) -> &'static str {
///         "RawSpline"
///     }
///
///     fn min_strikes(&self) -> usize {
///         3
///     }
///
///     fn calibrate(
///         &self,
///         forward: f64,
///         expiry: f64,
///         market_vols: &[(f64, f64)],
///         _filter: DataFilter,
///         _weighting: WeightingScheme,
///     ) -> volsurf::Result<Box<dyn SmileSection>> {
///         let mut pairs: Vec<(f64, f64)> = market_vols
///             .iter()
///             .map(|&(k, v)| (k, v * v * expiry))
///             .collect();
///         pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
///         let (strikes, variances) = pairs.into_iter().unzip();
///         Ok(Box::new(SplineSmile::new(forward, expiry, strikes, variances)?))
///     }
/// }
///
/// let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
/// let vols = vec![0.24, 0.22, 0.20, 0.22, 0.24];
///
/// let surface = SurfaceBuilder::new()
///     .spot(100.0)
///     .rate(0.05)
///     .calibrator(RawSpline)
///     .add_tenor(0.25, &strikes, &vols)
///     .add_tenor(1.00, &strikes, &vols)
///     .build()?;
///
/// assert_eq!(surface.smile_at(Tenor(0.25))?.model_name(), "CubicSpline");
/// assert!(surface.black_vol(Tenor(0.5), Strike(100.0))?.0 > 0.0);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
pub trait SmileCalibrator: Send + Sync + std::fmt::Debug {
    /// Name used in error messages and diagnostics.
    fn model_name(&self) -> &'static str;

    /// Fewest quotes the model can fit. Checked before [`calibrate`](Self::calibrate).
    fn min_strikes(&self) -> usize;

    /// Check the model's own parameters, independent of any market data.
    ///
    /// [`SurfaceBuilder::build`](crate::surface::SurfaceBuilder::build) calls
    /// this once before it touches a single tenor, so a misconfigured model
    /// reports its own error rather than whatever the first tenor happens to
    /// trip over. Models with no free parameters keep the default.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`](crate::VolSurfError::InvalidInput)
    /// if a parameter fixed at construction is out of range.
    fn validate(&self) -> error::Result<()> {
        Ok(())
    }

    /// Fit the model to `market_vols`, a slice of `(strike, implied_vol)` pairs.
    ///
    /// `filter` is applied to the quotes before fitting; `weighting` sets the
    /// per-quote weights in the objective. Implementations should route both
    /// through [`prepare_market_vols`](crate::calibration::prepare_market_vols)
    /// so filtering behaves consistently across models.
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`](crate::VolSurfError::InvalidInput)
    /// for malformed quotes and
    /// [`VolSurfError::CalibrationError`](crate::VolSurfError::CalibrationError)
    /// if the fit does not converge.
    fn calibrate(
        &self,
        forward: f64,
        expiry: f64,
        market_vols: &[(f64, f64)],
        filter: crate::calibration::DataFilter,
        weighting: crate::calibration::WeightingScheme,
    ) -> error::Result<Box<dyn SmileSection>>;
}
