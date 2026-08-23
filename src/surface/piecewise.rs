//! Piecewise surface: per-tenor SmileSections with cross-tenor interpolation.
//!
//! The most flexible surface representation. Each tenor has its own
//! independently calibrated [`SmileSection`], and cross-tenor queries
//! interpolate linearly in total variance space to maintain no-calendar-arbitrage.
//!
//! # Interpolation Strategy
//!
//! Total variance `w(T, K) = σ²(T,K)·T` is interpolated linearly between
//! bracketing tenors:
//!
//! ```text
//! w(T, K) = (1 − α)·w(T₁, K) + α·w(T₂, K)
//! ```
//!
//! where `α = (T − T₁)/(T₂ − T₁)`. This preserves the no-calendar-arbitrage
//! condition: if `w(T₁, K) ≤ w(T₂, K)` for all K, then the interpolated
//! variance also satisfies this monotonicity.
//!
//! # Extrapolation
//!
//! - Before the first tenor: flat vol (variance scales as `w₁ · T/T₁`)
//! - After the last tenor: flat vol (variance scales as `wₙ · T/Tₙ`)

use std::fmt;
use std::sync::Arc;

use crate::error::{self, VolSurfError};
use crate::smile::spline::SplineSmile;
use crate::smile::{ArbitrageScanConfig, SmileSection};
use crate::surface::EXPIRY_MATCH_TOL;
use crate::surface::VolSurface;
use crate::surface::arbitrage::{SurfaceDiagnostics, surface_diagnostics};
use crate::surface::interp::{TenorPosition, locate_tenor, strike_grid};
use crate::types::{Strike, Tenor, Variance};
use crate::validate::{validate_positive, validate_positive_slice, validate_strictly_increasing};

/// Number of strikes used when sampling smiles for interpolation.
/// Log-spaced grid from 0.5·F to 2.0·F provides adequate density for
/// accurate spline construction while keeping memory usage reasonable.
const SMILE_GRID_SIZE: usize = 51;

/// Piecewise volatility surface composed of per-tenor smile sections.
///
/// Stores one [`SmileSection`] per tenor and interpolates linearly in total
/// variance space for cross-tenor queries.
///
/// # Construction
///
/// ```no_run
/// use volsurf::smile::SviSmile;
/// use volsurf::smile::SmileSection;
/// use volsurf::surface::PiecewiseSurface;
///
/// // Each tenor has its own calibrated smile
/// // let smile_3m: Box<dyn SmileSection> = Box::new(svi_3m);
/// // let smile_1y: Box<dyn SmileSection> = Box::new(svi_1y);
/// // let surface = PiecewiseSurface::new(
/// //     vec![0.25, 1.0],
/// //     vec![smile_3m, smile_1y],
/// // ).unwrap();
/// ```
///
/// # Serialization
///
/// This type does **not** implement `Serialize`/`Deserialize` because it
/// stores `dyn SmileSection` trait objects. If you need to persist a
/// calibrated surface, use [`SsviSurface`](super::SsviSurface) or
/// [`EssviSurface`](super::EssviSurface) instead.
pub struct PiecewiseSurface {
    /// Sorted tenors (time to expiry in years).
    tenors: Vec<f64>,
    /// One smile section per tenor, shared so `smile_at` can hand back the
    /// stored object rather than a resampled copy of it.
    smiles: Vec<Arc<dyn SmileSection>>,
}

impl fmt::Debug for PiecewiseSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PiecewiseSurface")
            .field("tenors", &self.tenors)
            .field("smiles", &self.smiles)
            .finish()
    }
}

impl PiecewiseSurface {
    /// Create a piecewise surface from a set of calibrated smiles.
    ///
    /// # Arguments
    /// * `tenors` — Strictly increasing positive tenors (years)
    /// * `smiles` — One calibrated [`SmileSection`] per tenor, each with an
    ///   `expiry()` matching its paired tenor
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if lengths mismatch, tenors
    /// are empty, not strictly increasing, or not positive, or if any smile's
    /// `expiry()` disagrees with the tenor it is paired with.
    pub fn new(tenors: Vec<f64>, smiles: Vec<Box<dyn SmileSection>>) -> error::Result<Self> {
        if tenors.len() != smiles.len() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors and smiles must have the same length, got {} and {}",
                    tenors.len(),
                    smiles.len()
                ),
            });
        }
        if tenors.is_empty() {
            return Err(VolSurfError::InvalidInput {
                message: "at least one tenor is required".into(),
            });
        }
        validate_positive_slice(&tenors, "tenors")?;
        validate_strictly_increasing(&tenors, "tenors")?;
        // Queries locate smiles by the tenor grid, so a smile calibrated to a
        // different expiry would be evaluated at the wrong maturity.
        for (i, (&tenor, smile)) in tenors.iter().zip(&smiles).enumerate() {
            let expiry = smile.expiry();
            if !expiry.is_finite() || (expiry - tenor).abs() >= EXPIRY_MATCH_TOL {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "smiles[{i}] has expiry {expiry} but is paired with tenor {tenor}"
                    ),
                });
            }
        }

        let smiles = smiles.into_iter().map(Arc::from).collect();
        Ok(Self { tenors, smiles })
    }

    /// The forward at `expiry`: stored value on a tenor match, log-linear
    /// between tenors, held flat outside the grid.
    fn forward_at(&self, expiry: f64) -> f64 {
        match self.locate_tenor(expiry) {
            TenorPosition::Exact(i) => self.smiles[i].forward(),
            TenorPosition::Before => self.smiles[0].forward(),
            TenorPosition::After => self.smiles[self.smiles.len() - 1].forward(),
            TenorPosition::Between(i, j) => {
                let alpha = (expiry - self.tenors[i]) / (self.tenors[j] - self.tenors[i]);
                let f1 = self.smiles[i].forward();
                let f2 = self.smiles[j].forward();
                (f1.ln() * (1.0 - alpha) + f2.ln() * alpha).exp()
            }
        }
    }

    fn locate_tenor(&self, expiry: f64) -> TenorPosition {
        locate_tenor(&self.tenors, expiry)
    }

    /// The two `diagnostics` entry points differ only in how each smile is
    /// asked to report itself; the forwards and the variance closure are shared.
    fn diagnostics_via<R>(&self, report_at: R) -> error::Result<SurfaceDiagnostics>
    where
        R: Fn(usize) -> error::Result<crate::smile::ArbitrageReport>,
    {
        let forwards: Vec<f64> = self.smiles.iter().map(|smile| smile.forward()).collect();
        surface_diagnostics(&self.tenors, &forwards, report_at, |i, strike| {
            self.smiles[i]
                .variance(Strike(strike))
                .map(|variance| variance.0)
        })
    }
}

/// A stored smile handed out by [`PiecewiseSurface::smile_at`].
///
/// `smile_at` must return an owned `Box<dyn SmileSection>`, and a trait object
/// cannot be cloned — so sharing the original by reference count is what lets
/// an exact tenor match return the caller's own calibrated model instead of a
/// spline approximation of it. Every method forwards.
#[derive(Debug)]
struct SharedSmile(Arc<dyn SmileSection>);

impl SmileSection for SharedSmile {
    fn vol(&self, strike: Strike) -> error::Result<crate::types::Vol> {
        self.0.vol(strike)
    }

    fn variance(&self, strike: Strike) -> error::Result<Variance> {
        self.0.variance(strike)
    }

    fn density(&self, strike: Strike) -> error::Result<f64> {
        self.0.density(strike)
    }

    fn forward(&self) -> f64 {
        self.0.forward()
    }

    fn expiry(&self) -> f64 {
        self.0.expiry()
    }

    fn model_name(&self) -> &'static str {
        self.0.model_name()
    }

    fn is_arbitrage_free(&self) -> error::Result<crate::smile::ArbitrageReport> {
        self.0.is_arbitrage_free()
    }

    fn is_arbitrage_free_with(
        &self,
        config: ArbitrageScanConfig,
    ) -> error::Result<crate::smile::ArbitrageReport> {
        self.0.is_arbitrage_free_with(config)
    }
}

impl VolSurface for PiecewiseSurface {
    fn black_variance(&self, expiry: Tenor, strike: Strike) -> error::Result<Variance> {
        validate_positive(expiry.0, "expiry")?;
        validate_positive(strike.0, "strike")?;

        match self.locate_tenor(expiry.0) {
            TenorPosition::Exact(i) => self.smiles[i].variance(strike),

            TenorPosition::Before => {
                // Flat vol extrapolation: w(T, K) = w(T1, K) · T/T1
                let w1 = self.smiles[0].variance(strike)?;
                Ok(Variance(w1.0 * expiry.0 / self.tenors[0]))
            }

            TenorPosition::After => {
                // Flat vol extrapolation: w(T, K) = w(Tn, K) · T/Tn
                let n = self.tenors.len();
                let wn = self.smiles[n - 1].variance(strike)?;
                Ok(Variance(wn.0 * expiry.0 / self.tenors[n - 1]))
            }

            TenorPosition::Between(i, j) => {
                let t1 = self.tenors[i];
                let t2 = self.tenors[j];
                let alpha = (expiry.0 - t1) / (t2 - t1);
                let w1 = self.smiles[i].variance(strike)?;
                let w2 = self.smiles[j].variance(strike)?;
                Ok(Variance((1.0 - alpha) * w1.0 + alpha * w2.0))
            }
        }
    }

    fn forward(&self, expiry: Tenor) -> error::Result<f64> {
        validate_positive(expiry.0, "expiry")?;
        Ok(self.forward_at(expiry.0))
    }

    /// On an exact tenor match this returns the stored smile itself. Off-grid
    /// expiries have no stored object, so they are resampled onto a cubic
    /// spline over the log-spaced grid `[0.5·F, 2·F]`; outside that range the
    /// spline flat-extrapolates, so prefer [`black_variance`](VolSurface::black_variance)
    /// for deep-wing queries at interpolated tenors.
    fn smile_at(&self, expiry: Tenor) -> error::Result<Box<dyn SmileSection>> {
        validate_positive(expiry.0, "expiry")?;

        // Resampling a stored smile would swap its model identity, analytic
        // density, and wing behaviour for the spline's, leaving `smile_at(T)`
        // and `black_variance(T, ·)` disagreeing on the same surface.
        if let TenorPosition::Exact(i) = self.locate_tenor(expiry.0) {
            return Ok(Box::new(SharedSmile(Arc::clone(&self.smiles[i]))));
        }

        let forward = self.forward_at(expiry.0);
        let strikes = strike_grid(forward, SMILE_GRID_SIZE);
        let variances = strikes
            .iter()
            .map(|&k| self.black_variance(expiry, Strike(k)).map(|w| w.0))
            .collect::<error::Result<Vec<f64>>>()?;

        Ok(Box::new(SplineSmile::new(
            forward, expiry.0, strikes, variances,
        )?))
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        self.diagnostics_via(|i| self.smiles[i].is_arbitrage_free())
    }

    fn diagnostics_with(&self, config: ArbitrageScanConfig) -> error::Result<SurfaceDiagnostics> {
        self.diagnostics_via(|i| self.smiles[i].is_arbitrage_free_with(config))
    }

    fn tenors(&self) -> &[f64] {
        &self.tenors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::SviSmile;
    use crate::smile::spline::SplineSmile;
    use crate::types::{Strike, Tenor, Vol};
    use approx::assert_abs_diff_eq;

    /// Test-only smile with an arbitrary `expiry()`. No validated constructor
    /// yields a non-finite expiry, but `SmileSection` is a public trait.
    #[derive(Debug)]
    struct FixedExpirySmile(f64);

    impl SmileSection for FixedExpirySmile {
        fn vol(&self, _strike: Strike) -> error::Result<Vol> {
            Ok(Vol(0.20))
        }
        fn forward(&self) -> f64 {
            100.0
        }
        fn expiry(&self) -> f64 {
            self.0
        }
        fn model_name(&self) -> &'static str {
            "FixedExpiry"
        }
    }

    /// Helper: create a flat-vol SplineSmile at a given tenor.
    fn flat_smile(forward: f64, expiry: f64, vol: f64) -> Box<dyn SmileSection> {
        let w = vol * vol * expiry;
        let strikes = vec![
            forward * 0.5,
            forward * 0.75,
            forward,
            forward * 1.25,
            forward * 1.5,
        ];
        let variances = vec![w; 5];
        Box::new(SplineSmile::new(forward, expiry, strikes, variances).unwrap())
    }

    #[expect(dead_code)]
    fn u_shaped_smile(forward: f64, expiry: f64, atm_vol: f64, skew: f64) -> Box<dyn SmileSection> {
        let strikes = vec![
            forward * 0.7,
            forward * 0.85,
            forward,
            forward * 1.15,
            forward * 1.3,
        ];
        let variances: Vec<f64> = strikes
            .iter()
            .map(|&k| {
                let m = ((k / forward).ln()).abs();
                let v = atm_vol + skew * m;
                v * v * expiry
            })
            .collect();
        Box::new(SplineSmile::new(forward, expiry, strikes, variances).unwrap())
    }

    #[test]
    fn rejects_empty_tenors() {
        let result = PiecewiseSurface::new(vec![], vec![]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![0.25, 0.5], vec![s1]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_unsorted_tenors() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![0.5, 0.25], vec![s1, s2]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_non_positive_tenor() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![0.0], vec![s1]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_smile_expiry_disagreeing_with_tenor() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(100.0, 2.0, 0.20);
        let err = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap_err();
        assert!(
            err.to_string().contains("smiles[1]"),
            "error should identify the mismatched pair: {err}"
        );
    }

    #[test]
    fn rejects_non_finite_smile_expiry() {
        for expiry in [f64::NAN, f64::INFINITY] {
            let result = PiecewiseSurface::new(vec![0.5], vec![Box::new(FixedExpirySmile(expiry))]);
            assert!(
                matches!(result, Err(VolSurfError::InvalidInput { .. })),
                "expiry {expiry} must not pair with tenor 0.5"
            );
        }
    }

    #[test]
    fn accepts_expiry_within_match_tolerance() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let result = PiecewiseSurface::new(vec![0.5 + EXPIRY_MATCH_TOL / 2.0], vec![s1]);
        assert!(result.is_ok(), "sub-tolerance drift is the same tenor");
    }

    #[test]
    fn rejects_duplicate_tenors() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![0.25, 0.25], vec![s1, s2]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn single_tenor_surface_constructs() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let surface = PiecewiseSurface::new(vec![0.25], vec![s1]);
        assert!(surface.is_ok());
    }

    #[test]
    fn exact_tenor_matches_stored_smile() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        // Query at T=0.25 should return 20% vol
        let vol = surface.black_vol(Tenor(0.25), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 1e-10);

        // Query at T=1.0 should return 25% vol
        let vol = surface.black_vol(Tenor(1.0), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(vol.0, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn midpoint_tenor_has_averaged_variance() {
        let vol1 = 0.20;
        let vol2 = 0.30;
        let t1 = 0.5;
        let t2 = 1.0;
        let s1 = flat_smile(100.0, t1, vol1);
        let s2 = flat_smile(100.0, t2, vol2);
        let surface = PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap();

        let t_mid = 0.75;
        let w1 = vol1 * vol1 * t1; // 0.02
        let w2 = vol2 * vol2 * t2; // 0.09
        let w_mid = 0.5 * w1 + 0.5 * w2; // 0.055

        let var = surface.black_variance(Tenor(t_mid), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(var.0, w_mid, epsilon = 1e-10);
    }

    #[test]
    fn black_vol_and_black_variance_are_consistent() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] {
            for k in [80.0, 100.0, 120.0] {
                let vol = surface.black_vol(Tenor(t), Strike(k)).unwrap();
                let var = surface.black_variance(Tenor(t), Strike(k)).unwrap();
                assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn extrapolation_before_first_tenor_uses_flat_vol() {
        let vol = 0.20;
        let t1 = 0.5;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // At T=0.25 (before first tenor), flat vol extrapolation:
        // w(0.25, K) = w(0.5, K) * 0.25/0.5 = sigma^2 * 0.5 * 0.5 = sigma^2 * 0.25
        let query_t = 0.25;
        let v = surface.black_vol(Tenor(query_t), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    #[test]
    fn extrapolation_after_last_tenor_uses_flat_vol() {
        let vol = 0.20;
        let t1 = 1.0;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // At T=2.0 (after last tenor), flat vol extrapolation
        let v = surface.black_vol(Tenor(2.0), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    #[test]
    fn smile_at_exact_tenor_returns_queryable_section() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();

        let smile = surface.smile_at(Tenor(1.0)).unwrap();
        let vol = smile.vol(Strike(100.0)).unwrap();
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 1e-4);
        assert_abs_diff_eq!(smile.expiry(), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn smile_at_exact_tenor_preserves_the_stored_model() {
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.3, 0.0, 0.2).unwrap();
        let surface = PiecewiseSurface::new(vec![1.0], vec![Box::new(svi.clone())]).unwrap();

        let smile = surface.smile_at(Tenor(1.0)).unwrap();
        assert_eq!(smile.model_name(), "SVI");
        // Analytic SVI density, not a spline approximation of it.
        assert_abs_diff_eq!(
            smile.density(Strike(100.0)).unwrap(),
            svi.density(Strike(100.0)).unwrap(),
            epsilon = 1e-14
        );
    }

    #[test]
    fn smile_at_exact_tenor_agrees_with_black_variance_beyond_the_spline_grid() {
        // Strikes outside [0.5F, 2F], where a resampled spline would
        // flat-extrapolate while black_variance kept evaluating the model.
        let svi = SviSmile::new(100.0, 1.0, 0.04, 0.1, -0.3, 0.0, 0.2).unwrap();
        let surface = PiecewiseSurface::new(vec![1.0], vec![Box::new(svi)]).unwrap();

        let smile = surface.smile_at(Tenor(1.0)).unwrap();
        for &k in &[25.0, 40.0, 250.0, 400.0] {
            assert_abs_diff_eq!(
                smile.variance(Strike(k)).unwrap().0,
                surface.black_variance(Tenor(1.0), Strike(k)).unwrap().0,
                epsilon = 1e-14
            );
        }
    }

    /// The scan must come from the stored model, not from a spline resampling
    /// of it: this SVI's `g(k)` goes negative in the wings, and the analytic
    /// scan and a resampled spline disagree about that.
    #[test]
    fn smile_at_exact_tenor_delegates_the_arbitrage_scan() {
        let svi = SviSmile::new(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05).unwrap();
        let surface = PiecewiseSurface::new(vec![1.0], vec![Box::new(svi.clone())]).unwrap();
        let smile = surface.smile_at(Tenor(1.0)).unwrap();

        for cfg in [
            ArbitrageScanConfig::wide(),
            ArbitrageScanConfig {
                n_points: 61,
                k_min: -1.5,
                k_max: 1.5,
            },
        ] {
            let got = smile.is_arbitrage_free_with(cfg).unwrap();
            let want = svi.is_arbitrage_free_with(cfg).unwrap();
            assert!(!want.is_free(), "fixture should violate butterfly");
            assert_abs_diff_eq!(got.expiry, want.expiry, epsilon = 1e-14);
            assert_eq!(
                got.butterfly_violations.len(),
                want.butterfly_violations.len()
            );
            for (g, w) in got
                .butterfly_violations
                .iter()
                .zip(&want.butterfly_violations)
            {
                assert_abs_diff_eq!(g.strike, w.strike, epsilon = 1e-14);
                assert_abs_diff_eq!(g.density, w.density, epsilon = 1e-14);
            }
        }

        // The no-config method routes through the model's own default grid.
        let got = smile.is_arbitrage_free().unwrap();
        let want = svi.is_arbitrage_free().unwrap();
        assert_eq!(
            got.butterfly_violations.len(),
            want.butterfly_violations.len()
        );
    }

    #[test]
    fn forward_matches_the_smile_the_surface_would_return() {
        let s1 = flat_smile(90.0, 0.5, 0.22);
        let s2 = flat_smile(110.0, 1.0, 0.22);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        // Exact tenors, an interpolated one, and both extrapolation regimes.
        for &t in &[0.25, 0.5, 0.75, 1.0, 2.0] {
            assert_abs_diff_eq!(
                surface.forward(Tenor(t)).unwrap(),
                surface.smile_at(Tenor(t)).unwrap().forward(),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn calendar_violations_default_is_clean_for_a_sane_surface() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();
        assert!(surface.calendar_violations().unwrap().is_empty());
    }

    #[test]
    fn calendar_violations_default_detects_variance_decreasing_in_time() {
        // w(0.5) = 0.045 but w(1.0) = 0.01 — total variance falls with time.
        let s1 = flat_smile(100.0, 0.5, 0.30);
        let s2 = flat_smile(100.0, 1.0, 0.10);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let violations = surface.calendar_violations().unwrap();
        assert!(!violations.is_empty());
        for v in &violations {
            assert!(v.variance_long < v.variance_short);
            assert_abs_diff_eq!(v.tenor_short, 0.5, epsilon = 1e-14);
            assert_abs_diff_eq!(v.tenor_long, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn calendar_violations_is_reachable_through_dyn_vol_surface() {
        let s1 = flat_smile(100.0, 0.5, 0.30);
        let s2 = flat_smile(100.0, 1.0, 0.10);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let erased: &dyn VolSurface = &surface;
        assert_eq!(
            erased.calendar_violations().unwrap().len(),
            surface.diagnostics().unwrap().calendar_violations.len(),
            "the trait default should agree with the diagnostics scan"
        );
    }

    #[test]
    fn forward_rejects_non_positive_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        assert!(matches!(
            surface.forward(Tenor(0.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn smile_at_between_tenors_returns_interpolated() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.30);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let smile = surface.smile_at(Tenor(0.75)).unwrap();
        assert_abs_diff_eq!(smile.expiry(), 0.75, epsilon = 1e-14);

        // Check that the interpolated variance is between the two smiles
        let var = smile.variance(Strike(100.0)).unwrap();
        let w1 = 0.20 * 0.20 * 0.5;
        let w2 = 0.30 * 0.30 * 1.0;
        let w_expected = 0.5 * w1 + 0.5 * w2;
        assert_abs_diff_eq!(var.0, w_expected, epsilon = 1e-3);
    }

    #[test]
    fn smile_at_rejects_non_positive_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();

        assert!(matches!(
            surface.smile_at(Tenor(0.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            surface.smile_at(Tenor(-1.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn clean_surface_reports_no_violations() {
        // Increasing vol with tenor → no calendar violations
        let s1 = flat_smile(100.0, 0.25, 0.18);
        let s2 = flat_smile(100.0, 0.5, 0.20);
        let s3 = flat_smile(100.0, 1.0, 0.22);
        let surface = PiecewiseSurface::new(vec![0.25, 0.5, 1.0], vec![s1, s2, s3]).unwrap();

        let diag = surface.diagnostics().unwrap();
        assert!(
            diag.is_free(),
            "surface with increasing vol should be arb-free, but got {} calendar violations",
            diag.calendar_violations.len()
        );
    }

    #[test]
    fn inverted_surface_detects_calendar_violation() {
        // 1Y smile has LOWER variance than 6M → calendar violation
        let s1 = flat_smile(100.0, 0.5, 0.30); // w = 0.045
        let s2 = flat_smile(100.0, 1.0, 0.15); // w = 0.0225 < 0.045
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let diag = surface.diagnostics().unwrap();
        assert!(!diag.is_free(), "inverted surface should have violations");
        assert!(
            !diag.calendar_violations.is_empty(),
            "should have calendar violations"
        );
    }

    #[test]
    fn black_vol_rejects_zero_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        assert!(matches!(
            surface.black_vol(Tenor(0.0), Strike(100.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn black_variance_rejects_negative_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        assert!(matches!(
            surface.black_variance(Tenor(-0.5), Strike(100.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn debug_impl_does_not_panic() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        let debug_str = format!("{surface:?}");
        assert!(debug_str.contains("PiecewiseSurface"));
    }

    // Gap #10: Infinity tenor rejected

    #[test]
    fn rejects_infinity_tenor() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![f64::INFINITY], vec![s1]);
        assert!(
            matches!(result, Err(VolSurfError::InvalidInput { .. })),
            "Inf tenor should be rejected"
        );
    }

    #[test]
    fn rejects_negative_infinity_tenor() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let result = PiecewiseSurface::new(vec![f64::NEG_INFINITY], vec![s1]);
        assert!(
            matches!(result, Err(VolSurfError::InvalidInput { .. })),
            "-Inf tenor should be rejected"
        );
    }

    // Gap #11: Single-tenor unit tests

    #[test]
    fn single_tenor_extrapolation_before() {
        let vol = 0.25;
        let t1 = 1.0;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // Query before the only tenor — flat vol extrapolation
        let v = surface.black_vol(Tenor(0.1), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    #[test]
    fn single_tenor_extrapolation_after() {
        let vol = 0.25;
        let t1 = 0.5;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // Query after the only tenor — flat vol extrapolation
        let v = surface.black_vol(Tenor(2.0), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    #[test]
    fn single_tenor_smile_at_before() {
        let vol = 0.20;
        let t1 = 1.0;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        let smile = surface.smile_at(Tenor(0.5)).unwrap();
        assert_abs_diff_eq!(smile.expiry(), 0.5, epsilon = 1e-14);
        let v = smile.vol(Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-4);
    }

    #[test]
    fn single_tenor_smile_at_after() {
        let vol = 0.20;
        let t1 = 0.5;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        let smile = surface.smile_at(Tenor(2.0)).unwrap();
        assert_abs_diff_eq!(smile.expiry(), 2.0, epsilon = 1e-14);
        let v = smile.vol(Strike(100.0)).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-4);
    }

    // Gap #12: Tenor tolerance boundary

    #[test]
    fn near_exact_tenor_within_tolerance_matches() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        // Query at T = 0.25 + 1e-11 (within 1e-10 tolerance) — should match exactly
        let vol = surface
            .black_vol(Tenor(0.25 + 1e-11), Strike(100.0))
            .unwrap();
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn near_exact_tenor_outside_tolerance_interpolates() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        // Query at T = 0.25 + 1e-8 (outside 1e-10 tolerance) — should interpolate
        let vol = surface
            .black_vol(Tenor(0.25 + 1e-8), Strike(100.0))
            .unwrap();
        // This is very close to T=0.25, so vol should still be very close to 0.20
        // but technically goes through the interpolation path
        assert!(vol.0 > 0.0);
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 0.01);
    }

    // (continued from existing tests)

    #[test]
    fn three_tenor_surface_interpolates_correctly() {
        let s1 = flat_smile(100.0, 0.25, 0.18);
        let s2 = flat_smile(100.0, 0.5, 0.20);
        let s3 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 0.5, 1.0], vec![s1, s2, s3]).unwrap();

        // Between first and second tenor (T=0.375, alpha=0.5)
        let w1 = 0.18 * 0.18 * 0.25;
        let w2 = 0.20 * 0.20 * 0.5;
        let expected = 0.5 * w1 + 0.5 * w2;
        let var = surface.black_variance(Tenor(0.375), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(var.0, expected, epsilon = 1e-10);

        // Between second and third tenor (T=0.75, alpha=0.5)
        let w2b = 0.20 * 0.20 * 0.5;
        let w3 = 0.25 * 0.25 * 1.0;
        let expected2 = 0.5 * w2b + 0.5 * w3;
        let var2 = surface.black_variance(Tenor(0.75), Strike(100.0)).unwrap();
        assert_abs_diff_eq!(var2.0, expected2, epsilon = 1e-10);
    }

    #[test]
    fn smile_at_uses_log_linear_forward_interpolation() {
        // F1=100, F2=400 → geometric mean at alpha=0.5 is 200, not 250
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(400.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let smile = surface.smile_at(Tenor(0.75)).unwrap();
        let expected_fwd = (100.0_f64.ln() * 0.5 + 400.0_f64.ln() * 0.5).exp();
        assert_abs_diff_eq!(smile.forward(), expected_fwd, epsilon = 1e-10);
        assert_abs_diff_eq!(expected_fwd, 200.0, epsilon = 1e-10);
    }

    #[test]
    fn smile_at_and_black_variance_consistent_with_differing_forwards() {
        let s1 = flat_smile(90.0, 0.5, 0.22);
        let s2 = flat_smile(110.0, 1.0, 0.22);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let t = 0.75;
        let smile = surface.smile_at(Tenor(t)).unwrap();
        for &k in &[95.0, 100.0, 105.0] {
            let from_smile = smile.variance(Strike(k)).unwrap().0;
            let from_surface = surface.black_variance(Tenor(t), Strike(k)).unwrap().0;
            assert_abs_diff_eq!(from_smile, from_surface, epsilon = 1e-3);
        }
    }

    #[test]
    fn black_variance_rejects_invalid_strikes() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();

        for &bad_strike in &[0.0, -100.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    surface.black_variance(Tenor(0.5), Strike(bad_strike)),
                    Err(VolSurfError::InvalidInput { .. })
                ),
                "should reject strike={bad_strike}"
            );
        }
    }
}
