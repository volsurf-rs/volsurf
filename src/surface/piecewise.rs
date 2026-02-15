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

use crate::error::{self, VolSurfError};
use crate::smile::spline::SplineSmile;
use crate::smile::SmileSection;
use crate::surface::arbitrage::{CalendarViolation, SurfaceDiagnostics};
use crate::surface::VolSurface;
use crate::types::{Variance, Vol};
use crate::validate::validate_positive;

/// Number of strikes used when sampling smiles for interpolation.
const SMILE_GRID_SIZE: usize = 51;

/// Number of strikes used for calendar arbitrage checks.
const CALENDAR_CHECK_GRID_SIZE: usize = 41;

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
            .field("smiles", &self.smiles)
            .finish()
    }
}

impl PiecewiseSurface {
    /// Create a piecewise surface from a set of calibrated smiles.
    ///
    /// # Arguments
    /// * `tenors` — Strictly increasing positive tenors (years)
    /// * `smiles` — One calibrated [`SmileSection`] per tenor
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if lengths mismatch, tenors
    /// are empty, not strictly increasing, or not positive.
    pub fn new(
        tenors: Vec<f64>,
        smiles: Vec<Box<dyn SmileSection>>,
    ) -> error::Result<Self> {
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
        for (i, t) in tenors.iter().enumerate() {
            if *t <= 0.0 || t.is_nan() {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors must be positive, got tenors[{i}]={t}"
                ),
                });
            }
        }
        for w in tenors.windows(2) {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "tenors must be strictly increasing, but {} >= {}",
                    w[0], w[1]
                ),
                });
            }
        }

        Ok(Self { tenors, smiles })
    }

    /// Generate a strike grid for sampling smiles, centered on the forward.
    ///
    /// Uses log-spaced strikes from `0.5·F` to `2.0·F`.
    fn strike_grid(forward: f64, n: usize) -> Vec<f64> {
        let ln_lo = (0.5_f64).ln();
        let ln_hi = (2.0_f64).ln();
        let step = (ln_hi - ln_lo) / (n - 1) as f64;
        (0..n)
            .map(|i| forward * (ln_lo + step * i as f64).exp())
            .collect()
    }

    /// Find the bracketing tenor indices for a given expiry.
    ///
    /// Returns `(TenorPosition, left_index)` where left_index is the
    /// index of the tenor <= expiry.
    fn locate_tenor(&self, expiry: f64) -> TenorPosition {
        let n = self.tenors.len();

        // Check for exact match (within tolerance)
        for (i, &t) in self.tenors.iter().enumerate() {
            if (expiry - t).abs() < 1e-10 {
                return TenorPosition::Exact(i);
            }
        }

        if expiry < self.tenors[0] {
            return TenorPosition::Before;
        }
        if expiry > self.tenors[n - 1] {
            return TenorPosition::After;
        }

        // Binary search for bracketing interval
        let right = self.tenors.partition_point(|&t| t < expiry);
        TenorPosition::Between(right - 1, right)
    }
}

enum TenorPosition {
    /// Exactly matches tenor at index i.
    Exact(usize),
    /// Before the first tenor.
    Before,
    /// After the last tenor.
    After,
    /// Between tenors[i] and tenors[j].
    Between(usize, usize),
}

impl VolSurface for PiecewiseSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        validate_positive(expiry, "expiry")?;
        let var = self.black_variance(expiry, strike)?;
        Ok(Vol((var.0 / expiry).sqrt()))
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> error::Result<Variance> {
        validate_positive(expiry, "expiry")?;

        match self.locate_tenor(expiry) {
            TenorPosition::Exact(i) => self.smiles[i].variance(strike),

            TenorPosition::Before => {
                // Flat vol extrapolation: w(T, K) = w(T1, K) · T/T1
                let w1 = self.smiles[0].variance(strike)?;
                Ok(Variance(w1.0 * expiry / self.tenors[0]))
            }

            TenorPosition::After => {
                // Flat vol extrapolation: w(T, K) = w(Tn, K) · T/Tn
                let n = self.tenors.len();
                let wn = self.smiles[n - 1].variance(strike)?;
                Ok(Variance(wn.0 * expiry / self.tenors[n - 1]))
            }

            TenorPosition::Between(i, j) => {
                let t1 = self.tenors[i];
                let t2 = self.tenors[j];
                let alpha = (expiry - t1) / (t2 - t1);
                let w1 = self.smiles[i].variance(strike)?;
                let w2 = self.smiles[j].variance(strike)?;
                Ok(Variance((1.0 - alpha) * w1.0 + alpha * w2.0))
            }
        }
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        validate_positive(expiry, "expiry")?;

        // Determine the forward and strike grid for the interpolated smile
        let (forward, strikes, variances) = match self.locate_tenor(expiry) {
            TenorPosition::Exact(i) => {
                let fwd = self.smiles[i].forward();
                let grid = Self::strike_grid(fwd, SMILE_GRID_SIZE);
                let vars: error::Result<Vec<f64>> = grid
                    .iter()
                    .map(|&k| self.smiles[i].variance(k).map(|v| v.0))
                    .collect();
                (fwd, grid, vars?)
            }

            TenorPosition::Before => {
                let fwd = self.smiles[0].forward();
                let grid = Self::strike_grid(fwd, SMILE_GRID_SIZE);
                let t1 = self.tenors[0];
                let scale = expiry / t1;
                let vars: error::Result<Vec<f64>> = grid
                    .iter()
                    .map(|&k| self.smiles[0].variance(k).map(|v| v.0 * scale))
                    .collect();
                (fwd, grid, vars?)
            }

            TenorPosition::After => {
                let n = self.tenors.len();
                let fwd = self.smiles[n - 1].forward();
                let grid = Self::strike_grid(fwd, SMILE_GRID_SIZE);
                let tn = self.tenors[n - 1];
                let scale = expiry / tn;
                let vars: error::Result<Vec<f64>> = grid
                    .iter()
                    .map(|&k| self.smiles[n - 1].variance(k).map(|v| v.0 * scale))
                    .collect();
                (fwd, grid, vars?)
            }

            TenorPosition::Between(i, j) => {
                let f1 = self.smiles[i].forward();
                let f2 = self.smiles[j].forward();
                let t1 = self.tenors[i];
                let t2 = self.tenors[j];
                let alpha = (expiry - t1) / (t2 - t1);
                let fwd = (1.0 - alpha) * f1 + alpha * f2;
                let grid = Self::strike_grid(fwd, SMILE_GRID_SIZE);
                let vars: error::Result<Vec<f64>> = grid
                    .iter()
                    .map(|&k| {
                        let w1 = self.smiles[i].variance(k)?.0;
                        let w2 = self.smiles[j].variance(k)?.0;
                        Ok((1.0 - alpha) * w1 + alpha * w2)
                    })
                    .collect();
                (fwd, grid, vars?)
            }
        };

        let spline = SplineSmile::new(forward, expiry, strikes, variances)?;
        Ok(Box::new(spline))
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        // Collect per-tenor butterfly reports
        let mut smile_reports = Vec::with_capacity(self.smiles.len());
        for smile in &self.smiles {
            smile_reports.push(smile.is_arbitrage_free()?);
        }

        // Calendar spread checks between consecutive tenors
        let mut calendar_violations = Vec::new();
        for i in 0..self.tenors.len().saturating_sub(1) {
            let f1 = self.smiles[i].forward();
            let f2 = self.smiles[i + 1].forward();
            let fwd_avg = 0.5 * (f1 + f2);
            let grid = Self::strike_grid(fwd_avg, CALENDAR_CHECK_GRID_SIZE);

            for &k in &grid {
                let w_short = self.smiles[i].variance(k)?;
                let w_long = self.smiles[i + 1].variance(k)?;
                if w_long.0 < w_short.0 - 1e-10 {
                    calendar_violations.push(CalendarViolation {
                        strike: k,
                        tenor_short: self.tenors[i],
                        tenor_long: self.tenors[i + 1],
                        variance_short: w_short.0,
                        variance_long: w_long.0,
                    });
                }
            }
        }

        let is_free =
            smile_reports.iter().all(|r| r.is_free) && calendar_violations.is_empty();

        Ok(SurfaceDiagnostics {
            smile_reports,
            calendar_violations,
            is_free,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::spline::SplineSmile;
    use approx::assert_abs_diff_eq;

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

    /// Helper: create a U-shaped smile at a given tenor.
    #[allow(dead_code)]
    fn u_shaped_smile(
        forward: f64,
        expiry: f64,
        atm_vol: f64,
        skew: f64,
    ) -> Box<dyn SmileSection> {
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

    // --- Constructor validation ---

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

    // --- Exact tenor query ---

    #[test]
    fn exact_tenor_matches_stored_smile() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        // Query at T=0.25 should return 20% vol
        let vol = surface.black_vol(0.25, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 1e-10);

        // Query at T=1.0 should return 25% vol
        let vol = surface.black_vol(1.0, 100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.25, epsilon = 1e-10);
    }

    // --- Midpoint tenor interpolation ---

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

        let var = surface.black_variance(t_mid, 100.0).unwrap();
        assert_abs_diff_eq!(var.0, w_mid, epsilon = 1e-10);
    }

    // --- Vol and variance consistency ---

    #[test]
    fn black_vol_and_black_variance_are_consistent() {
        let s1 = flat_smile(100.0, 0.25, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.25);
        let surface = PiecewiseSurface::new(vec![0.25, 1.0], vec![s1, s2]).unwrap();

        for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] {
            for k in [80.0, 100.0, 120.0] {
                let vol = surface.black_vol(t, k).unwrap();
                let var = surface.black_variance(t, k).unwrap();
                assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-12);
            }
        }
    }

    // --- Extrapolation ---

    #[test]
    fn extrapolation_before_first_tenor_uses_flat_vol() {
        let vol = 0.20;
        let t1 = 0.5;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // At T=0.25 (before first tenor), flat vol extrapolation:
        // w(0.25, K) = w(0.5, K) * 0.25/0.5 = sigma^2 * 0.5 * 0.5 = sigma^2 * 0.25
        let query_t = 0.25;
        let v = surface.black_vol(query_t, 100.0).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    #[test]
    fn extrapolation_after_last_tenor_uses_flat_vol() {
        let vol = 0.20;
        let t1 = 1.0;
        let s1 = flat_smile(100.0, t1, vol);
        let surface = PiecewiseSurface::new(vec![t1], vec![s1]).unwrap();

        // At T=2.0 (after last tenor), flat vol extrapolation
        let v = surface.black_vol(2.0, 100.0).unwrap();
        assert_abs_diff_eq!(v.0, vol, epsilon = 1e-10);
    }

    // --- smile_at() ---

    #[test]
    fn smile_at_exact_tenor_returns_queryable_section() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();

        let smile = surface.smile_at(1.0).unwrap();
        let vol = smile.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.20, epsilon = 1e-4);
        assert_abs_diff_eq!(smile.expiry(), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn smile_at_between_tenors_returns_interpolated() {
        let s1 = flat_smile(100.0, 0.5, 0.20);
        let s2 = flat_smile(100.0, 1.0, 0.30);
        let surface = PiecewiseSurface::new(vec![0.5, 1.0], vec![s1, s2]).unwrap();

        let smile = surface.smile_at(0.75).unwrap();
        assert_abs_diff_eq!(smile.expiry(), 0.75, epsilon = 1e-14);

        // Check that the interpolated variance is between the two smiles
        let var = smile.variance(100.0).unwrap();
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
            surface.smile_at(0.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            surface.smile_at(-1.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    // --- Diagnostics ---

    #[test]
    fn clean_surface_reports_no_violations() {
        // Increasing vol with tenor → no calendar violations
        let s1 = flat_smile(100.0, 0.25, 0.18);
        let s2 = flat_smile(100.0, 0.5, 0.20);
        let s3 = flat_smile(100.0, 1.0, 0.22);
        let surface =
            PiecewiseSurface::new(vec![0.25, 0.5, 1.0], vec![s1, s2, s3]).unwrap();

        let diag = surface.diagnostics().unwrap();
        assert!(
            diag.is_free,
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
        assert!(!diag.is_free, "inverted surface should have violations");
        assert!(
            !diag.calendar_violations.is_empty(),
            "should have calendar violations"
        );
    }

    // --- Error handling ---

    #[test]
    fn black_vol_rejects_zero_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        assert!(matches!(
            surface.black_vol(0.0, 100.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn black_variance_rejects_negative_expiry() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        assert!(matches!(
            surface.black_variance(-0.5, 100.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    // --- Debug impl ---

    #[test]
    fn debug_impl_does_not_panic() {
        let s1 = flat_smile(100.0, 1.0, 0.20);
        let surface = PiecewiseSurface::new(vec![1.0], vec![s1]).unwrap();
        let debug_str = format!("{surface:?}");
        assert!(debug_str.contains("PiecewiseSurface"));
    }

    // --- Multi-tenor interpolation ---

    #[test]
    fn three_tenor_surface_interpolates_correctly() {
        let s1 = flat_smile(100.0, 0.25, 0.18);
        let s2 = flat_smile(100.0, 0.5, 0.20);
        let s3 = flat_smile(100.0, 1.0, 0.25);
        let surface =
            PiecewiseSurface::new(vec![0.25, 0.5, 1.0], vec![s1, s2, s3]).unwrap();

        // Between first and second tenor (T=0.375, alpha=0.5)
        let w1 = 0.18 * 0.18 * 0.25;
        let w2 = 0.20 * 0.20 * 0.5;
        let expected = 0.5 * w1 + 0.5 * w2;
        let var = surface.black_variance(0.375, 100.0).unwrap();
        assert_abs_diff_eq!(var.0, expected, epsilon = 1e-10);

        // Between second and third tenor (T=0.75, alpha=0.5)
        let w2b = 0.20 * 0.20 * 0.5;
        let w3 = 0.25 * 0.25 * 1.0;
        let expected2 = 0.5 * w2b + 0.5 * w3;
        let var2 = surface.black_variance(0.75, 100.0).unwrap();
        assert_abs_diff_eq!(var2.0, expected2, epsilon = 1e-10);
    }
}
