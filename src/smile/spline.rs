//! Cubic spline interpolation of variance.
//!
//! A non-parametric smile model that fits a natural cubic spline through
//! observed (strike, total-variance) points. Useful when parametric models
//! (SVI, SABR) fail to capture exotic smile shapes or when constructing
//! intermediate-tenor smiles from interpolated variance data.
//!
//! # Algorithm
//!
//! The spline is solved via the Thomas algorithm (O(n) tridiagonal solver)
//! with natural boundary conditions (S''(x₀) = S''(xₙ₋₁) = 0).
//! Evaluation uses binary search + Horner form for O(log n) per query.
//! Flat extrapolation outside the knot range prevents divergence.
//!
//! # References
//! - Fengler, M.R. "Arbitrage-free Smoothing of the Implied Volatility Surface" (2009)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::arbitrage::{ArbitrageReport, ButterflyViolation};
use crate::smile::SmileSection;
use crate::types::{Variance, Vol};
use crate::validate::validate_positive;

/// Coefficients for one cubic polynomial interval.
///
/// On interval \[xᵢ, xᵢ₊₁\], the spline is:
/// `S(x) = a + b·(x - xᵢ) + c·(x - xᵢ)² + d·(x - xᵢ)³`
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplineCoeff {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

/// Natural cubic spline smile on total variance.
///
/// Interpolates through (strike, total-variance) knot points using a natural
/// cubic spline. Flat extrapolation outside the knot range.
///
/// # Construction
///
/// ```
/// use volsurf::smile::SplineSmile;
///
/// let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
/// let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
/// let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineSmile {
    forward: f64,
    expiry: f64,
    strikes: Vec<f64>,
    variances: Vec<f64>,
    coeffs: Vec<SplineCoeff>,
}

impl SplineSmile {
    /// Create a spline smile from strike-variance pairs.
    ///
    /// Solves the natural cubic spline tridiagonal system via the Thomas
    /// algorithm and stores per-interval coefficients for O(log n) evaluation.
    ///
    /// # Arguments
    /// * `forward` — Forward price at expiry (must be > 0)
    /// * `expiry` — Time to expiry in years (must be > 0)
    /// * `strikes` — Strictly increasing strike values (at least 3)
    /// * `variances` — Non-negative total variances σ²T at each strike
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if fewer than 3 data points
    /// are provided, strikes are not strictly increasing, variances are
    /// negative, or scalar inputs are invalid.
    pub fn new(
        forward: f64,
        expiry: f64,
        strikes: Vec<f64>,
        variances: Vec<f64>,
    ) -> error::Result<Self> {
        // --- Input validation ---
        validate_positive(forward, "forward")?;
        validate_positive(expiry, "expiry")?;
        if strikes.len() != variances.len() {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                "strikes and variances must have the same length, got {} and {}",
                strikes.len(),
                variances.len()
                ),
            });
        }
        if strikes.len() < 3 {
            return Err(VolSurfError::InvalidInput {
                message: "spline requires at least 3 data points".into(),
            });
        }
        for k in &strikes {
            if !k.is_finite() {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "strikes must be finite, got {k}"
                ),
                });
            }
        }
        for (i, w) in strikes.windows(2).enumerate() {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "strikes must be strictly increasing, but strikes[{}]={} >= strikes[{}]={}",
                    i,
                    w[0],
                    i + 1,
                    w[1]
                ),
                });
            }
        }
        for v in &variances {
            if *v < 0.0 || v.is_nan() {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "variances must be non-negative, got {v}"
                ),
                });
            }
            if !v.is_finite() {
                return Err(VolSurfError::InvalidInput {
                message: format!(
                    "variances must be finite, got {v}"
                ),
                });
            }
        }

        // --- Compute spline coefficients via Thomas algorithm ---
        let n = strikes.len();
        let coeffs = build_spline_coefficients(&strikes, &variances, n);

        Ok(Self {
            forward,
            expiry,
            strikes,
            variances,
            coeffs,
        })
    }

    /// Evaluate the spline to get total variance at a given strike.
    ///
    /// Uses flat extrapolation outside the knot range and Horner-form
    /// polynomial evaluation on interior intervals.
    fn eval_variance(&self, strike: f64) -> f64 {
        let n = self.strikes.len();
        // Flat extrapolation
        if strike <= self.strikes[0] {
            return self.variances[0];
        }
        if strike >= self.strikes[n - 1] {
            return self.variances[n - 1];
        }
        // Binary search for interval index
        let i = self.strikes.partition_point(|&x| x < strike) - 1;
        let dx = strike - self.strikes[i];
        let c = &self.coeffs[i];
        // Horner form: a + dx*(b + dx*(c + dx*d))
        c.a + dx * (c.b + dx * (c.c + dx * c.d))
    }
}

/// Solve the natural cubic spline tridiagonal system and return
/// per-interval coefficients.
///
/// Uses the Thomas algorithm: O(n) forward elimination + back substitution.
fn build_spline_coefficients(
    x: &[f64],
    y: &[f64],
    n: usize,
) -> Vec<SplineCoeff> {
    // Interval widths
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();

    // Solve for second-derivative coefficients c[0..n] with c[0]=c[n-1]=0
    // (natural boundary conditions).
    // Interior system is (n-2) equations for c[1..n-1].
    let mut c = vec![0.0; n];

    if n > 2 {
        let m = n - 2; // number of interior unknowns

        // Build tridiagonal system: sub-diagonal l, diagonal d, super-diagonal u, rhs r
        let mut diag = vec![0.0; m];
        let mut rhs = vec![0.0; m];

        for j in 0..m {
            let i = j + 1; // index into original arrays
            diag[j] = 2.0 * (h[i - 1] + h[i]);
            rhs[j] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        // Thomas algorithm: forward sweep
        // sub-diagonal: h[i] for i=1..m-1 (relative to the tridiagonal matrix)
        // super-diagonal: h[i] for i=0..m-2
        for j in 1..m {
            let w = h[j] / diag[j - 1];
            diag[j] -= w * h[j];
            rhs[j] -= w * rhs[j - 1];
        }

        // Back substitution
        c[m] = rhs[m - 1] / diag[m - 1];
        for j in (0..m - 1).rev() {
            let i = j + 1;
            c[i] = (rhs[j] - h[j + 1] * c[i + 1]) / diag[j];
        }
    }

    // Compute a, b, d from c and the data
    let mut coeffs = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let a_i = y[i];
        let d_i = (c[i + 1] - c[i]) / (3.0 * h[i]);
        let b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        coeffs.push(SplineCoeff {
            a: a_i,
            b: b_i,
            c: c[i],
            d: d_i,
        });
    }

    coeffs
}

impl SmileSection for SplineSmile {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        validate_positive(strike, "strike")?;
        let w = self.eval_variance(strike);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                "negative interpolated variance {w} at strike {strike}"
                ),
            });
        }
        Ok(Vol((w / self.expiry).sqrt()))
    }

    fn variance(&self, strike: f64) -> error::Result<Variance> {
        validate_positive(strike, "strike")?;
        let w = self.eval_variance(strike);
        if w < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                "negative interpolated variance {w} at strike {strike}"
                ),
            });
        }
        Ok(Variance(w))
    }

    fn forward(&self) -> f64 {
        self.forward
    }

    fn expiry(&self) -> f64 {
        self.expiry
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        // Number of grid points for density-based arbitrage scan.
        let n_samples = 200;
        let k_min = self.strikes[0];
        let k_max = self.strikes[self.strikes.len() - 1];
        let dk = (k_max - k_min) / (n_samples as f64);

        let mut violations = Vec::new();
        for i in 1..n_samples {
            let k = k_min + dk * (i as f64);
            let d = self.density(k)?;
            // Tolerance for negative density detection.
            if d < -1e-8 {
                violations.push(ButterflyViolation {
                    strike: k,
                    density: d,
                    magnitude: d.abs(),
                });
            }
        }

        Ok(ArbitrageReport {
            is_free: violations.is_empty(),
            butterfly_violations: violations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Flat 20% vol smile for validation tests.
    fn make_flat_smile() -> SplineSmile {
        SplineSmile::new(100.0, 1.0, vec![80.0, 100.0, 120.0], vec![0.04, 0.04, 0.04]).unwrap()
    }

    // --- Constructor validation tests ---

    #[test]
    fn rejects_fewer_than_3_points() {
        let result = SplineSmile::new(100.0, 0.25, vec![1.0, 2.0], vec![0.04, 0.04]);
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, 2.0, 3.0],
            vec![0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_unsorted_strikes() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![3.0, 1.0, 2.0],
            vec![0.04, 0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_duplicate_strikes() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, 2.0, 2.0, 3.0],
            vec![0.04, 0.04, 0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_negative_variance() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, 2.0, 3.0],
            vec![0.04, -0.01, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_nan_variance() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, 2.0, 3.0],
            vec![0.04, f64::NAN, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_nan_strike() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, f64::NAN, 3.0],
            vec![0.04, 0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_infinite_variance() {
        let result = SplineSmile::new(
            100.0,
            0.25,
            vec![1.0, 2.0, 3.0],
            vec![0.04, f64::INFINITY, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_zero_expiry() {
        let result = SplineSmile::new(
            100.0,
            0.0,
            vec![1.0, 2.0, 3.0],
            vec![0.04, 0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    #[test]
    fn rejects_negative_forward() {
        let result = SplineSmile::new(
            -100.0,
            0.25,
            vec![1.0, 2.0, 3.0],
            vec![0.04, 0.04, 0.04],
        );
        assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
    }

    // --- Interpolation tests ---

    #[test]
    fn interpolates_through_knot_points() {
        // Spline must pass through knot points exactly.
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let expiry = 1.0;
        let smile = SplineSmile::new(100.0, expiry, strikes.clone(), variances.clone())
            .unwrap();

        for (k, w) in strikes.iter().zip(variances.iter()) {
            let vol = smile.vol(*k).unwrap();
            let expected = (*w / expiry).sqrt();
            assert_abs_diff_eq!(vol.0, expected, epsilon = 1e-14);
        }
    }

    #[test]
    fn variance_passes_through_knot_points() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes.clone(), variances.clone())
            .unwrap();

        for (k, w) in strikes.iter().zip(variances.iter()) {
            let var = smile.variance(*k).unwrap();
            assert_abs_diff_eq!(var.0, *w, epsilon = 1e-14);
        }
    }

    #[test]
    fn flat_smile_returns_constant_vol() {
        // All variances equal → vol should be constant everywhere.
        let w = 0.04; // 20% vol for T=1
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![w; 5];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        let expected_vol = w.sqrt();
        for k in [75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0] {
            let vol = smile.vol(k).unwrap();
            assert_abs_diff_eq!(vol.0, expected_vol, epsilon = 1e-14);
        }
    }

    #[test]
    fn variance_override_is_consistent_with_vol() {
        // variance(K) should equal vol(K)^2 * expiry
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let expiry = 0.5;
        let smile = SplineSmile::new(100.0, expiry, strikes, variances).unwrap();

        for k in [82.0, 95.0, 105.0, 118.0] {
            let vol = smile.vol(k).unwrap();
            let var = smile.variance(k).unwrap();
            assert_abs_diff_eq!(var.0, vol.0 * vol.0 * expiry, epsilon = 1e-14);
        }
    }

    // --- Extrapolation tests ---

    #[test]
    fn extrapolation_below_range_returns_first_variance() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        let var_below = smile.variance(50.0).unwrap();
        assert_abs_diff_eq!(var_below.0, 0.065, epsilon = 1e-14);
    }

    #[test]
    fn extrapolation_above_range_returns_last_variance() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        let var_above = smile.variance(200.0).unwrap();
        assert_abs_diff_eq!(var_above.0, 0.065, epsilon = 1e-14);
    }

    #[test]
    fn extrapolation_no_nan_or_inf() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        for k in [0.01, 1.0, 50.0, 500.0, 10000.0] {
            let vol = smile.vol(k).unwrap();
            assert!(vol.0.is_finite(), "vol at K={k} should be finite");
        }
    }

    // --- Mid-interval accuracy: cubic recovery ---

    #[test]
    fn recovers_linear_function_exactly() {
        // Natural cubic spline through points of a linear function
        // should recover it exactly (all cubic/quadratic coefficients vanish).
        // f(x) = 0.02 + 0.0005 * x
        let f = |x: f64| 0.02 + 0.0005 * x;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances: Vec<f64> = strikes.iter().map(|&k| f(k)).collect();

        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        // Test at mid-interval points
        for k in [85.0, 95.0, 105.0, 115.0] {
            let expected = f(k);
            let actual = smile.eval_variance(k);
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-14);
        }
    }

    #[test]
    fn mid_interval_interpolation_is_accurate() {
        // Verify that mid-interval values are reasonable for a smooth smile.
        // Quadratic variance: w(K) = 0.04 + 1e-5 * (K-100)²
        let f = |k: f64| 0.04 + 1e-5 * (k - 100.0) * (k - 100.0);
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances: Vec<f64> = strikes.iter().map(|&k| f(k)).collect();

        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        // Mid-interval values should be close to the true quadratic
        for k in [85.0, 95.0, 105.0, 115.0] {
            let expected = f(k);
            let actual = smile.eval_variance(k);
            // Natural spline won't perfectly match quadratic at boundaries,
            // but interior should be very close
            assert_abs_diff_eq!(actual, expected, epsilon = 5e-4);
        }
    }

    // --- Density tests ---

    #[test]
    fn density_non_negative_for_convex_smile() {
        // A convex (U-shaped) variance smile should have non-negative density.
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        for k in [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0] {
            let d = smile.density(k).unwrap();
            assert!(
                d >= -1e-8,
                "density at K={k} should be non-negative, got {d}"
            );
        }
    }

    #[test]
    fn density_integrates_approximately_to_one() {
        // Numerical integration of density across strike range should ≈ 1.
        let strikes = vec![60.0, 80.0, 100.0, 120.0, 140.0];
        let variances = vec![0.08, 0.05, 0.04, 0.05, 0.08];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        let n = 1000;
        let k_lo = 20.0;
        let k_hi = 300.0;
        let dk = (k_hi - k_lo) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let k = k_lo + dk * (i as f64 + 0.5);
            integral += smile.density(k).unwrap() * dk;
        }
        // Tolerance is generous: finite integration range + numerical density
        assert_abs_diff_eq!(integral, 1.0, epsilon = 0.10);
    }

    // --- Arbitrage check ---

    #[test]
    fn convex_smile_is_arbitrage_free() {
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        let report = smile.is_arbitrage_free().unwrap();
        assert!(
            report.is_free,
            "convex smile should be arb-free, got {} violations",
            report.butterfly_violations.len()
        );
    }

    // --- Forward/expiry accessors ---

    #[test]
    fn forward_accessor() {
        let smile = SplineSmile::new(
            105.0,
            0.5,
            vec![80.0, 100.0, 120.0],
            vec![0.04, 0.04, 0.04],
        )
        .unwrap();
        assert_abs_diff_eq!(smile.forward(), 105.0, epsilon = 1e-14);
    }

    #[test]
    fn expiry_accessor() {
        let smile = SplineSmile::new(
            100.0,
            0.25,
            vec![80.0, 100.0, 120.0],
            vec![0.04, 0.04, 0.04],
        )
        .unwrap();
        assert_abs_diff_eq!(smile.expiry(), 0.25, epsilon = 1e-14);
    }

    // --- 3-point minimum ---

    #[test]
    fn three_points_works() {
        let smile = SplineSmile::new(
            100.0,
            1.0,
            vec![80.0, 100.0, 120.0],
            vec![0.06, 0.04, 0.06],
        );
        assert!(smile.is_ok());
    }

    // --- Zero variance at a knot ---

    #[test]
    fn zero_variance_produces_zero_vol() {
        let smile = SplineSmile::new(
            100.0,
            1.0,
            vec![80.0, 100.0, 120.0],
            vec![0.04, 0.0, 0.04],
        )
        .unwrap();
        let vol = smile.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.0, epsilon = 1e-14);
    }

    // --- Monotonicity of variance on monotone input ---

    #[test]
    fn spline_is_smooth() {
        // Verify that the spline doesn't have wild oscillations for smooth input.
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.065, 0.045, 0.04, 0.045, 0.065];
        let smile = SplineSmile::new(100.0, 1.0, strikes, variances).unwrap();

        // Check that vol changes smoothly between knots
        let mut prev_vol = smile.vol(80.0).unwrap().0;
        for i in 1..=40 {
            let k = 80.0 + i as f64;
            let vol = smile.vol(k).unwrap().0;
            let change = (vol - prev_vol).abs();
            assert!(
                change < 0.05,
                "vol change too large between K={} and K={}: {change}",
                k - 1.0,
                k
            );
            prev_vol = vol;
        }
    }

    // --- Strike validation in vol()/variance() ---

    #[test]
    fn vol_rejects_nan_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.vol(f64::NAN),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn vol_rejects_negative_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.vol(-1.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn vol_rejects_zero_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.vol(0.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn variance_rejects_nan_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.variance(f64::NAN),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn variance_rejects_zero_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.variance(0.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    // --- Gap #5: negative interpolated variance error path ---

    /// Construct a SplineSmile that will overshoot into negative variance,
    /// bypassing new()'s validation by building with borderline-valid data
    /// where the natural cubic oscillates below zero between knots.
    fn make_overshooting_spline() -> SplineSmile {
        // Alternating high-low pattern forces cubic to overshoot.
        // Knots at: [1, 2, 3, 4, 5] with variances [0.5, 0.0001, 0.5, 0.0001, 0.5]
        // The cubic between knot 2 and 3 must swing from 0.0001 to 0.5,
        // but with matching second derivatives from the surrounding steep
        // segments, the natural cubic oscillates into negatives.
        SplineSmile::new(
            100.0,
            1.0,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.5, 0.0001, 0.5, 0.0001, 0.5],
        )
        .unwrap()
    }

    #[test]
    fn vol_returns_error_for_negative_interpolated_variance() {
        let smile = make_overshooting_spline();
        // Scan between knots looking for the negative overshoot
        let mut found_negative = false;
        for i in 0..400 {
            let strike = 1.01 + i as f64 * 0.01;
            if let Err(VolSurfError::NumericalError { .. }) = smile.vol(strike) {
                found_negative = true;
                break;
            }
        }
        assert!(
            found_negative,
            "should find at least one strike where spline variance goes negative"
        );
    }

    #[test]
    fn variance_returns_error_for_negative_interpolated_variance() {
        let smile = make_overshooting_spline();
        let mut found_negative = false;
        for i in 0..400 {
            let strike = 1.01 + i as f64 * 0.01;
            if let Err(VolSurfError::NumericalError { .. }) = smile.variance(strike) {
                found_negative = true;
                break;
            }
        }
        assert!(
            found_negative,
            "should find at least one strike where spline variance goes negative"
        );
    }

    // --- Gap #16: Default density() error propagation ---
    //
    // SplineSmile uses the default SmileSection::density() implementation
    // (Breeden-Litzenberger via finite differences). When vol() fails,
    // the error should propagate through density().

    #[test]
    fn default_density_propagates_vol_error() {
        let smile = make_overshooting_spline();
        // Find a strike where vol() fails (negative variance)
        let mut error_strike = None;
        for i in 0..400 {
            let strike = 1.01 + i as f64 * 0.01;
            if smile.vol(strike).is_err() {
                error_strike = Some(strike);
                break;
            }
        }
        let strike = error_strike.expect("should find a failing strike");

        // density() should propagate the error from vol()
        let density_result = smile.density(strike);
        assert!(
            density_result.is_err(),
            "density() should propagate vol() error at K={strike}"
        );
    }

    #[test]
    fn default_density_rejects_zero_strike() {
        // Default density() validates strike > 0 before calling vol()
        let smile = make_flat_smile();
        assert!(matches!(
            smile.density(0.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    #[test]
    fn default_density_rejects_negative_strike() {
        let smile = make_flat_smile();
        assert!(matches!(
            smile.density(-10.0),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }
}
