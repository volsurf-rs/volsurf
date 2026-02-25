//! Internal optimization utilities for smile calibration.

/// Configuration for the 2D Nelder-Mead simplex optimizer.
pub(crate) struct NelderMeadConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence threshold on simplex diameter.
    pub diameter_tol: f64,
    /// Convergence threshold on objective value spread.
    pub fvalue_tol: f64,
}

/// Result of a 2D Nelder-Mead optimization.
pub(crate) struct NelderMeadResult {
    /// Optimal x coordinate.
    pub x: f64,
    /// Optimal y coordinate.
    pub y: f64,
    /// Objective value at the optimum.
    pub fval: f64,
}

/// Minimize `objective(x, y)` using the Nelder-Mead simplex method in 2D.
///
/// Starts from `(x0, y0)` with initial perturbations `(step_x, step_y)`
/// to form the initial simplex. Returns the best vertex found.
pub(crate) fn nelder_mead_2d<F>(
    objective: F,
    x0: f64,
    y0: f64,
    step_x: f64,
    step_y: f64,
    config: &NelderMeadConfig,
) -> NelderMeadResult
where
    F: Fn(f64, f64) -> f64,
{
    let eval = |x: f64, y: f64| -> f64 {
        let v = objective(x, y);
        if v.is_finite() { v } else { f64::MAX }
    };

    let mut simplex = [(x0, y0), (x0 + step_x, y0), (x0, y0 + step_y)];
    let mut f_vals = [
        eval(simplex[0].0, simplex[0].1),
        eval(simplex[1].0, simplex[1].1),
        eval(simplex[2].0, simplex[2].1),
    ];

    for _ in 0..config.max_iter {
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| f_vals[a].total_cmp(&f_vals[b]));
        simplex = [simplex[idx[0]], simplex[idx[1]], simplex[idx[2]]];
        f_vals = [f_vals[idx[0]], f_vals[idx[1]], f_vals[idx[2]]];

        let diameter = simplex
            .iter()
            .flat_map(|a| {
                simplex
                    .iter()
                    .map(move |b| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
            })
            .fold(0.0_f64, f64::max);
        let f_spread = f_vals[2] - f_vals[0];

        if diameter < config.diameter_tol || f_spread < config.fvalue_tol {
            break;
        }

        // Centroid of best two
        let cx = (simplex[0].0 + simplex[1].0) / 2.0;
        let cy = (simplex[0].1 + simplex[1].1) / 2.0;

        // Reflection
        let rx = cx + (cx - simplex[2].0);
        let ry = cy + (cy - simplex[2].1);
        let fr = eval(rx, ry);

        if fr < f_vals[1] && fr >= f_vals[0] {
            simplex[2] = (rx, ry);
            f_vals[2] = fr;
        } else if fr < f_vals[0] {
            // Expansion
            let ex = cx + 2.0 * (rx - cx);
            let ey = cy + 2.0 * (ry - cy);
            let fe = eval(ex, ey);
            if fe < fr {
                simplex[2] = (ex, ey);
                f_vals[2] = fe;
            } else {
                simplex[2] = (rx, ry);
                f_vals[2] = fr;
            }
        } else {
            // Contraction
            let (hx, hy) = if fr < f_vals[2] {
                (cx + 0.5 * (rx - cx), cy + 0.5 * (ry - cy))
            } else {
                (
                    cx + 0.5 * (simplex[2].0 - cx),
                    cy + 0.5 * (simplex[2].1 - cy),
                )
            };
            let fh = eval(hx, hy);
            if fh < f_vals[2].min(fr) {
                simplex[2] = (hx, hy);
                f_vals[2] = fh;
            } else {
                // Shrink toward best vertex
                for j in 1..3 {
                    simplex[j].0 = simplex[0].0 + 0.5 * (simplex[j].0 - simplex[0].0);
                    simplex[j].1 = simplex[0].1 + 0.5 * (simplex[j].1 - simplex[0].1);
                    f_vals[j] = eval(simplex[j].0, simplex[j].1);
                }
            }
        }
    }

    let best_idx = if f_vals[0] <= f_vals[1] && f_vals[0] <= f_vals[2] {
        0
    } else if f_vals[1] <= f_vals[2] {
        1
    } else {
        2
    };

    NelderMeadResult {
        x: simplex[best_idx].0,
        y: simplex[best_idx].1,
        fval: f_vals[best_idx],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> NelderMeadConfig {
        NelderMeadConfig {
            max_iter: 1000,
            diameter_tol: 1e-12,
            fvalue_tol: 1e-14,
        }
    }

    #[test]
    fn converges_on_sphere_function() {
        // f(x,y) = x^2 + y^2, minimum at (0,0) with fval=0
        let result = nelder_mead_2d(|x, y| x * x + y * y, 1.0, 1.0, 0.5, 0.5, &default_config());
        assert!((result.x).abs() < 1e-6, "x should be ~0, got {}", result.x);
        assert!((result.y).abs() < 1e-6, "y should be ~0, got {}", result.y);
        assert!(
            result.fval < 1e-12,
            "fval should be ~0, got {}",
            result.fval
        );
    }

    #[test]
    fn converges_on_rosenbrock() {
        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1) with fval=0
        let result = nelder_mead_2d(
            |x, y| (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2),
            -1.0,
            -1.0,
            0.5,
            0.5,
            &NelderMeadConfig {
                max_iter: 5000,
                diameter_tol: 1e-12,
                fvalue_tol: 1e-14,
            },
        );
        assert!(
            (result.x - 1.0).abs() < 1e-3,
            "x should be ~1, got {}",
            result.x
        );
        assert!(
            (result.y - 1.0).abs() < 1e-3,
            "y should be ~1, got {}",
            result.y
        );
    }

    #[test]
    fn converges_on_shifted_minimum() {
        // f(x,y) = (x-3)^2 + (y+2)^2, minimum at (3,-2)
        let result = nelder_mead_2d(
            |x, y| (x - 3.0).powi(2) + (y + 2.0).powi(2),
            0.0,
            0.0,
            1.0,
            1.0,
            &default_config(),
        );
        assert!(
            (result.x - 3.0).abs() < 1e-6,
            "x should be ~3, got {}",
            result.x
        );
        assert!(
            (result.y + 2.0).abs() < 1e-6,
            "y should be ~-2, got {}",
            result.y
        );
        assert!(result.fval < 1e-12);
    }

    #[test]
    fn respects_max_iter_limit() {
        // Rosenbrock with only 5 iterations — should NOT converge
        let result = nelder_mead_2d(
            |x, y| (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2),
            -1.0,
            -1.0,
            0.5,
            0.5,
            &NelderMeadConfig {
                max_iter: 5,
                diameter_tol: 1e-20,
                fvalue_tol: 1e-20,
            },
        );
        // After only 5 iterations from (-1,-1), should still be far from (1,1)
        assert!(
            (result.x - 1.0).abs() > 0.1 || (result.y - 1.0).abs() > 0.1,
            "should not converge in 5 iterations"
        );
    }

    #[test]
    fn diameter_convergence_terminates() {
        // Very large diameter_tol should stop quickly without panic
        let result = nelder_mead_2d(
            |x, y| x * x + y * y,
            5.0,
            5.0,
            2.0,
            2.0,
            &NelderMeadConfig {
                max_iter: 100000,
                diameter_tol: 100.0, // Immediate stop — initial diameter is small
                fvalue_tol: 1e-20,
            },
        );
        // Just verify it returns a result without panicking
        assert!(result.fval.is_finite(), "should return finite fval");
    }

    #[test]
    fn handles_small_step_gracefully() {
        // Very small step creates a nearly-degenerate initial simplex
        let result = nelder_mead_2d(
            |x, y| x * x + y * y,
            1.0,
            1.0,
            0.001,
            0.001,
            &default_config(),
        );
        // Should still converge, just slowly
        assert!(result.fval < 0.5, "should improve despite small steps");
    }

    #[test]
    fn already_at_minimum() {
        // Start exactly at the minimum
        let result = nelder_mead_2d(
            |x, y| x * x + y * y,
            0.0,
            0.0,
            0.01,
            0.01,
            &default_config(),
        );
        assert!(result.fval < 1e-4, "starting near minimum should converge");
    }

    #[test]
    fn avoids_nan_region() {
        // NaN for x < 0, valid quadratic for x >= 0. Minimum at (0, 0).
        let result = nelder_mead_2d(
            |x, y| {
                if x < 0.0 { f64::NAN } else { x * x + y * y }
            },
            1.0,
            1.0,
            0.5,
            0.5,
            &default_config(),
        );
        assert!(
            result.fval.is_finite(),
            "fval must be finite, got {}",
            result.fval
        );
        assert!(
            result.fval < 1.0,
            "should find minimum in valid region, got {}",
            result.fval
        );
    }

    #[test]
    fn all_nan_returns_max() {
        // f_spread = MAX - MAX = 0 -> converges immediately (no signal to exploit)
        let result = nelder_mead_2d(|_, _| f64::NAN, 1.0, 1.0, 0.5, 0.5, &default_config());
        assert_eq!(result.fval, f64::MAX, "all-NaN objective should yield MAX");
        assert!(result.x.is_finite(), "x must be finite");
        assert!(result.y.is_finite(), "y must be finite");
    }

    #[test]
    fn inf_objective_treated_as_worst() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            let result = nelder_mead_2d(
                |x, y| {
                    if x > 2.0 {
                        bad
                    } else {
                        (x - 1.0).powi(2) + y * y
                    }
                },
                1.0,
                0.0,
                0.5,
                0.5,
                &default_config(),
            );
            assert!(result.fval.is_finite(), "fval finite for {bad}");
            assert!(result.fval < 0.1, "should converge near (1,0) for {bad}");
        }
    }

    #[test]
    fn initial_simplex_partial_nan() {
        // Two of three initial vertices land in NaN territory (x < 0).
        // Simplex: (-1, 0) NaN, (1, 0) valid, (-1, 1) NaN.
        // Optimizer must sort the valid vertex to best and still converge.
        let result = nelder_mead_2d(
            |x, y| {
                if x < 0.0 {
                    f64::NAN
                } else {
                    (x - 1.0).powi(2) + y * y
                }
            },
            -1.0,
            0.0,
            2.0,
            1.0,
            &NelderMeadConfig {
                max_iter: 2000,
                diameter_tol: 1e-10,
                fvalue_tol: 1e-12,
            },
        );
        assert!(
            result.fval.is_finite(),
            "fval must be finite, got {}",
            result.fval
        );
        assert!(
            result.fval < 0.1,
            "should converge near (1,0), got fval={}",
            result.fval
        );
    }

    #[test]
    fn shrink_path_nan_eval() {
        // Narrow horizontal band: valid for |y| < 0.04, NaN outside.
        // Vertex at y=0.1 is NaN. Reflection to y=-0.1 is NaN. Inside
        // contraction to y=0.05 is also NaN (0.05 > 0.04), forcing shrink.
        // During shrink, midpoint j=2 lands at y=0.05 (NaN), exercising the
        // eval guard on the shrink re-evaluation path (line 113).
        let result = nelder_mead_2d(
            |x, y| {
                if y.abs() > 0.04 {
                    f64::NAN
                } else {
                    x * x + y * y
                }
            },
            0.0,
            0.0,
            0.1,
            0.1,
            &NelderMeadConfig {
                max_iter: 5000,
                diameter_tol: 1e-12,
                fvalue_tol: 1e-14,
            },
        );
        assert!(
            result.fval.is_finite(),
            "fval must be finite, got {}",
            result.fval
        );
        assert!(
            result.fval < 0.01,
            "should converge near origin, got {}",
            result.fval
        );
    }

    #[test]
    fn expansion_into_nan_falls_back_to_reflection() {
        // Objective: x^2 + y^2 with NaN wall at x < -3. The simplex geometry
        // is set up so the first reflection is better than best, triggering
        // expansion. The expansion point overshoots past x = -3 into NaN.
        // The optimizer should fall back to the reflection point.
        let result = nelder_mead_2d(
            |x, y| {
                if x < -3.0 { f64::NAN } else { x * x + y * y }
            },
            2.0,
            0.5,
            3.0,
            0.5,
            &default_config(),
        );
        assert!(
            result.fval.is_finite(),
            "fval must be finite, got {}",
            result.fval
        );
        assert!(
            result.fval < 0.01,
            "should converge near origin, got {}",
            result.fval
        );
    }

    #[test]
    fn f64_max_objective_passes_through() {
        // f64::MAX is finite, so the guard must NOT clamp it. The optimizer
        // should treat it as a very large (but valid) objective value.
        let result = nelder_mead_2d(
            |x, y| {
                if x > 5.0 {
                    f64::MAX
                } else {
                    (x - 1.0).powi(2) + y * y
                }
            },
            1.0,
            0.0,
            0.5,
            0.5,
            &default_config(),
        );
        assert!(result.fval.is_finite(), "fval must be finite");
        assert!(
            result.fval < 0.01,
            "should converge near (1,0), got {}",
            result.fval
        );
    }

    #[test]
    fn nan_ring_around_minimum() {
        // Valid bowl inside r < 1.5 from (3, 3), NaN ring for 1.5 <= r < 4,
        // valid but large beyond r >= 4. The optimizer starts outside the ring
        // and must navigate through NaN to reach the inner bowl.
        let result = nelder_mead_2d(
            |x, y| {
                let r2 = (x - 3.0).powi(2) + (y - 3.0).powi(2);
                let r = r2.sqrt();
                if r < 1.5 {
                    r2
                } else if r < 4.0 {
                    f64::NAN
                } else {
                    100.0 + r2
                }
            },
            7.0,
            7.0,
            1.0,
            1.0,
            &NelderMeadConfig {
                max_iter: 5000,
                diameter_tol: 1e-10,
                fvalue_tol: 1e-12,
            },
        );
        // The optimizer cannot cross the NaN ring, so it stays in the outer
        // valid region. The key property: it does not panic or return NaN.
        assert!(
            result.fval.is_finite(),
            "fval must be finite despite NaN ring, got {}",
            result.fval
        );
        assert!(result.x.is_finite() && result.y.is_finite());
    }
}
