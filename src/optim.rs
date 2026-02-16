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
#[allow(dead_code)]
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
    let mut simplex = [(x0, y0), (x0 + step_x, y0), (x0, y0 + step_y)];
    let mut f_vals = [
        objective(simplex[0].0, simplex[0].1),
        objective(simplex[1].0, simplex[1].1),
        objective(simplex[2].0, simplex[2].1),
    ];

    for _ in 0..config.max_iter {
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| {
            f_vals[a]
                .partial_cmp(&f_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
        let fr = objective(rx, ry);

        if fr < f_vals[1] && fr >= f_vals[0] {
            simplex[2] = (rx, ry);
            f_vals[2] = fr;
        } else if fr < f_vals[0] {
            // Expansion
            let ex = cx + 2.0 * (rx - cx);
            let ey = cy + 2.0 * (ry - cy);
            let fe = objective(ex, ey);
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
            let fh = objective(hx, hy);
            if fh < f_vals[2].min(fr) {
                simplex[2] = (hx, hy);
                f_vals[2] = fh;
            } else {
                // Shrink toward best vertex
                for j in 1..3 {
                    simplex[j].0 = simplex[0].0 + 0.5 * (simplex[j].0 - simplex[0].0);
                    simplex[j].1 = simplex[0].1 + 0.5 * (simplex[j].1 - simplex[0].1);
                    f_vals[j] = objective(simplex[j].0, simplex[j].1);
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
}
