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
    let mut simplex = [
        (x0, y0),
        (x0 + step_x, y0),
        (x0, y0 + step_y),
    ];
    let mut f_vals = [
        objective(simplex[0].0, simplex[0].1),
        objective(simplex[1].0, simplex[1].1),
        objective(simplex[2].0, simplex[2].1),
    ];

    for _ in 0..config.max_iter {
        // Sort by objective value
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| {
            f_vals[a]
                .partial_cmp(&f_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = [simplex[idx[0]], simplex[idx[1]], simplex[idx[2]]];
        f_vals = [f_vals[idx[0]], f_vals[idx[1]], f_vals[idx[2]]];

        // Check convergence
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

    // Return best vertex
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
