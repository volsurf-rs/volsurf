use crate::surface::EXPIRY_MATCH_TOL;

/// Where an expiry falls relative to a strictly increasing tenor grid.
pub(crate) enum TenorPosition {
    /// Matches `tenors[i]` within [`EXPIRY_MATCH_TOL`].
    Exact(usize),
    /// Before the first tenor.
    Before,
    /// After the last tenor.
    After,
    /// Strictly between `tenors[i]` and `tenors[j]`, with `j == i + 1`.
    Between(usize, usize),
}

/// Locate `expiry` on a strictly increasing, non-empty tenor grid.
pub(crate) fn locate_tenor(tenors: &[f64], expiry: f64) -> TenorPosition {
    for (i, &t) in tenors.iter().enumerate() {
        if (expiry - t).abs() < EXPIRY_MATCH_TOL {
            return TenorPosition::Exact(i);
        }
    }
    if expiry < tenors[0] {
        return TenorPosition::Before;
    }
    if expiry > tenors[tenors.len() - 1] {
        return TenorPosition::After;
    }
    let right = tenors.partition_point(|&t| t < expiry);
    TenorPosition::Between(right - 1, right)
}

/// Construct the standard log-spaced strike grid from 0.5·F to 2.0·F.
pub(crate) fn strike_grid(forward: f64, n: usize) -> Vec<f64> {
    let log_min = (0.5_f64).ln();
    let log_max = (2.0_f64).ln();
    let step = (log_max - log_min) / (n - 1) as f64;
    (0..n)
        .map(|i| forward * (log_min + step * i as f64).exp())
        .collect()
}

/// Interpolate `(θ, F)` at an arbitrary expiry from stored tenor grids.
///
/// - Exact matches (within 1e-10) return stored values directly.
/// - Before the first tenor: flat-vol extrapolation (θ scaled by T/T₀), nearest forward.
/// - After the last tenor: flat-vol extrapolation (θ scaled by T/T_n), nearest forward.
/// - Between tenors: linear θ interpolation, log-linear forward interpolation.
pub(crate) fn interpolate_theta_forward(
    tenors: &[f64],
    thetas: &[f64],
    forwards: &[f64],
    expiry: f64,
) -> (f64, f64) {
    debug_assert!(!tenors.is_empty(), "tenors must not be empty");
    debug_assert_eq!(tenors.len(), thetas.len());
    debug_assert_eq!(tenors.len(), forwards.len());
    let n = tenors.len();

    match locate_tenor(tenors, expiry) {
        TenorPosition::Exact(i) => (thetas[i], forwards[i]),
        TenorPosition::Before => (thetas[0] * expiry / tenors[0], forwards[0]),
        TenorPosition::After => (thetas[n - 1] * expiry / tenors[n - 1], forwards[n - 1]),
        TenorPosition::Between(left, right) => {
            let alpha = (expiry - tenors[left]) / (tenors[right] - tenors[left]);
            let theta = (1.0 - alpha) * thetas[left] + alpha * thetas[right];
            let forward =
                (forwards[left].ln() * (1.0 - alpha) + forwards[right].ln() * alpha).exp();
            (theta, forward)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{interpolate_theta_forward, strike_grid};
    use approx::assert_abs_diff_eq;

    fn multi_tenor() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let thetas = vec![0.04, 0.08, 0.16, 0.32];
        let forwards = vec![100.0, 102.0, 105.0, 110.0];
        (tenors, thetas, forwards)
    }

    #[test]
    fn strike_grid_has_expected_endpoints_and_atm_midpoint() {
        let grid = strike_grid(100.0, 3);
        assert_abs_diff_eq!(grid[0], 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grid[1], 100.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grid[2], 200.0, epsilon = 1e-12);
    }

    #[test]
    fn single_element_exact() {
        let (theta, fwd) = interpolate_theta_forward(&[0.5], &[0.08], &[100.0], 0.5);
        assert_abs_diff_eq!(theta, 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn single_element_before() {
        let (theta, fwd) = interpolate_theta_forward(&[1.0], &[0.16], &[105.0], 0.5);
        assert_abs_diff_eq!(theta, 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 105.0, epsilon = 1e-14);
    }

    #[test]
    fn single_element_after() {
        let (theta, fwd) = interpolate_theta_forward(&[0.5], &[0.08], &[100.0], 1.0);
        assert_abs_diff_eq!(theta, 0.16, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn multi_exact_match() {
        let (t, th, f) = multi_tenor();
        let (theta, fwd) = interpolate_theta_forward(&t, &th, &f, 1.0);
        assert_abs_diff_eq!(theta, 0.16, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 105.0, epsilon = 1e-14);
    }

    #[test]
    fn multi_extrapolate_left() {
        let (t, th, f) = multi_tenor();
        // T=0.1 < T₀=0.25 → θ = 0.04 * 0.1/0.25 = 0.016, F = 100.0
        let (theta, fwd) = interpolate_theta_forward(&t, &th, &f, 0.1);
        assert_abs_diff_eq!(theta, 0.016, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn multi_extrapolate_right() {
        let (t, th, f) = multi_tenor();
        // T=4.0 > Tₙ=2.0 → θ = 0.32 * 4.0/2.0 = 0.64, F = 110.0
        let (theta, fwd) = interpolate_theta_forward(&t, &th, &f, 4.0);
        assert_abs_diff_eq!(theta, 0.64, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 110.0, epsilon = 1e-14);
    }

    #[test]
    fn multi_interpolation_theta_linear() {
        let (t, th, f) = multi_tenor();
        // T=0.75 between T=0.5 and T=1.0 → α = (0.75-0.5)/(1.0-0.5) = 0.5
        // θ = 0.5*0.08 + 0.5*0.16 = 0.12
        let (theta, _) = interpolate_theta_forward(&t, &th, &f, 0.75);
        assert_abs_diff_eq!(theta, 0.12, epsilon = 1e-14);
    }

    #[test]
    fn multi_interpolation_forward_log_linear() {
        let (t, th, f) = multi_tenor();
        // T=0.75 between T=0.5 (F=102) and T=1.0 (F=105), α=0.5
        // F = exp(0.5*ln(102) + 0.5*ln(105)) = √(102*105)
        let (_, fwd) = interpolate_theta_forward(&t, &th, &f, 0.75);
        let expected = (102.0_f64 * 105.0).sqrt();
        assert_abs_diff_eq!(fwd, expected, epsilon = 1e-10);
    }

    #[test]
    fn exact_match_within_tolerance() {
        let (t, th, f) = multi_tenor();
        let (theta, fwd) = interpolate_theta_forward(&t, &th, &f, 0.5 + 5e-11);
        assert_abs_diff_eq!(theta, 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 102.0, epsilon = 1e-14);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "tenors must not be empty")]
    fn panics_on_empty_input() {
        interpolate_theta_forward(&[], &[], &[], 0.5);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn panics_on_mismatched_lengths() {
        interpolate_theta_forward(&[0.5, 1.0], &[0.08], &[100.0, 105.0], 0.75);
    }
}
