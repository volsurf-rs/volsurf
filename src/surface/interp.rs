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

    for (i, &t) in tenors.iter().enumerate() {
        if (expiry - t).abs() < 1e-10 {
            return (thetas[i], forwards[i]);
        }
    }

    if expiry < tenors[0] {
        let theta = thetas[0] * expiry / tenors[0];
        return (theta, forwards[0]);
    }

    if expiry > tenors[n - 1] {
        let theta = thetas[n - 1] * expiry / tenors[n - 1];
        return (theta, forwards[n - 1]);
    }

    let right = tenors.partition_point(|&t| t < expiry);
    let left = right - 1;
    let alpha = (expiry - tenors[left]) / (tenors[right] - tenors[left]);
    let theta = (1.0 - alpha) * thetas[left] + alpha * thetas[right];
    let forward = (forwards[left].ln() * (1.0 - alpha) + forwards[right].ln() * alpha).exp();
    (theta, forward)
}

#[cfg(test)]
mod tests {
    use super::interpolate_theta_forward;
    use approx::assert_abs_diff_eq;

    fn multi_tenor() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let tenors = vec![0.25, 0.5, 1.0, 2.0];
        let thetas = vec![0.04, 0.08, 0.16, 0.32];
        let forwards = vec![100.0, 102.0, 105.0, 110.0];
        (tenors, thetas, forwards)
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

    #[test]
    #[should_panic(expected = "tenors must not be empty")]
    fn panics_on_empty_input() {
        interpolate_theta_forward(&[], &[], &[], 0.5);
    }

    #[test]
    #[should_panic]
    fn panics_on_mismatched_lengths() {
        interpolate_theta_forward(&[0.5, 1.0], &[0.08], &[100.0, 105.0], 0.75);
    }
}
