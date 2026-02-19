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
