//! Calibration scaffolding shared by the SSVI and eSSVI surfaces.
//!
//! Both surfaces run the same outer procedure — validate the tenor/forward
//! grid, fit per-tenor SVI, check that the resulting θ are monotone, then
//! optimize (η, γ) by grid search followed by Nelder-Mead. Only the objective
//! and the model tag differ, so everything around the objective lives here.
//!
//! The query paths are deliberately *not* shared: eSSVI threads a
//! maturity-dependent ρ(θ) through every evaluation, and unifying that would
//! put a virtual call in `black_variance`.

use crate::error::{self, VolSurfError};
use crate::validate::validate_positive_slice;

/// Points per axis in the 2-D calibration grid searches.
pub(crate) const GRID_N: usize = 15;

/// Bound keeping |ρ| strictly inside 1.
///
/// The SSVI radical `√((φk + ρ)² + 1 − ρ²)` collapses to `|φk + ρ|` at
/// |ρ| = 1, which zeroes `w''` and makes the density degenerate, so neither
/// calibration nor `ρ(θ)` may return an endpoint.
pub(crate) const RHO_CLAMP: f64 = 0.999;

/// Below this |ρₘ − ρ₀| the eSSVI correlation term structure counts as flat.
///
/// The exponent `a` in `ρ(θ) = ρ₀ + (ρₘ − ρ₀)·(θ/θ_max)^a` is then
/// unidentifiable and the Eq. 5.7 bound, which divides by that difference,
/// does not apply.
pub(crate) const RHO_FLAT_EPS: f64 = 1e-14;

/// Validate the per-tenor calibration inputs common to both surfaces.
///
/// The message format is part of the crate's observable API, so it is
/// reproduced here verbatim rather than folded into `validate.rs`.
pub(crate) fn validate_calibration_grid(
    tenors: &[f64],
    forwards: &[f64],
    n_market: usize,
) -> error::Result<()> {
    if tenors.len() != forwards.len() || tenors.len() != n_market {
        return Err(VolSurfError::InvalidInput {
            message: format!(
                "tenors, forwards, and market_data must have the same length: {}, {}, {}",
                tenors.len(),
                forwards.len(),
                n_market
            ),
        });
    }
    validate_positive_slice(tenors, "tenors")?;
    validate_positive_slice(forwards, "forwards")
}

/// Reject non-monotone ATM total variance across tenors.
///
/// θ must be strictly increasing for the surface to be calendar-arbitrage-free;
/// a per-tenor SVI fit that inverts two adjacent θ cannot be repaired globally.
pub(crate) fn check_theta_monotone(
    thetas: &[f64],
    tenors: &[f64],
    model: &'static str,
) -> error::Result<()> {
    debug_assert_eq!(thetas.len(), tenors.len(), "one theta per tenor");
    for (i, w) in thetas.windows(2).enumerate() {
        if w[1] <= w[0] {
            return Err(VolSurfError::CalibrationError {
                message: format!(
                    "per-tenor SVI calibration produced non-monotone ATM variances: \
                     theta[{i}]={:.6} >= theta[{}]={:.6} (tenors {}, {})",
                    w[0],
                    i + 1,
                    w[1],
                    tenors[i],
                    tenors[i + 1]
                ),
                model,
                rms_error: None,
            });
        }
    }
    Ok(())
}

/// Optimize (η, γ) by grid-seeded Nelder-Mead.
///
/// Grid search seeds over η ∈ \[0.01, 3\], γ ∈ \[0, 1\]; Nelder-Mead is then
/// unconstrained above, and only the result is clamped, to η ≥ 1e-6 and
/// γ ∈ \[0, 1\] — so a returned η may fall outside the seed range. Returns
/// `(η, γ, rms)`, where `rms` is the root-mean-square total-variance residual
/// over `n_points` observations.
pub(crate) fn optimize_eta_gamma<F>(
    objective: F,
    n_points: usize,
    model: &'static str,
) -> error::Result<(f64, f64, f64)>
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let eta_lo = 0.01_f64;
    let eta_hi = 3.0_f64;
    let gamma_lo = 0.0_f64;
    let gamma_hi = 1.0_f64;

    let (best_eta, best_gamma, _best_rss) = crate::optim::grid_search_2d(
        GRID_N,
        |ie| eta_lo + (eta_hi - eta_lo) * ie as f64 / (GRID_N - 1) as f64,
        |ig| gamma_lo + (gamma_hi - gamma_lo) * ig as f64 / (GRID_N - 1) as f64,
        objective,
    )
    .ok_or_else(|| VolSurfError::CalibrationError {
        message: "grid search found no valid starting point".into(),
        model,
        rms_error: None,
    })?;

    let step_eta = (eta_hi - eta_lo) / (GRID_N as f64) * 0.5;
    let step_gamma = (gamma_hi - gamma_lo) / (GRID_N as f64) * 0.5;

    let nm_config = crate::optim::NelderMeadConfig::calibration();
    let nm_result = crate::optim::nelder_mead_2d(
        objective, best_eta, best_gamma, step_eta, step_gamma, &nm_config,
    );

    let rms = if n_points > 0 {
        (nm_result.fval / n_points as f64).sqrt()
    } else {
        0.0
    };

    Ok((nm_result.x.max(1e-6), nm_result.y.clamp(0.0, 1.0), rms))
}
