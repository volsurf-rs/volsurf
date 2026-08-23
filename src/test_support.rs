//! Fixtures shared across the crate's unit tests.
//!
//! Each model's `mod tests` is private, so helpers that are genuinely the same
//! across models were being retyped per module. Anything model-agnostic belongs
//! here; model-specific fixtures stay with their model.

use crate::surface::VolSurface;
use crate::types::{Strike, Tenor};

/// Sample `surface` at every `(tenor, strike)` pair, yielding market-data
/// quotes a calibrator can be round-tripped against.
///
/// Panics on a failed query — a fixture that cannot evaluate its own surface is
/// a broken test, not a condition to propagate.
pub(crate) fn synthetic_surface_data<S: VolSurface>(
    surface: &S,
    tenors: &[f64],
    strikes_per_tenor: &[Vec<f64>],
) -> Vec<Vec<(f64, f64)>> {
    debug_assert_eq!(
        tenors.len(),
        strikes_per_tenor.len(),
        "one strike ladder per tenor"
    );
    tenors
        .iter()
        .zip(strikes_per_tenor)
        .map(|(&t, strikes)| {
            strikes
                .iter()
                .map(|&k| (k, surface.black_vol(Tenor(t), Strike(k)).unwrap().0))
                .collect()
        })
        .collect()
}

/// The same uniform strike ladder `lo, lo + step, …` repeated at `n_tenors`
/// tenors — the shape `synthetic_surface_data` expects.
pub(crate) fn strike_ladder(
    n_tenors: usize,
    n_strikes: usize,
    lo: f64,
    step: f64,
) -> Vec<Vec<f64>> {
    vec![(0..n_strikes).map(|i| lo + step * i as f64).collect(); n_tenors]
}
