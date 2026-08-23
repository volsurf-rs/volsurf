//! Multi-tenor volatility surface construction.
//!
//! A volatility surface maps (expiry, strike) → implied vol across multiple
//! tenors. This module provides several surface representations:
//!
//! - [`SsviSurface`] — Global SSVI parameterization (Gatheral-Jacquier)
//! - [`EssviSurface`] — Extended SSVI with calendar-spread no-arb guarantees
//! - [`PiecewiseSurface`] — Per-tenor [`SmileSection`]s with cross-tenor
//!   variance interpolation
//! - [`SurfaceBuilder`] — Ergonomic builder API for surface construction

pub mod arbitrage;
pub mod builder;
pub(crate) mod calib;
pub mod essvi;
pub(crate) mod interp;
pub mod piecewise;
pub mod ssvi;

pub use arbitrage::{CalendarViolation, SurfaceDiagnostics};
pub use builder::{SmileModel, SurfaceBuilder};
pub use essvi::{EssviSlice, EssviSurface, PerTenorFit, StructuralViolation};
pub use piecewise::PiecewiseSurface;
pub use ssvi::{SsviSlice, SsviSurface};

pub(crate) const CALENDAR_ARB_TOL: f64 = 1e-10;
pub(crate) const CALENDAR_CHECK_GRID_SIZE: usize = 41;
pub(crate) const EXPIRY_MATCH_TOL: f64 = 1e-10;

use crate::error;
use crate::smile::{ArbitrageScanConfig, SmileSection};
use crate::types::{Strike, Tenor, Variance, Vol};

/// A full volatility surface: (expiry, strike) → vol.
///
/// All implementations must be `Send + Sync` for safe concurrent pricing
/// across multiple threads. Surfaces are immutable after construction.
///
/// # Design
/// - No global state — evaluation date is implicit in the tenors
/// - Immutable after construction — no observer pattern
/// - Ragged strike grids — each tenor can have different strikes
/// - Local vol is computed via [`DupireLocalVol`](crate::local_vol::DupireLocalVol)
///   by composing it around any `VolSurface`, not as a trait method here.
///   This avoids forcing every surface type to embed Dupire numerics.
///
/// # Examples
///
/// ```
/// use volsurf::surface::{SsviSurface, VolSurface};
/// use volsurf::types::{Strike, Tenor};
///
/// let surface = SsviSurface::new(
///     -0.3, 0.5, 0.5,
///     vec![0.25, 0.5, 1.0],
///     vec![100.0, 100.0, 100.0],
///     vec![0.04, 0.08, 0.16],
/// )?;
///
/// let vol = surface.black_vol(Tenor(0.5), Strike(100.0))?;
/// assert!(vol.0 > 0.0);
///
/// let var = surface.black_variance(Tenor(0.5), Strike(100.0))?;
/// assert!((var.0 - vol.0 * vol.0 * 0.5).abs() < 1e-12);
///
/// let smile = surface.smile_at(Tenor(0.5))?;
/// assert!(smile.vol(Strike(100.0))?.0 > 0.0);
/// # Ok::<(), volsurf::VolSurfError>(())
/// ```
pub trait VolSurface: Send + Sync + std::fmt::Debug {
    /// Black implied volatility σ(T, K).
    ///
    /// The `black_` prefix disambiguates from [`LocalVol::local_vol`](crate::local_vol::LocalVol::local_vol),
    /// which also maps (T, K) → σ but means the instantaneous diffusion
    /// coefficient. [`SmileSection::vol`] omits the prefix because there is
    /// no ambiguity at the single-tenor level.
    fn black_vol(&self, expiry: Tenor, strike: Strike) -> error::Result<Vol> {
        crate::validate::validate_positive(expiry.0, "expiry")?;
        let variance = self.black_variance(expiry, strike)?;
        Ok(Vol((variance.0 / expiry.0).sqrt()))
    }

    /// Black total variance σ²(T, K) · T.
    ///
    /// Cross-tenor interpolation is performed in variance space because
    /// total variance must be non-decreasing in time for no-arbitrage.
    fn black_variance(&self, expiry: Tenor, strike: Strike) -> error::Result<Variance>;

    /// Forward price F(T) at the given expiry.
    ///
    /// Every surface already knows its forwards — parametric surfaces
    /// interpolate them alongside θ, piecewise surfaces read them off the
    /// stored smiles. Exposing that directly spares callers who need only the
    /// forward from building a whole smile section for it: Dupire's
    /// log-moneyness conversion needs three forwards per query and no vols.
    ///
    /// The default implementation goes through [`smile_at`](VolSurface::smile_at);
    /// implementations with a cheaper route should override it.
    fn forward(&self, expiry: Tenor) -> error::Result<f64> {
        Ok(self.smile_at(expiry)?.forward())
    }

    /// A smile section at the given expiry.
    ///
    /// Returns an owned `Box<dyn SmileSection>` because parametric surfaces
    /// (SSVI, eSSVI) compute slices on the fly from global parameters —
    /// there is no stored object to borrow. A reference return would require
    /// interior mutability. The heap allocation is acceptable: `smile_at()`
    /// is called once per tenor setup, not per option.
    fn smile_at(&self, expiry: Tenor) -> error::Result<Box<dyn SmileSection>>;

    /// Surface-level arbitrage diagnostics (butterfly + calendar).
    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics>;

    /// Surface-level diagnostics with custom butterfly scan configuration.
    ///
    /// Passes `config` through to per-smile `is_arbitrage_free_with()` calls.
    /// Calendar spread checks use the same hardcoded grid as `diagnostics()`.
    fn diagnostics_with(&self, config: &ArbitrageScanConfig) -> error::Result<SurfaceDiagnostics>;

    /// Calendar spread violations: total variance decreasing in time.
    ///
    /// Reachable through `&dyn VolSurface`, unlike the model-specific checks
    /// the parametric surfaces used to expose only as inherent methods.
    ///
    /// The default scans total variance across adjacent tenor pairs on a
    /// log-spaced strike grid — the same scan [`diagnostics`](VolSurface::diagnostics)
    /// performs. Surfaces whose parameterization admits an exact test should
    /// override this; [`SsviSurface`] does, via `∂w/∂θ`.
    fn calendar_violations(&self) -> error::Result<Vec<CalendarViolation>> {
        let tenors = self.tenors().to_vec();
        let forwards = tenors
            .iter()
            .map(|&t| self.forward(Tenor(t)))
            .collect::<error::Result<Vec<_>>>()?;

        arbitrage::calendar_scan(&tenors, &forwards, |i, strike| {
            Ok(self.black_variance(Tenor(tenors[i]), Strike(strike))?.0)
        })
    }

    /// The tenor grid (expiries in years) that this surface covers.
    fn tenors(&self) -> &[f64];
}
