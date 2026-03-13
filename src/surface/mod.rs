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
pub(crate) const EXPIRY_MATCH_TOL: f64 = 1e-10;

use crate::error;
use crate::smile::SmileSection;
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
    fn black_vol(&self, expiry: Tenor, strike: Strike) -> error::Result<Vol>;

    /// Black total variance σ²(T, K) · T.
    ///
    /// Cross-tenor interpolation is performed in variance space because
    /// total variance must be non-decreasing in time for no-arbitrage.
    fn black_variance(&self, expiry: Tenor, strike: Strike) -> error::Result<Variance>;

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
}
