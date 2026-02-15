//! Local volatility extraction from implied volatility surfaces.
//!
//! Local volatility σ_loc(T, K) is the instantaneous volatility in the
//! Dupire framework. It is derived from the implied volatility surface
//! via the Dupire formula.
//!
//! # Usage
//! Compose [`DupireLocalVol`] around any [`VolSurface`](crate::surface::VolSurface)
//! rather than calling `local_vol()` on the surface directly. This keeps surface
//! types free of Dupire numerics.
//!
//! # References
//! - Dupire, B. "Pricing with a Smile" (1994)

pub mod dupire;

pub use dupire::DupireLocalVol;

use crate::error;
use crate::types::Vol;

/// Local volatility surface.
///
/// Provides σ_loc(T, K) at any point, derived from an implied vol surface.
pub trait LocalVol: Send + Sync {
    /// Local volatility at the given expiry and strike.
    fn local_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol>;
}
