//! # volsurf
//!
//! Production-ready volatility surface library for derivatives pricing.
//!
//! Provides the full pipeline: raw option quotes → implied vol extraction →
//! parametric surface fitting → arbitrage-free enforcement → local vol / pricing
//! engine input.
//!
//! ## Architecture
//!
//! - **`implied`** — Implied volatility extraction (Black, Bachelier, displaced diffusion)
//! - **`smile`** — Single-tenor smile models (SVI, SABR, cubic spline)
//! - **`surface`** — Multi-tenor surface construction (SSVI, eSSVI, piecewise)
//! - **`local_vol`** — Dupire local volatility extraction
//!
//! ## Design
//!
//! - **Newtypes for vol units and query arguments.** [`Vol`], [`NormalVol`],
//!   [`DisplacedVol`] and [`Variance`] keep the quoting conventions apart on
//!   both sides of a call — [`black_price`](implied::black_price) takes the
//!   same [`Vol`] that [`BlackImpliedVol`](implied::BlackImpliedVol) returns.
//!   [`Strike`] and [`Tenor`] wrap the query arguments that could be
//!   transposed — `black_vol(Tenor(0.5), Strike(100.0))` reads one way only.
//! - **No panics.** Every fallible operation returns [`Result`]. Library code
//!   never calls `unwrap()` or `expect()`.
//! - **Immutable surfaces.** Once constructed, a surface cannot be modified.
//!   No interior mutability, no observer pattern.
//! - **Thread-safe.** All traits require `Send + Sync`. Surfaces can be shared
//!   via `Arc<dyn VolSurface>` across pricing threads.
//! - **Serializable.** All value types and model structs implement Serde
//!   `Serialize` / `Deserialize` with validation on deserialization where
//!   invariants exist (SVI, SABR, SSVI, eSSVI parameters).

pub mod calibration;
pub mod conventions;
pub mod error;
pub mod implied;
pub mod local_vol;
mod optim;
mod serde_raw;
pub mod smile;
pub mod surface;
#[cfg(test)]
mod test_support;
pub mod types;
mod validate;

#[doc(inline)]
pub use calibration::{DataFilter, WeightingScheme, apply_filter};
#[doc(inline)]
pub use error::{Result, VolSurfError};
#[doc(inline)]
pub use local_vol::{BoundaryLocalVol, DupireLocalVol, LocalVol};
#[doc(inline)]
pub use smile::{ArbitrageScanConfig, SmileCalibrator, SmileSection};
#[doc(inline)]
pub use surface::VolSurface;
#[doc(inline)]
pub use types::{DisplacedVol, NormalVol, OptionType, Strike, Tenor, Variance, Vol};
