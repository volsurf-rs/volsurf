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
//! - **Newtypes for outputs, bare `f64` for inputs.** [`Vol`], [`Variance`],
//!   [`Strike`], [`Tenor`] wrap return values to prevent accidental mixing.
//!   Inputs take raw `f64` for ergonomics — validation happens inside model
//!   constructors and the builder.
//! - **No panics.** Every fallible operation returns [`Result`]. Library code
//!   never calls `unwrap()` or `expect()`.
//! - **Immutable surfaces.** Once constructed, a surface cannot be modified.
//!   No interior mutability, no observer pattern.
//! - **Thread-safe.** All traits require `Send + Sync`. Surfaces can be shared
//!   via `Arc<dyn VolSurface>` across pricing threads.
//! - **Serializable.** All value types and model structs implement Serde
//!   `Serialize` / `Deserialize` with validation on deserialization where
//!   invariants exist (SVI, SABR, SSVI, eSSVI parameters).

pub mod conventions;
pub mod error;
pub mod implied;
pub mod local_vol;
mod optim;
pub mod smile;
pub mod surface;
pub mod types;
mod validate;

#[doc(inline)]
pub use error::{Result, VolSurfError};
#[doc(inline)]
pub use local_vol::LocalVol;
#[doc(inline)]
pub use smile::SmileSection;
#[doc(inline)]
pub use surface::VolSurface;
#[doc(inline)]
pub use types::{OptionType, Strike, Tenor, Variance, Vol};
