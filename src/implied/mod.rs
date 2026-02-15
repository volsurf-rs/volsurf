//! Implied volatility extraction from option prices.
//!
//! Provides three models for extracting implied volatility:
//!
//! - [`BlackImpliedVol`] — Standard Black (lognormal) model via Jäckel's algorithm
//! - [`NormalImpliedVol`] — Bachelier (normal) model for fixed income / short-dated FX
//! - [`DisplacedImpliedVol`] — Displaced diffusion hybrid (interpolates normal ↔ Black)

pub mod black;
pub mod displaced;
pub mod normal;

pub use black::{black_price, BlackImpliedVol};
pub use displaced::DisplacedImpliedVol;
pub use normal::NormalImpliedVol;
