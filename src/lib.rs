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

// TODO: Remove once stub implementations are filled in.
#![allow(dead_code)]

pub mod conventions;
pub mod error;
pub mod implied;
pub mod local_vol;
pub mod smile;
pub mod surface;
pub mod types;

pub use error::{Result, VolSurfError};
pub use types::{OptionType, Strike, Tenor, Variance, Vol};
