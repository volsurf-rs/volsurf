//! Dupire local volatility extraction.
//!
//! Computes local volatility from the implied volatility surface using:
//!
//! ```text
//! σ²_loc(T, K) = ∂w/∂T / [1 − (k/w)·∂w/∂k + ¼(−¼ − 1/w + k²/w²)·(∂w/∂k)² + ½·∂²w/∂k²]
//! ```
//!
//! where w(T, k) is total implied variance and k = ln(K/F).
//!
//! # References
//! - Dupire, B. "Pricing with a Smile" (1994)
//! - Gatheral, J. "The Volatility Surface: A Practitioner's Guide" (2006), Ch. 2

#![allow(dead_code)] // Stub — not yet implemented (v0.3 scope)

use std::fmt;
use std::sync::Arc;

use crate::error::{self, VolSurfError};
use crate::surface::VolSurface;
use crate::types::Vol;

use super::LocalVol;

/// Dupire local volatility derived from an implied vol surface.
pub struct DupireLocalVol {
    /// The implied volatility surface to differentiate.
    surface: Arc<dyn VolSurface>,
    /// Finite difference step for numerical derivatives.
    bump_size: f64,
}

impl fmt::Debug for DupireLocalVol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DupireLocalVol")
            .field("surface", &self.surface)
            .field("bump_size", &self.bump_size)
            .finish()
    }
}

impl DupireLocalVol {
    /// Create a Dupire local vol from an implied vol surface.
    ///
    /// `bump_size` controls the finite-difference step for numerical
    /// derivatives (default: 0.01 = 1% of strike/tenor).
    pub fn new(surface: Arc<dyn VolSurface>, bump_size: f64) -> Self {
        Self { surface, bump_size }
    }
}

impl LocalVol for DupireLocalVol {
    fn local_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        let _ = (self.surface.as_ref(), self.bump_size, expiry, strike);
        Err(VolSurfError::NumericalError(
            "not yet implemented".to_string(),
        ))
    }
}
