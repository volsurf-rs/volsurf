use std::sync::Arc;

use volsurf::local_vol::{BoundaryLocalVol, DupireLocalVol, LocalVol};
use volsurf::{Strike, Tenor, VolSurface};
use wasm_bindgen::prelude::*;

use crate::error::to_js_err;

/// Build a [`DupireLocalVol`] from a surface, applying an optional bump size.
///
/// Shared seam used by both `WasmDupireLocalVol::from_arc` and
/// `WasmBoundaryLocalVol::from_arc`, and by the per-surface `dupire_local_vol*`
/// methods (which is why it takes an `Arc<dyn VolSurface>` and is not itself
/// `#[wasm_bindgen]`-exported).
fn dupire_from_arc(
    surface: Arc<dyn VolSurface>,
    bump_size: Option<f64>,
) -> Result<DupireLocalVol, JsValue> {
    let mut lv = DupireLocalVol::new(surface);
    if let Some(h) = bump_size {
        lv = lv.with_bump_size(h).map_err(to_js_err)?;
    }
    Ok(lv)
}

/// Dupire local volatility derived from an implied vol surface.
#[wasm_bindgen]
pub struct WasmDupireLocalVol {
    inner: DupireLocalVol,
}

impl WasmDupireLocalVol {
    /// Construct from a shared surface, with an optional finite-difference bump
    /// size (defaults to 1% when `None`). `pub(crate)` because it takes an
    /// `Arc<dyn VolSurface>`, which is not a wasm-bindgen-able type.
    pub(crate) fn from_arc(
        surface: Arc<dyn VolSurface>,
        bump_size: Option<f64>,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: dupire_from_arc(surface, bump_size)?,
        })
    }
}

#[wasm_bindgen]
impl WasmDupireLocalVol {
    /// Local volatility σ_loc at (expiry, strike).
    pub fn local_vol(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        self.inner
            .local_vol(Tenor(expiry), Strike(strike))
            .map(|v| v.0)
            .map_err(to_js_err)
    }
}

/// Dupire local volatility with the small-time boundary handled (v2.2 / PAN-25).
///
/// For `expiry ≤ floor` (the Dupire bump size) it evaluates at `expiry = floor`,
/// rescuing the otherwise-rejected `t = 0`; for `expiry > floor` it delegates
/// bit-identically to [`WasmDupireLocalVol`].
#[wasm_bindgen]
pub struct WasmBoundaryLocalVol {
    inner: BoundaryLocalVol<DupireLocalVol>,
}

impl WasmBoundaryLocalVol {
    /// Construct from a shared surface, wrapping the Dupire local vol in the
    /// boundary adapter via [`DupireLocalVol::with_boundary`].
    pub(crate) fn from_arc(
        surface: Arc<dyn VolSurface>,
        bump_size: Option<f64>,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: dupire_from_arc(surface, bump_size)?.with_boundary(),
        })
    }
}

#[wasm_bindgen]
impl WasmBoundaryLocalVol {
    /// Local volatility σ_loc at (expiry, strike), with the small-time boundary
    /// handled (`expiry = 0` is valid and maps to the floor).
    pub fn local_vol(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        self.inner
            .local_vol(Tenor(expiry), Strike(strike))
            .map(|v| v.0)
            .map_err(to_js_err)
    }
}
