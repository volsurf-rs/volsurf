use wasm_bindgen::prelude::*;

mod arbitrage;
mod builder;
mod error;
mod smile;
mod surface;

pub use arbitrage::{
    WasmArbitrageReport, WasmButterflyViolation, WasmCalendarViolation, WasmSurfaceDiagnostics,
};
pub use builder::{WasmPiecewiseSurface, WasmSurfaceBuilder};
pub use smile::{
    WasmDataFilter, WasmSabrSmile, WasmSmile, WasmSviSmile, WasmWeightingScheme,
    weighting_model_default, weighting_uniform, weighting_vega,
};
pub use surface::{WasmEssviSurface, WasmPerTenorFit, WasmSsviSurface};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
