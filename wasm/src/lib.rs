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
pub use smile::{WasmSabrSmile, WasmSmile, WasmSviSmile};
pub use surface::{WasmEssviSurface, WasmSsviSurface};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
