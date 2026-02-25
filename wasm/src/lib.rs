use wasm_bindgen::prelude::*;

mod builder;
mod error;
mod implied;
mod smile;
mod surface;
mod types;

pub use builder::{WasmPiecewiseSurface, WasmSurfaceBuilder};
pub use smile::{WasmSabrSmile, WasmSviSmile};
pub use surface::{WasmEssviSurface, WasmSsviSurface};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
