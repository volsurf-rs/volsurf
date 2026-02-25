use wasm_bindgen::prelude::*;

mod builder;
mod error;
mod implied;
mod smile;
mod surface;
mod types;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
