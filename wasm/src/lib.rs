use wasm_bindgen::prelude::*;

mod arbitrage;
mod builder;
mod error;
mod implied;
mod local_vol;
mod smile;
mod surface;

pub use arbitrage::{
    WasmArbitrageReport, WasmButterflyViolation, WasmCalendarViolation, WasmSurfaceDiagnostics,
};
pub use builder::{WasmPiecewiseSurface, WasmSurfaceBuilder};
pub use implied::{
    WasmBlackImpliedVol, WasmDisplacedImpliedVol, WasmNormalImpliedVol, WasmOptionType,
    black_price, displaced_price, forward_price, log_moneyness, moneyness, normal_price,
};
pub use local_vol::{WasmBoundaryLocalVol, WasmDupireLocalVol};
pub use smile::{
    WasmArbitrageScanConfig, WasmDataFilter, WasmSabrSmile, WasmSmile, WasmSviSmile,
    WasmWeightingScheme, weighting_model_default, weighting_uniform, weighting_vega,
};
pub use surface::{WasmEssviSurface, WasmPerTenorFit, WasmSsviSurface};

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
