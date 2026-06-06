use volsurf::OptionType;
use wasm_bindgen::prelude::*;

use crate::error::to_js_err;

/// Option type: call or put. Mirrors [`volsurf::OptionType`].
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmOptionType {
    Call,
    Put,
}

impl From<WasmOptionType> for OptionType {
    fn from(v: WasmOptionType) -> Self {
        match v {
            WasmOptionType::Call => OptionType::Call,
            WasmOptionType::Put => OptionType::Put,
        }
    }
}

impl From<OptionType> for WasmOptionType {
    fn from(v: OptionType) -> Self {
        match v {
            OptionType::Call => WasmOptionType::Call,
            OptionType::Put => WasmOptionType::Put,
        }
    }
}

/// Undiscounted Black (lognormal) option price.
#[wasm_bindgen]
pub fn black_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: WasmOptionType,
) -> Result<f64, JsValue> {
    volsurf::implied::black_price(forward, strike, vol, expiry, option_type.into())
        .map_err(to_js_err)
}

/// Undiscounted Bachelier (normal) option price.
#[wasm_bindgen]
pub fn normal_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: WasmOptionType,
) -> Result<f64, JsValue> {
    volsurf::implied::normal_price(forward, strike, vol, expiry, option_type.into())
        .map_err(to_js_err)
}

/// Undiscounted displaced-diffusion option price.
#[wasm_bindgen]
pub fn displaced_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    beta: f64,
    option_type: WasmOptionType,
) -> Result<f64, JsValue> {
    volsurf::implied::displaced_price(forward, strike, vol, expiry, beta, option_type.into())
        .map_err(to_js_err)
}

/// Black (lognormal) implied volatility extraction via Jäckel's algorithm.
#[wasm_bindgen]
pub struct WasmBlackImpliedVol;

#[wasm_bindgen]
impl WasmBlackImpliedVol {
    /// Extract Black implied volatility from an undiscounted option price.
    pub fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: WasmOptionType,
    ) -> Result<f64, JsValue> {
        volsurf::implied::BlackImpliedVol::compute(
            option_price,
            forward,
            strike,
            expiry,
            option_type.into(),
        )
        .map(|v| v.0)
        .map_err(to_js_err)
    }
}

/// Bachelier (normal) implied volatility extraction.
#[wasm_bindgen]
pub struct WasmNormalImpliedVol;

#[wasm_bindgen]
impl WasmNormalImpliedVol {
    /// Extract normal (Bachelier) implied volatility from an undiscounted option price.
    pub fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: WasmOptionType,
    ) -> Result<f64, JsValue> {
        volsurf::implied::NormalImpliedVol::compute(
            option_price,
            forward,
            strike,
            expiry,
            option_type.into(),
        )
        .map(|v| v.0)
        .map_err(to_js_err)
    }
}

/// Displaced-diffusion implied volatility extraction (interpolates normal ↔ Black).
#[wasm_bindgen]
pub struct WasmDisplacedImpliedVol {
    inner: volsurf::implied::DisplacedImpliedVol,
}

#[wasm_bindgen]
impl WasmDisplacedImpliedVol {
    /// Construct with displacement parameter `beta` ∈ [0, 1].
    #[wasm_bindgen(constructor)]
    pub fn new(beta: f64) -> Result<WasmDisplacedImpliedVol, JsValue> {
        let inner = volsurf::implied::DisplacedImpliedVol::new(beta).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// Extract displaced-diffusion implied volatility from an undiscounted option price.
    pub fn compute(
        &self,
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: WasmOptionType,
    ) -> Result<f64, JsValue> {
        self.inner
            .compute(option_price, forward, strike, expiry, option_type.into())
            .map(|v| v.0)
            .map_err(to_js_err)
    }
}

/// Log-moneyness: k = ln(K / F).
#[wasm_bindgen]
pub fn log_moneyness(strike: f64, forward: f64) -> Result<f64, JsValue> {
    volsurf::conventions::log_moneyness(strike, forward).map_err(to_js_err)
}

/// Simple moneyness: m = K / F.
#[wasm_bindgen]
pub fn moneyness(strike: f64, forward: f64) -> Result<f64, JsValue> {
    volsurf::conventions::moneyness(strike, forward).map_err(to_js_err)
}

/// Forward price from spot: F = S · exp((r − q) · T).
#[wasm_bindgen]
pub fn forward_price(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    expiry: f64,
) -> Result<f64, JsValue> {
    volsurf::conventions::forward_price(spot, rate, dividend_yield, expiry).map_err(to_js_err)
}
