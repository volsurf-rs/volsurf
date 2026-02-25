use volsurf::VolSurface;
use volsurf::surface::{EssviSurface, SsviSurface};
use wasm_bindgen::prelude::*;

use crate::error::to_js_err;

#[wasm_bindgen]
pub struct WasmSsviSurface {
    inner: SsviSurface,
}

#[wasm_bindgen]
impl WasmSsviSurface {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rho: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> Result<WasmSsviSurface, JsValue> {
        let inner =
            SsviSurface::new(rho, eta, gamma, tenors, forwards, thetas).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn black_vol(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self.inner.black_vol(expiry, strike).map_err(to_js_err)?.0)
    }

    pub fn black_variance(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self
            .inner
            .black_variance(expiry, strike)
            .map_err(to_js_err)?
            .0)
    }

    #[wasm_bindgen(getter)]
    pub fn rho(&self) -> f64 {
        self.inner.rho()
    }

    #[wasm_bindgen(getter)]
    pub fn eta(&self) -> f64 {
        self.inner.eta()
    }

    #[wasm_bindgen(getter)]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    pub fn tenors(&self) -> Vec<f64> {
        self.inner.tenors().to_vec()
    }

    pub fn forwards(&self) -> Vec<f64> {
        self.inner.forwards().to_vec()
    }

    pub fn thetas(&self) -> Vec<f64> {
        self.inner.thetas().to_vec()
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmSsviSurface, JsValue> {
        let inner: SsviSurface =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

#[wasm_bindgen]
pub struct WasmEssviSurface {
    inner: EssviSurface,
}

#[wasm_bindgen]
impl WasmEssviSurface {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rho_0: f64,
        rho_m: f64,
        a: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> Result<WasmEssviSurface, JsValue> {
        let inner = EssviSurface::new(rho_0, rho_m, a, eta, gamma, tenors, forwards, thetas)
            .map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn black_vol(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self.inner.black_vol(expiry, strike).map_err(to_js_err)?.0)
    }

    pub fn black_variance(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self
            .inner
            .black_variance(expiry, strike)
            .map_err(to_js_err)?
            .0)
    }

    #[wasm_bindgen(getter)]
    pub fn rho_0(&self) -> f64 {
        self.inner.rho_0()
    }

    #[wasm_bindgen(getter)]
    pub fn rho_m(&self) -> f64 {
        self.inner.rho_m()
    }

    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner.a()
    }

    #[wasm_bindgen(getter)]
    pub fn eta(&self) -> f64 {
        self.inner.eta()
    }

    #[wasm_bindgen(getter)]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[wasm_bindgen(getter)]
    pub fn theta_max(&self) -> f64 {
        self.inner.theta_max()
    }

    pub fn tenors(&self) -> Vec<f64> {
        self.inner.tenors().to_vec()
    }

    pub fn forwards(&self) -> Vec<f64> {
        self.inner.forwards().to_vec()
    }

    pub fn thetas(&self) -> Vec<f64> {
        self.inner.thetas().to_vec()
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmEssviSurface, JsValue> {
        let inner: EssviSurface =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}
