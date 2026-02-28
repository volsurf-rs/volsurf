use volsurf::VolSurface;
use volsurf::surface::{EssviSurface, SsviSurface};
use wasm_bindgen::prelude::*;

use crate::arbitrage::WasmSurfaceDiagnostics;
use crate::error::to_js_err;
use crate::smile::WasmSmile;

fn market_data_from_flat(
    flat: &[f64],
    tenor_sizes: &[usize],
) -> Result<Vec<Vec<(f64, f64)>>, JsValue> {
    let total: usize = tenor_sizes.iter().map(|&n| n * 2).sum();
    if flat.len() != total {
        return Err(JsValue::from_str(&format!(
            "market_data_flat length {} does not match tenor_sizes (expected {})",
            flat.len(),
            total,
        )));
    }
    let mut offset = 0;
    let mut out = Vec::with_capacity(tenor_sizes.len());
    for &n in tenor_sizes {
        let pairs: Vec<(f64, f64)> = flat[offset..offset + n * 2]
            .chunks(2)
            .map(|c| (c[0], c[1]))
            .collect();
        out.push(pairs);
        offset += n * 2;
    }
    Ok(out)
}

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

    pub fn smile_at(&self, expiry: f64) -> Result<WasmSmile, JsValue> {
        let smile = self.inner.smile_at(expiry).map_err(to_js_err)?;
        Ok(WasmSmile::new(smile))
    }

    pub fn diagnostics(&self) -> Result<WasmSurfaceDiagnostics, JsValue> {
        self.inner
            .diagnostics()
            .map(WasmSurfaceDiagnostics::from)
            .map_err(to_js_err)
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmSsviSurface, JsValue> {
        let inner: SsviSurface =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn calibrate(
        market_data_flat: Vec<f64>,
        tenor_sizes: Vec<usize>,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
    ) -> Result<WasmSsviSurface, JsValue> {
        let market_data = market_data_from_flat(&market_data_flat, &tenor_sizes)?;
        let inner = SsviSurface::calibrate(&market_data, &tenors, &forwards).map_err(to_js_err)?;
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
    // #[expect] is unfulfilled here — wasm_bindgen(constructor) transforms the fn before clippy
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

    pub fn smile_at(&self, expiry: f64) -> Result<WasmSmile, JsValue> {
        let smile = self.inner.smile_at(expiry).map_err(to_js_err)?;
        Ok(WasmSmile::new(smile))
    }

    pub fn diagnostics(&self) -> Result<WasmSurfaceDiagnostics, JsValue> {
        self.inner
            .diagnostics()
            .map(WasmSurfaceDiagnostics::from)
            .map_err(to_js_err)
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmEssviSurface, JsValue> {
        let inner: EssviSurface =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn calibrate(
        market_data_flat: Vec<f64>,
        tenor_sizes: Vec<usize>,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
    ) -> Result<WasmEssviSurface, JsValue> {
        let market_data = market_data_from_flat(&market_data_flat, &tenor_sizes)?;
        let inner = EssviSurface::calibrate(&market_data, &tenors, &forwards).map_err(to_js_err)?;
        Ok(Self { inner })
    }
}
