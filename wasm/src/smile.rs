use volsurf::Strike;
use volsurf::calibration::{DataFilter, WeightingScheme};
use volsurf::smile::{SabrSmile, SmileSection, SviSmile};
use wasm_bindgen::prelude::*;

use crate::arbitrage::WasmArbitrageReport;
use crate::error::to_js_err;

macro_rules! impl_wasm_smile_methods {
    ($name:ident) => {
        #[wasm_bindgen]
        impl $name {
            pub fn vol(&self, strike: f64) -> Result<f64, JsValue> {
                self.inner
                    .vol(Strike(strike))
                    .map(|v| v.0)
                    .map_err(to_js_err)
            }

            pub fn variance(&self, strike: f64) -> Result<f64, JsValue> {
                self.inner
                    .variance(Strike(strike))
                    .map(|v| v.0)
                    .map_err(to_js_err)
            }

            pub fn density(&self, strike: f64) -> Result<f64, JsValue> {
                self.inner.density(Strike(strike)).map_err(to_js_err)
            }

            #[wasm_bindgen(getter)]
            pub fn forward(&self) -> f64 {
                self.inner.forward()
            }

            #[wasm_bindgen(getter)]
            pub fn expiry(&self) -> f64 {
                self.inner.expiry()
            }

            pub fn is_arbitrage_free(&self) -> Result<WasmArbitrageReport, JsValue> {
                self.inner
                    .is_arbitrage_free()
                    .map(WasmArbitrageReport::from)
                    .map_err(to_js_err)
            }
        }
    };
}

fn pairs_from_flat(flat: &[f64]) -> Result<Vec<(f64, f64)>, JsValue> {
    if !flat.len().is_multiple_of(2) {
        return Err(JsValue::from_str(
            "market_vols must have even length (strike, vol pairs)",
        ));
    }
    Ok(flat.chunks(2).map(|c| (c[0], c[1])).collect())
}

#[wasm_bindgen]
pub struct WasmDataFilter {
    inner: DataFilter,
}

#[wasm_bindgen]
impl WasmDataFilter {
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_log_moneyness: Option<f64>,
        min_vol: Option<f64>,
        vol_cliff_filter: Option<bool>,
    ) -> WasmDataFilter {
        Self {
            inner: DataFilter {
                max_log_moneyness,
                min_vol,
                vol_cliff_filter,
            },
        }
    }

    #[wasm_bindgen(getter)]
    pub fn max_log_moneyness(&self) -> Option<f64> {
        self.inner.max_log_moneyness
    }

    #[wasm_bindgen(getter)]
    pub fn min_vol(&self) -> Option<f64> {
        self.inner.min_vol
    }

    #[wasm_bindgen(getter)]
    pub fn vol_cliff_filter(&self) -> Option<bool> {
        self.inner.vol_cliff_filter
    }
}

#[wasm_bindgen]
pub struct WasmWeightingScheme {
    inner: WeightingScheme,
}

#[wasm_bindgen]
pub fn weighting_model_default() -> WasmWeightingScheme {
    WasmWeightingScheme {
        inner: WeightingScheme::ModelDefault,
    }
}

#[wasm_bindgen]
pub fn weighting_vega() -> WasmWeightingScheme {
    WasmWeightingScheme {
        inner: WeightingScheme::Vega,
    }
}

#[wasm_bindgen]
pub fn weighting_uniform() -> WasmWeightingScheme {
    WasmWeightingScheme {
        inner: WeightingScheme::Uniform,
    }
}

#[wasm_bindgen]
pub struct WasmSviSmile {
    inner: SviSmile,
}

impl WasmSviSmile {
    pub(crate) fn from_inner(inner: SviSmile) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl WasmSviSmile {
    #[wasm_bindgen(constructor)]
    pub fn new(
        forward: f64,
        expiry: f64,
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
    ) -> Result<WasmSviSmile, JsValue> {
        let inner = SviSmile::new(forward, expiry, a, b, rho, m, sigma).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn calibrate(
        forward: f64,
        expiry: f64,
        market_vols_flat: Vec<f64>,
    ) -> Result<WasmSviSmile, JsValue> {
        let pairs = pairs_from_flat(&market_vols_flat)?;
        let inner = SviSmile::calibrate(forward, expiry, &pairs).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn calibrate_with_config(
        forward: f64,
        expiry: f64,
        market_vols_flat: Vec<f64>,
        filter: Option<WasmDataFilter>,
        weighting: Option<WasmWeightingScheme>,
        seed: Option<WasmSviSmile>,
    ) -> Result<WasmSviSmile, JsValue> {
        let pairs = pairs_from_flat(&market_vols_flat)?;
        let f = filter.map_or_else(DataFilter::default, |f| f.inner);
        let w = weighting.map_or_else(WeightingScheme::default, |w| w.inner);
        let inner = SviSmile::calibrate_with_config(
            forward,
            expiry,
            &pairs,
            &f,
            &w,
            seed.as_ref().map(|s| &s.inner),
        )
        .map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmSviSmile, JsValue> {
        let inner: SviSmile =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

impl_wasm_smile_methods!(WasmSviSmile);

#[wasm_bindgen]
pub struct WasmSabrSmile {
    inner: SabrSmile,
}

#[wasm_bindgen]
impl WasmSabrSmile {
    #[wasm_bindgen(constructor)]
    pub fn new(
        forward: f64,
        expiry: f64,
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
    ) -> Result<WasmSabrSmile, JsValue> {
        let inner = SabrSmile::new(forward, expiry, alpha, beta, rho, nu).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn calibrate(
        forward: f64,
        expiry: f64,
        beta: f64,
        market_vols_flat: Vec<f64>,
    ) -> Result<WasmSabrSmile, JsValue> {
        let pairs = pairs_from_flat(&market_vols_flat)?;
        let inner = SabrSmile::calibrate(forward, expiry, beta, &pairs).map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn calibrate_with_config(
        forward: f64,
        expiry: f64,
        beta: f64,
        market_vols_flat: Vec<f64>,
        filter: Option<WasmDataFilter>,
        weighting: Option<WasmWeightingScheme>,
        seed: Option<WasmSabrSmile>,
    ) -> Result<WasmSabrSmile, JsValue> {
        let pairs = pairs_from_flat(&market_vols_flat)?;
        let f = filter.map_or_else(DataFilter::default, |f| f.inner);
        let w = weighting.map_or_else(WeightingScheme::default, |w| w.inner);
        let inner = SabrSmile::calibrate_with_config(
            forward,
            expiry,
            beta,
            &pairs,
            &f,
            &w,
            seed.as_ref().map(|s| &s.inner),
        )
        .map_err(to_js_err)?;
        Ok(Self { inner })
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<WasmSabrSmile, JsValue> {
        let inner: SabrSmile =
            serde_json::from_str(s).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }
}

impl_wasm_smile_methods!(WasmSabrSmile);

#[wasm_bindgen]
pub struct WasmSmile {
    inner: Box<dyn SmileSection>,
}

impl WasmSmile {
    pub(crate) fn new(inner: Box<dyn SmileSection>) -> Self {
        Self { inner }
    }
}

impl_wasm_smile_methods!(WasmSmile);
