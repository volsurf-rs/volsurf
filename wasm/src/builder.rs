use std::sync::Arc;

use volsurf::VolSurface;
use volsurf::surface::{SmileModel, SurfaceBuilder};
use wasm_bindgen::prelude::*;

use crate::error::to_js_err;

fn consumed() -> JsValue {
    JsValue::from_str("builder already consumed by build()")
}

#[wasm_bindgen]
pub struct WasmSurfaceBuilder {
    inner: Option<SurfaceBuilder>,
}

#[wasm_bindgen]
impl WasmSurfaceBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmSurfaceBuilder {
        Self {
            inner: Some(SurfaceBuilder::new()),
        }
    }

    pub fn model_svi(&mut self) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.model(SmileModel::Svi));
        Ok(())
    }

    pub fn model_cubic_spline(&mut self) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.model(SmileModel::CubicSpline));
        Ok(())
    }

    pub fn model_sabr(&mut self, beta: f64) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.model(SmileModel::Sabr { beta }));
        Ok(())
    }

    pub fn spot(&mut self, spot: f64) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.spot(spot));
        Ok(())
    }

    pub fn rate(&mut self, rate: f64) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.rate(rate));
        Ok(())
    }

    pub fn dividend_yield(&mut self, q: f64) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.dividend_yield(q));
        Ok(())
    }

    pub fn add_tenor(
        &mut self,
        expiry: f64,
        strikes: Vec<f64>,
        vols: Vec<f64>,
    ) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.add_tenor(expiry, &strikes, &vols));
        Ok(())
    }

    pub fn add_tenor_with_forward(
        &mut self,
        expiry: f64,
        strikes: Vec<f64>,
        vols: Vec<f64>,
        forward: f64,
    ) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.add_tenor_with_forward(expiry, &strikes, &vols, forward));
        Ok(())
    }

    pub fn build(&mut self) -> Result<WasmPiecewiseSurface, JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        let surface = b.build().map_err(to_js_err)?;
        Ok(WasmPiecewiseSurface {
            inner: Arc::new(surface) as Arc<dyn VolSurface>,
        })
    }
}

#[wasm_bindgen]
pub struct WasmPiecewiseSurface {
    inner: Arc<dyn VolSurface>,
}

#[wasm_bindgen]
impl WasmPiecewiseSurface {
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
}
