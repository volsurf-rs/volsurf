use std::sync::Arc;

use volsurf::VolSurface;
use volsurf::calibration::{DataFilter, WeightingScheme};
use volsurf::surface::{SmileModel, SurfaceBuilder};
use volsurf::{Strike, Tenor};
use wasm_bindgen::prelude::*;

use crate::arbitrage::WasmSurfaceDiagnostics;
use crate::error::to_js_err;
use crate::smile::WasmSmile;

fn consumed() -> JsValue {
    JsValue::from_str("builder already consumed by build()")
}

#[wasm_bindgen]
pub struct WasmSurfaceBuilder {
    inner: Option<SurfaceBuilder>,
}

impl Default for WasmSurfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
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
        if !beta.is_finite() || !(0.0..=1.0).contains(&beta) {
            return Err(JsValue::from_str(&format!(
                "SABR beta must be in [0, 1] and finite, got {beta}"
            )));
        }
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.model(SmileModel::Sabr { beta }));
        Ok(())
    }

    pub fn data_filter(
        &mut self,
        max_log_moneyness: Option<f64>,
        min_vol: Option<f64>,
        vol_cliff_filter: Option<bool>,
    ) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.data_filter(DataFilter {
            max_log_moneyness,
            min_vol,
            vol_cliff_filter,
        }));
        Ok(())
    }

    pub fn weighting_vega(&mut self) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.weighting(WeightingScheme::Vega));
        Ok(())
    }

    pub fn weighting_uniform(&mut self) -> Result<(), JsValue> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.weighting(WeightingScheme::Uniform));
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
        let backup = b.clone();
        match b.build() {
            Ok(surface) => Ok(WasmPiecewiseSurface {
                inner: Arc::new(surface) as Arc<dyn VolSurface>,
            }),
            Err(e) => {
                self.inner = Some(backup);
                Err(to_js_err(e))
            }
        }
    }
}

#[wasm_bindgen]
pub struct WasmPiecewiseSurface {
    inner: Arc<dyn VolSurface>,
}

#[wasm_bindgen]
impl WasmPiecewiseSurface {
    pub fn black_vol(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self
            .inner
            .black_vol(Tenor(expiry), Strike(strike))
            .map_err(to_js_err)?
            .0)
    }

    pub fn black_variance(&self, expiry: f64, strike: f64) -> Result<f64, JsValue> {
        Ok(self
            .inner
            .black_variance(Tenor(expiry), Strike(strike))
            .map_err(to_js_err)?
            .0)
    }

    pub fn smile_at(&self, expiry: f64) -> Result<WasmSmile, JsValue> {
        let smile = self.inner.smile_at(Tenor(expiry)).map_err(to_js_err)?;
        Ok(WasmSmile::new(smile))
    }

    pub fn diagnostics(&self) -> Result<WasmSurfaceDiagnostics, JsValue> {
        self.inner
            .diagnostics()
            .map(WasmSurfaceDiagnostics::from)
            .map_err(to_js_err)
    }
}
