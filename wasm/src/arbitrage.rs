use volsurf::smile::ArbitrageReport;
use volsurf::surface::SurfaceDiagnostics;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmButterflyViolation {
    strike: f64,
    density: f64,
    magnitude: f64,
}

#[wasm_bindgen]
impl WasmButterflyViolation {
    #[wasm_bindgen(getter)]
    pub fn strike(&self) -> f64 {
        self.strike
    }

    #[wasm_bindgen(getter)]
    pub fn density(&self) -> f64 {
        self.density
    }

    #[wasm_bindgen(getter)]
    pub fn magnitude(&self) -> f64 {
        self.magnitude
    }
}

#[wasm_bindgen]
pub struct WasmArbitrageReport {
    inner: ArbitrageReport,
}

impl From<ArbitrageReport> for WasmArbitrageReport {
    fn from(inner: ArbitrageReport) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl WasmArbitrageReport {
    #[wasm_bindgen(getter)]
    pub fn is_arbitrage_free(&self) -> bool {
        self.inner.is_free
    }

    pub fn butterfly_violations(&self) -> Vec<WasmButterflyViolation> {
        self.inner
            .butterfly_violations
            .iter()
            .map(|v| WasmButterflyViolation {
                strike: v.strike,
                density: v.density,
                magnitude: v.magnitude,
            })
            .collect()
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[wasm_bindgen]
pub struct WasmCalendarViolation {
    strike: f64,
    tenor_short: f64,
    tenor_long: f64,
    variance_short: f64,
    variance_long: f64,
}

#[wasm_bindgen]
impl WasmCalendarViolation {
    #[wasm_bindgen(getter)]
    pub fn strike(&self) -> f64 {
        self.strike
    }

    #[wasm_bindgen(getter)]
    pub fn tenor_short(&self) -> f64 {
        self.tenor_short
    }

    #[wasm_bindgen(getter)]
    pub fn tenor_long(&self) -> f64 {
        self.tenor_long
    }

    #[wasm_bindgen(getter)]
    pub fn variance_short(&self) -> f64 {
        self.variance_short
    }

    #[wasm_bindgen(getter)]
    pub fn variance_long(&self) -> f64 {
        self.variance_long
    }
}

#[wasm_bindgen]
pub struct WasmSurfaceDiagnostics {
    inner: SurfaceDiagnostics,
}

impl From<SurfaceDiagnostics> for WasmSurfaceDiagnostics {
    fn from(inner: SurfaceDiagnostics) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl WasmSurfaceDiagnostics {
    #[wasm_bindgen(getter)]
    pub fn is_arbitrage_free(&self) -> bool {
        self.inner.is_free
    }

    pub fn smile_reports(&self) -> Vec<WasmArbitrageReport> {
        self.inner
            .smile_reports
            .iter()
            .cloned()
            .map(WasmArbitrageReport::from)
            .collect()
    }

    pub fn calendar_violations(&self) -> Vec<WasmCalendarViolation> {
        self.inner
            .calendar_violations
            .iter()
            .map(|v| WasmCalendarViolation {
                strike: v.strike,
                tenor_short: v.tenor_short,
                tenor_long: v.tenor_long,
                variance_short: v.variance_short,
                variance_long: v.variance_long,
            })
            .collect()
    }

    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
