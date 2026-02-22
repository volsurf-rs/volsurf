use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volsurf::VolSurface;
use volsurf::surface::{EssviSurface, SsviSurface, SurfaceBuilder};

use crate::error::to_py_err;
use crate::types::{PySmile, PySmileModel, PySurface, PySurfaceDiagnostics};

#[pyclass(frozen, name = "SsviSurface")]
pub struct PySsviSurface {
    inner: SsviSurface,
}

#[pymethods]
impl PySsviSurface {
    #[new]
    #[pyo3(signature = (rho, eta, gamma, tenors, forwards, thetas))]
    fn new(
        rho: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> PyResult<Self> {
        let inner =
            SsviSurface::new(rho, eta, gamma, tenors, forwards, thetas).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn black_vol(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self.inner.black_vol(expiry, strike).map_err(to_py_err)?.0)
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self
            .inner
            .black_variance(expiry, strike)
            .map_err(to_py_err)?
            .0)
    }

    fn smile_at(&self, expiry: f64) -> PyResult<PySmile> {
        let smile = self.inner.smile_at(expiry).map_err(to_py_err)?;
        Ok(PySmile { inner: smile })
    }

    fn diagnostics(&self) -> PyResult<PySurfaceDiagnostics> {
        Ok(self.inner.diagnostics().map_err(to_py_err)?.into())
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho()
    }

    #[getter]
    fn eta(&self) -> f64 {
        self.inner.eta()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    fn tenors(&self) -> Vec<f64> {
        self.inner.tenors().to_vec()
    }

    fn forwards(&self) -> Vec<f64> {
        self.inner.forwards().to_vec()
    }

    fn thetas(&self) -> Vec<f64> {
        self.inner.thetas().to_vec()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (s))]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: SsviSurface =
            serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyclass(frozen, name = "EssviSurface")]
pub struct PyEssviSurface {
    inner: EssviSurface,
}

#[pymethods]
impl PyEssviSurface {
    #[new]
    #[expect(clippy::too_many_arguments)]
    #[pyo3(signature = (rho_0, rho_m, a, eta, gamma, tenors, forwards, thetas))]
    fn new(
        rho_0: f64,
        rho_m: f64,
        a: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> PyResult<Self> {
        let inner = EssviSurface::new(rho_0, rho_m, a, eta, gamma, tenors, forwards, thetas)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn black_vol(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self.inner.black_vol(expiry, strike).map_err(to_py_err)?.0)
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        Ok(self
            .inner
            .black_variance(expiry, strike)
            .map_err(to_py_err)?
            .0)
    }

    fn smile_at(&self, expiry: f64) -> PyResult<PySmile> {
        let smile = self.inner.smile_at(expiry).map_err(to_py_err)?;
        Ok(PySmile { inner: smile })
    }

    fn diagnostics(&self) -> PyResult<PySurfaceDiagnostics> {
        Ok(self.inner.diagnostics().map_err(to_py_err)?.into())
    }

    #[getter]
    fn rho_0(&self) -> f64 {
        self.inner.rho_0()
    }

    #[getter]
    fn rho_m(&self) -> f64 {
        self.inner.rho_m()
    }

    #[getter]
    fn a(&self) -> f64 {
        self.inner.a()
    }

    #[getter]
    fn eta(&self) -> f64 {
        self.inner.eta()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn theta_max(&self) -> f64 {
        self.inner.theta_max()
    }

    fn tenors(&self) -> Vec<f64> {
        self.inner.tenors().to_vec()
    }

    fn forwards(&self) -> Vec<f64> {
        self.inner.forwards().to_vec()
    }

    fn thetas(&self) -> Vec<f64> {
        self.inner.thetas().to_vec()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (s))]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: EssviSurface =
            serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "SurfaceBuilder")]
pub struct PySurfaceBuilder {
    inner: SurfaceBuilder,
}

#[pymethods]
impl PySurfaceBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: SurfaceBuilder::new(),
        }
    }

    fn model(&mut self, model: &PySmileModel) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.model(model.inner);
    }

    fn spot(&mut self, spot: f64) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.spot(spot);
    }

    fn rate(&mut self, rate: f64) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.rate(rate);
    }

    fn dividend_yield(&mut self, q: f64) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.dividend_yield(q);
    }

    #[pyo3(signature = (expiry, strikes, vols))]
    fn add_tenor(&mut self, expiry: f64, strikes: Vec<f64>, vols: Vec<f64>) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.add_tenor(expiry, &strikes, &vols);
    }

    #[pyo3(signature = (expiry, strikes, vols, forward))]
    fn add_tenor_with_forward(
        &mut self,
        expiry: f64,
        strikes: Vec<f64>,
        vols: Vec<f64>,
        forward: f64,
    ) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.add_tenor_with_forward(expiry, &strikes, &vols, forward);
    }

    fn build(&mut self) -> PyResult<PySurface> {
        let taken = std::mem::take(&mut self.inner);
        let surface = taken.build().map_err(to_py_err)?;
        Ok(PySurface {
            inner: Arc::new(surface) as Arc<dyn VolSurface>,
        })
    }
}
