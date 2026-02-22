use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use volsurf::VolSurface;
use volsurf::surface::{EssviSurface, SsviSurface, SurfaceBuilder};

use crate::error::to_py_err;
use crate::types::{PySmile, PySmileModel, PySurface, PySurfaceDiagnostics};

macro_rules! impl_vol_grid {
    ($name:ident) => {
        #[pymethods]
        impl $name {
            #[pyo3(signature = (expiries, strikes))]
            fn vol_grid<'py>(
                &self,
                py: Python<'py>,
                expiries: PyReadonlyArray1<'py, f64>,
                strikes: PyReadonlyArray1<'py, f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let exp: Vec<f64> = expiries.as_array().to_vec();
                let stk: Vec<f64> = strikes.as_array().to_vec();
                let (nexp, nstk) = (exp.len(), stk.len());
                let inner = &self.inner;
                let data = py.detach(|| {
                    let mut out = Vec::with_capacity(nexp * nstk);
                    for &t in &exp {
                        for &k in &stk {
                            out.push(inner.black_vol(t, k)?.0);
                        }
                    }
                    Ok::<_, volsurf::VolSurfError>(out)
                });
                let arr =
                    numpy::ndarray::Array2::from_shape_vec([nexp, nstk], data.map_err(to_py_err)?)
                        .expect("shape matches capacity");
                Ok(arr.into_pyarray(py))
            }
        }
    };
}

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

impl_vol_grid!(PySsviSurface);

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

impl_vol_grid!(PyEssviSurface);

fn consumed() -> PyErr {
    PyRuntimeError::new_err("builder already consumed by build()")
}

#[pyclass(name = "SurfaceBuilder")]
pub struct PySurfaceBuilder {
    inner: Option<SurfaceBuilder>,
}

#[pymethods]
impl PySurfaceBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(SurfaceBuilder::new()),
        }
    }

    fn model(&mut self, model: &PySmileModel) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.model(model.inner));
        Ok(())
    }

    fn spot(&mut self, spot: f64) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.spot(spot));
        Ok(())
    }

    fn rate(&mut self, rate: f64) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.rate(rate));
        Ok(())
    }

    fn dividend_yield(&mut self, q: f64) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.dividend_yield(q));
        Ok(())
    }

    #[pyo3(signature = (expiry, strikes, vols))]
    fn add_tenor(&mut self, expiry: f64, strikes: Vec<f64>, vols: Vec<f64>) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.add_tenor(expiry, &strikes, &vols));
        Ok(())
    }

    #[pyo3(signature = (expiry, strikes, vols, forward))]
    fn add_tenor_with_forward(
        &mut self,
        expiry: f64,
        strikes: Vec<f64>,
        vols: Vec<f64>,
        forward: f64,
    ) -> PyResult<()> {
        let b = self.inner.take().ok_or_else(consumed)?;
        self.inner = Some(b.add_tenor_with_forward(expiry, &strikes, &vols, forward));
        Ok(())
    }

    fn build(&mut self) -> PyResult<PySurface> {
        let b = self.inner.take().ok_or_else(consumed)?;
        let surface = b.build().map_err(to_py_err)?;
        Ok(PySurface {
            inner: Arc::new(surface) as Arc<dyn VolSurface>,
        })
    }
}
