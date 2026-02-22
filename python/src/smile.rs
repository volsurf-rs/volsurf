use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volsurf::smile::{SabrSmile, SmileSection, SplineSmile, SviSmile};

use crate::error::to_py_err;
use crate::types::PyArbitrageReport;

#[pyclass(frozen, name = "SviSmile")]
pub struct PySviSmile {
    inner: SviSmile,
}

#[pymethods]
impl PySviSmile {
    #[new]
    #[pyo3(signature = (forward, expiry, a, b, rho, m, sigma))]
    fn new(
        forward: f64,
        expiry: f64,
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
    ) -> PyResult<Self> {
        let inner = SviSmile::new(forward, expiry, a, b, rho, m, sigma).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn vol(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.vol(strike).map_err(to_py_err)?.0)
    }

    fn variance(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.variance(strike).map_err(to_py_err)?.0)
    }

    fn density(&self, strike: f64) -> PyResult<f64> {
        self.inner.density(strike).map_err(to_py_err)
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry()
    }

    fn is_arbitrage_free(&self) -> PyResult<PyArbitrageReport> {
        Ok(self.inner.is_arbitrage_free().map_err(to_py_err)?.into())
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (s))]
    fn from_json(s: &str) -> PyResult<Self> {
        serde_json::from_str(s)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (strikes))]
    fn vol_array<'py>(
        &self,
        py: Python<'py>,
        strikes: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let stk: Vec<f64> = strikes.as_array().to_vec();
        let inner = &self.inner;
        let data = py.detach(|| {
            stk.iter()
                .map(|&k| inner.vol(k).map(|v| v.0))
                .collect::<Result<Vec<f64>, _>>()
        });
        Ok(data.map_err(to_py_err)?.into_pyarray(py))
    }
}

#[pyclass(frozen, name = "SabrSmile")]
pub struct PySabrSmile {
    inner: SabrSmile,
}

#[pymethods]
impl PySabrSmile {
    #[new]
    #[pyo3(signature = (forward, expiry, alpha, beta, rho, nu))]
    fn new(forward: f64, expiry: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> PyResult<Self> {
        let inner = SabrSmile::new(forward, expiry, alpha, beta, rho, nu).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn vol(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.vol(strike).map_err(to_py_err)?.0)
    }

    fn variance(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.variance(strike).map_err(to_py_err)?.0)
    }

    fn density(&self, strike: f64) -> PyResult<f64> {
        self.inner.density(strike).map_err(to_py_err)
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry()
    }

    fn is_arbitrage_free(&self) -> PyResult<PyArbitrageReport> {
        Ok(self.inner.is_arbitrage_free().map_err(to_py_err)?.into())
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (s))]
    fn from_json(s: &str) -> PyResult<Self> {
        serde_json::from_str(s)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (strikes))]
    fn vol_array<'py>(
        &self,
        py: Python<'py>,
        strikes: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let stk: Vec<f64> = strikes.as_array().to_vec();
        let inner = &self.inner;
        let data = py.detach(|| {
            stk.iter()
                .map(|&k| inner.vol(k).map(|v| v.0))
                .collect::<Result<Vec<f64>, _>>()
        });
        Ok(data.map_err(to_py_err)?.into_pyarray(py))
    }
}

#[pyclass(frozen, name = "SplineSmile")]
pub struct PySplineSmile {
    inner: SplineSmile,
}

#[pymethods]
impl PySplineSmile {
    #[new]
    #[pyo3(signature = (forward, expiry, strikes, variances))]
    fn new(forward: f64, expiry: f64, strikes: Vec<f64>, variances: Vec<f64>) -> PyResult<Self> {
        let inner = SplineSmile::new(forward, expiry, strikes, variances).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn vol(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.vol(strike).map_err(to_py_err)?.0)
    }

    fn variance(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.variance(strike).map_err(to_py_err)?.0)
    }

    fn density(&self, strike: f64) -> PyResult<f64> {
        self.inner.density(strike).map_err(to_py_err)
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    #[getter]
    fn expiry(&self) -> f64 {
        self.inner.expiry()
    }

    fn is_arbitrage_free(&self) -> PyResult<PyArbitrageReport> {
        Ok(self.inner.is_arbitrage_free().map_err(to_py_err)?.into())
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (s))]
    fn from_json(s: &str) -> PyResult<Self> {
        serde_json::from_str(s)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (strikes))]
    fn vol_array<'py>(
        &self,
        py: Python<'py>,
        strikes: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let stk: Vec<f64> = strikes.as_array().to_vec();
        let inner = &self.inner;
        let data = py.detach(|| {
            stk.iter()
                .map(|&k| inner.vol(k).map(|v| v.0))
                .collect::<Result<Vec<f64>, _>>()
        });
        Ok(data.map_err(to_py_err)?.into_pyarray(py))
    }
}
