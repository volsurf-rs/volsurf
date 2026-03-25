use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volsurf::Strike;
use volsurf::smile::{SabrSmile, SmileSection, SplineSmile, SviSmile};

use crate::error::to_py_err;
use crate::types::{PyArbitrageReport, PyArbitrageScanConfig};

macro_rules! impl_smile_methods {
    ($name:ident) => {
        #[pymethods]
        impl $name {
            fn vol(&self, strike: f64) -> PyResult<f64> {
                Ok(self.inner.vol(Strike(strike)).map_err(to_py_err)?.0)
            }

            fn variance(&self, strike: f64) -> PyResult<f64> {
                Ok(self.inner.variance(Strike(strike)).map_err(to_py_err)?.0)
            }

            fn density(&self, strike: f64) -> PyResult<f64> {
                self.inner.density(Strike(strike)).map_err(to_py_err)
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

            fn is_arbitrage_free_with(
                &self,
                config: &PyArbitrageScanConfig,
            ) -> PyResult<PyArbitrageReport> {
                Ok(self
                    .inner
                    .is_arbitrage_free_with(&config.inner)
                    .map_err(to_py_err)?
                    .into())
            }

            #[getter]
            fn model_name(&self) -> &str {
                self.inner.model_name()
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
                // frozen pyclass: &self is immutable + alive for method duration; inner is pure Rust
                let data = py.detach(|| {
                    stk.iter()
                        .map(|&k| inner.vol(Strike(k)).map(|v| v.0))
                        .collect::<Result<Vec<f64>, _>>()
                });
                Ok(data.map_err(to_py_err)?.into_pyarray(py))
            }
        }
    };
}

#[pyclass(frozen, name = "SviSmile")]
pub struct PySviSmile {
    inner: SviSmile,
}

impl PySviSmile {
    pub(crate) fn from_inner(inner: SviSmile) -> Self {
        Self { inner }
    }
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

    #[staticmethod]
    #[pyo3(signature = (forward, expiry, market_vols))]
    fn calibrate(forward: f64, expiry: f64, market_vols: Vec<(f64, f64)>) -> PyResult<Self> {
        let inner = SviSmile::calibrate(forward, expiry, &market_vols).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (forward, expiry, market_vols, filter=None, weighting=None, seed=None))]
    fn calibrate_with_config(
        forward: f64,
        expiry: f64,
        market_vols: Vec<(f64, f64)>,
        filter: Option<&crate::types::PyDataFilter>,
        weighting: Option<&crate::types::PyWeightingScheme>,
        seed: Option<&PySviSmile>,
    ) -> PyResult<Self> {
        let f = filter.map(|f| f.inner).unwrap_or_default();
        let w = weighting.map(|w| w.inner).unwrap_or_default();
        let inner = SviSmile::calibrate_with_config(
            forward,
            expiry,
            &market_vols,
            &f,
            &w,
            seed.map(|s| &s.inner),
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }
}

impl_smile_methods!(PySviSmile);

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

    #[staticmethod]
    #[pyo3(signature = (forward, expiry, beta, market_vols))]
    fn calibrate(
        forward: f64,
        expiry: f64,
        beta: f64,
        market_vols: Vec<(f64, f64)>,
    ) -> PyResult<Self> {
        let inner = SabrSmile::calibrate(forward, expiry, beta, &market_vols).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (forward, expiry, beta, market_vols, filter=None, weighting=None, seed=None))]
    fn calibrate_with_config(
        forward: f64,
        expiry: f64,
        beta: f64,
        market_vols: Vec<(f64, f64)>,
        filter: Option<&crate::types::PyDataFilter>,
        weighting: Option<&crate::types::PyWeightingScheme>,
        seed: Option<&PySabrSmile>,
    ) -> PyResult<Self> {
        let f = filter.map(|f| f.inner).unwrap_or_default();
        let w = weighting.map(|w| w.inner).unwrap_or_default();
        let inner = SabrSmile::calibrate_with_config(
            forward,
            expiry,
            beta,
            &market_vols,
            &f,
            &w,
            seed.map(|s| &s.inner),
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }
}

impl_smile_methods!(PySabrSmile);

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
}

impl_smile_methods!(PySplineSmile);
