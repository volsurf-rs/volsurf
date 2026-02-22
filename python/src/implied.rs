use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::types::PyOptionType;

#[pyfunction]
#[pyo3(signature = (forward, strike, vol, expiry, option_type))]
pub fn black_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: PyOptionType,
) -> PyResult<f64> {
    volsurf::implied::black_price(forward, strike, vol, expiry, option_type.into())
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (forward, strike, vol, expiry, option_type))]
pub fn normal_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    option_type: PyOptionType,
) -> PyResult<f64> {
    volsurf::implied::normal_price(forward, strike, vol, expiry, option_type.into())
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (forward, strike, vol, expiry, beta, option_type))]
pub fn displaced_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    beta: f64,
    option_type: PyOptionType,
) -> PyResult<f64> {
    volsurf::implied::displaced_price(forward, strike, vol, expiry, beta, option_type.into())
        .map_err(to_py_err)
}

#[pyclass(frozen, name = "BlackImpliedVol")]
pub struct PyBlackImpliedVol;

#[pymethods]
impl PyBlackImpliedVol {
    #[staticmethod]
    #[pyo3(signature = (option_price, forward, strike, expiry, option_type))]
    fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: PyOptionType,
    ) -> PyResult<f64> {
        volsurf::implied::BlackImpliedVol::compute(
            option_price,
            forward,
            strike,
            expiry,
            option_type.into(),
        )
        .map(|v| v.0)
        .map_err(to_py_err)
    }
}

#[pyclass(frozen, name = "NormalImpliedVol")]
pub struct PyNormalImpliedVol;

#[pymethods]
impl PyNormalImpliedVol {
    #[staticmethod]
    #[pyo3(signature = (option_price, forward, strike, expiry, option_type))]
    fn compute(
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: PyOptionType,
    ) -> PyResult<f64> {
        volsurf::implied::NormalImpliedVol::compute(
            option_price,
            forward,
            strike,
            expiry,
            option_type.into(),
        )
        .map(|v| v.0)
        .map_err(to_py_err)
    }
}

#[pyclass(frozen, name = "DisplacedImpliedVol")]
pub struct PyDisplacedImpliedVol {
    inner: volsurf::implied::DisplacedImpliedVol,
}

#[pymethods]
impl PyDisplacedImpliedVol {
    #[new]
    #[pyo3(signature = (beta))]
    fn new(beta: f64) -> PyResult<Self> {
        let inner = volsurf::implied::DisplacedImpliedVol::new(beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    #[pyo3(signature = (option_price, forward, strike, expiry, option_type))]
    fn compute(
        &self,
        option_price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
        option_type: PyOptionType,
    ) -> PyResult<f64> {
        self.inner
            .compute(option_price, forward, strike, expiry, option_type.into())
            .map(|v| v.0)
            .map_err(to_py_err)
    }
}

#[pyfunction]
#[pyo3(signature = (strike, forward))]
pub fn log_moneyness(strike: f64, forward: f64) -> PyResult<f64> {
    volsurf::conventions::log_moneyness(strike, forward).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (strike, forward))]
pub fn moneyness(strike: f64, forward: f64) -> PyResult<f64> {
    volsurf::conventions::moneyness(strike, forward).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (spot, rate, dividend_yield, expiry))]
pub fn forward_price(spot: f64, rate: f64, dividend_yield: f64, expiry: f64) -> PyResult<f64> {
    volsurf::conventions::forward_price(spot, rate, dividend_yield, expiry).map_err(to_py_err)
}
