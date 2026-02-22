use std::sync::Arc;

use pyo3::prelude::*;
use volsurf::local_vol::{DupireLocalVol, LocalVol};

use crate::error::to_py_err;
use crate::types::PySurface;

#[pyclass(frozen, name = "DupireLocalVol")]
pub struct PyDupireLocalVol {
    inner: DupireLocalVol,
}

#[pymethods]
impl PyDupireLocalVol {
    #[new]
    #[pyo3(signature = (surface))]
    fn new(surface: &PySurface) -> Self {
        Self {
            inner: DupireLocalVol::new(Arc::clone(&surface.inner)),
        }
    }

    fn local_vol(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        self.inner
            .local_vol(expiry, strike)
            .map_err(to_py_err)
            .map(|v| v.0)
    }
}
