use std::sync::Arc;

use pyo3::prelude::*;
use volsurf::local_vol::{DupireLocalVol, LocalVol};
use volsurf::{Strike, Tenor};

use crate::error::to_py_err;
use crate::types::PySurface;

#[pyclass(frozen, name = "DupireLocalVol")]
pub struct PyDupireLocalVol {
    inner: DupireLocalVol,
}

#[pymethods]
impl PyDupireLocalVol {
    #[new]
    #[pyo3(signature = (surface, bump_size=None))]
    fn new(surface: &PySurface, bump_size: Option<f64>) -> PyResult<Self> {
        let mut lv = DupireLocalVol::new(Arc::clone(&surface.inner));
        if let Some(h) = bump_size {
            lv = lv.with_bump_size(h).map_err(to_py_err)?;
        }
        Ok(Self { inner: lv })
    }

    fn local_vol(&self, expiry: f64, strike: f64) -> PyResult<f64> {
        self.inner
            .local_vol(Tenor(expiry), Strike(strike))
            .map_err(to_py_err)
            .map(|v| v.0)
    }
}
