use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use volsurf::local_vol::{DupireLocalVol, LocalVol};
use volsurf::{Strike, Tenor, VolSurface};

use crate::error::to_py_err;
use crate::surface::{PyEssviSurface, PySsviSurface};
use crate::types::PySurface;

#[pyclass(frozen, name = "DupireLocalVol")]
pub struct PyDupireLocalVol {
    inner: DupireLocalVol,
}

#[pymethods]
impl PyDupireLocalVol {
    #[new]
    #[pyo3(signature = (surface, bump_size=None))]
    fn new(surface: &Bound<'_, PyAny>, bump_size: Option<f64>) -> PyResult<Self> {
        let surface: Arc<dyn VolSurface> =
            if let Ok(surface) = surface.extract::<PyRef<PySurface>>() {
                Arc::clone(&surface.inner)
            } else if let Ok(surface) = surface.extract::<PyRef<PySsviSurface>>() {
                Arc::new(surface.inner.clone())
            } else if let Ok(surface) = surface.extract::<PyRef<PyEssviSurface>>() {
                Arc::new(surface.inner.clone())
            } else {
                return Err(PyTypeError::new_err(
                    "surface must be a Surface, SsviSurface, or EssviSurface",
                ));
            };

        let mut lv = DupireLocalVol::new(surface);
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
