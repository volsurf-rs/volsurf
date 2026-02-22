use pyo3::prelude::*;

mod error;
mod implied;
mod local_vol;
mod smile;
mod surface;
mod types;

use local_vol::*;
use smile::*;
use surface::*;
use types::*;

#[pymodule]
fn volsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmile>()?;
    m.add_class::<PySurface>()?;
    m.add_class::<PySmileModel>()?;
    m.add_class::<PyOptionType>()?;
    m.add_class::<PyStickyKind>()?;
    m.add_class::<PyArbitrageReport>()?;
    m.add_class::<PyButterflyViolation>()?;
    m.add_class::<PyCalendarViolation>()?;
    m.add_class::<PySurfaceDiagnostics>()?;
    m.add_class::<PySviSmile>()?;
    m.add_class::<PySabrSmile>()?;
    m.add_class::<PySplineSmile>()?;
    m.add_function(wrap_pyfunction!(implied::black_price, m)?)?;
    m.add_function(wrap_pyfunction!(implied::normal_price, m)?)?;
    m.add_function(wrap_pyfunction!(implied::displaced_price, m)?)?;
    m.add_function(wrap_pyfunction!(implied::log_moneyness, m)?)?;
    m.add_function(wrap_pyfunction!(implied::moneyness, m)?)?;
    m.add_function(wrap_pyfunction!(implied::forward_price, m)?)?;
    m.add_class::<implied::PyBlackImpliedVol>()?;
    m.add_class::<implied::PyNormalImpliedVol>()?;
    m.add_class::<implied::PyDisplacedImpliedVol>()?;
    m.add_class::<PySsviSurface>()?;
    m.add_class::<PyEssviSurface>()?;
    m.add_class::<PySurfaceBuilder>()?;
    m.add_class::<PyDupireLocalVol>()?;
    Ok(())
}
