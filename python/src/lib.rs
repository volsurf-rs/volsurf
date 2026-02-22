use pyo3::prelude::*;

mod error;
mod types;

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
    Ok(())
}
