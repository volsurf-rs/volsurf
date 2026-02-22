use pyo3::prelude::*;

#[pymodule]
fn volsurf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
