use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use volsurf::VolSurfError;

pub(crate) fn to_py_err(e: VolSurfError) -> PyErr {
    match e {
        VolSurfError::InvalidInput { message } => PyValueError::new_err(message),
        VolSurfError::CalibrationError {
            message,
            model,
            rms_error,
        } => {
            let msg = match rms_error {
                Some(rms) => format!("{model}: {message} (rms={rms:.6})"),
                None => format!("{model}: {message}"),
            };
            PyRuntimeError::new_err(msg)
        }
        VolSurfError::NumericalError { message } => PyRuntimeError::new_err(message),
        _ => PyRuntimeError::new_err(format!("{e}")),
    }
}
