use volsurf::VolSurfError;
use wasm_bindgen::JsValue;

pub(crate) fn to_js_err(e: VolSurfError) -> JsValue {
    match e {
        VolSurfError::CalibrationError {
            message,
            model,
            rms_error,
        } => match rms_error {
            Some(rms) => JsValue::from_str(&format!("{model}: {message} (rms={rms:.6})")),
            None => JsValue::from_str(&format!("{model}: {message}")),
        },
        _ => JsValue::from_str(&format!("{e}")),
    }
}
