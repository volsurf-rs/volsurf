//! Error types for the volsurf library.
//!
//! All fallible operations return `Result<T, VolSurfError>` rather than panicking,
//! providing meaningful diagnostics for calibration failures, invalid inputs, and
//! numerical issues.

use thiserror::Error;

/// Convenience type alias for results in this crate.
pub type Result<T> = std::result::Result<T, VolSurfError>;

/// Errors that can occur during volatility surface construction and queries.
///
/// This enum is `#[non_exhaustive]`: new variants may be added in minor
/// releases. Match statements should include a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VolSurfError {
    /// Smile or surface calibration failed to converge.
    #[error("calibration failed: {message}")]
    CalibrationError {
        message: String,
        /// Model that failed (e.g., "SVI", "SABR").
        model: &'static str,
        /// Final RMS error, if the optimizer ran.
        ///
        /// `Some(rms)` — the optimizer converged but the result was invalid or
        /// the error exceeded the tolerance. The value indicates how close the
        /// fit was; adjusting initial parameters may help.
        ///
        /// `None` — a structural failure (no valid grid point, singular matrix,
        /// failed alpha solve) prevented the optimizer from running. Retrying
        /// with the same data will not help.
        ///
        /// Units: vol space for SABR, total variance for SSVI and eSSVI.
        rms_error: Option<f64>,
    },

    /// Input data is invalid (e.g., negative vol, zero expiry, mismatched grid sizes).
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    /// Numerical computation failed (e.g., NaN, ill-conditioned matrix).
    #[error("numerical error: {message}")]
    NumericalError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gap #2: Structured error field access

    #[test]
    fn calibration_error_fields_accessible() {
        let err = VolSurfError::CalibrationError {
            message: "convergence failed".into(),
            model: "SVI",
            rms_error: Some(0.05),
        };
        match &err {
            VolSurfError::CalibrationError {
                message,
                model,
                rms_error,
            } => {
                assert_eq!(message, "convergence failed");
                assert_eq!(*model, "SVI");
                assert_eq!(*rms_error, Some(0.05));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn calibration_error_rms_none() {
        let err = VolSurfError::CalibrationError {
            message: "no grid point found".into(),
            model: "SVI",
            rms_error: None,
        };
        match &err {
            VolSurfError::CalibrationError { rms_error, .. } => {
                assert!(rms_error.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn invalid_input_message_accessible() {
        let err = VolSurfError::InvalidInput {
            message: "strike must be positive".into(),
        };
        match &err {
            VolSurfError::InvalidInput { message } => {
                assert!(message.contains("positive"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn error_display_includes_message() {
        let err = VolSurfError::CalibrationError {
            message: "test failure".into(),
            model: "SVI",
            rms_error: Some(0.01),
        };
        let display = format!("{err}");
        assert!(display.contains("test failure"));

        let err2 = VolSurfError::InvalidInput {
            message: "bad input".into(),
        };
        assert!(format!("{err2}").contains("bad input"));

        let err3 = VolSurfError::NumericalError {
            message: "NaN detected".into(),
        };
        assert!(format!("{err3}").contains("NaN detected"));
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VolSurfError>();
    }
}
