//! Error types for the volsurf library.
//!
//! All fallible operations return `Result<T, VolSurfError>` rather than panicking,
//! providing meaningful diagnostics for calibration failures, invalid inputs, and
//! numerical issues.

use thiserror::Error;

/// Convenience type alias for results in this crate.
pub type Result<T> = std::result::Result<T, VolSurfError>;

/// Errors that can occur during volatility surface construction and queries.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VolSurfError {
    /// Smile or surface calibration failed to converge.
    #[error("calibration failed: {message}")]
    CalibrationError {
        message: String,
        /// Model that failed (e.g., "SVI", "SABR").
        model: &'static str,
        /// Final RMS error at convergence, if available.
        rms_error: Option<f64>,
    },

    /// Input data is invalid (e.g., negative vol, zero expiry, mismatched grid sizes).
    #[error("invalid input: {message}")]
    InvalidInput {
        message: String,
    },

    /// Numerical computation failed (e.g., NaN, ill-conditioned matrix).
    #[error("numerical error: {message}")]
    NumericalError {
        message: String,
    },

    /// Arbitrage violation detected in the surface.
    #[error("arbitrage detected: {message}")]
    ArbitrageViolation {
        message: String,
    },
}
