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
    #[error("calibration failed: {0}")]
    CalibrationError(String),

    /// Input data is invalid (e.g., negative vol, zero expiry, mismatched grid sizes).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Numerical computation failed (e.g., NaN, ill-conditioned matrix).
    #[error("numerical error: {0}")]
    NumericalError(String),

    /// Arbitrage violation detected in the surface.
    #[error("arbitrage detected: {0}")]
    ArbitrageViolation(String),
}
