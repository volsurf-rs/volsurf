//! Input validation helpers.
//!
//! Standardizes validation across the crate using `!is_finite()` to reject
//! NaN, +Inf, and -Inf uniformly.

use crate::error::VolSurfError;

/// Validate that a value is strictly positive and finite (rejects NaN, Inf, zero, negatives).
pub(crate) fn validate_positive(value: f64, name: &str) -> crate::error::Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(VolSurfError::InvalidInput {
            message: format!("{name} must be positive and finite, got {value}"),
        });
    }
    Ok(value)
}

/// Validate that a value is non-negative and finite (rejects NaN, Inf, negatives).
pub(crate) fn validate_non_negative(value: f64, name: &str) -> crate::error::Result<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(VolSurfError::InvalidInput {
            message: format!("{name} must be non-negative and finite, got {value}"),
        });
    }
    Ok(value)
}

/// Validate that a value is finite (rejects NaN and Inf; allows zero and negatives).
pub(crate) fn validate_finite(value: f64, name: &str) -> crate::error::Result<f64> {
    if !value.is_finite() {
        return Err(VolSurfError::InvalidInput {
            message: format!("{name} must be finite, got {value}"),
        });
    }
    Ok(value)
}
