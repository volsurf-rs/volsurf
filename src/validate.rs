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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gap #6: validate edge cases ---

    // validate_positive

    #[test]
    fn positive_accepts_normal_value() {
        assert_eq!(validate_positive(1.0, "x").unwrap(), 1.0);
        assert_eq!(validate_positive(0.001, "x").unwrap(), 0.001);
        assert_eq!(validate_positive(1e300, "x").unwrap(), 1e300);
    }

    #[test]
    fn positive_rejects_zero() {
        assert!(validate_positive(0.0, "x").is_err());
    }

    #[test]
    fn positive_rejects_negative() {
        assert!(validate_positive(-1.0, "x").is_err());
        assert!(validate_positive(-1e-300, "x").is_err());
    }

    #[test]
    fn positive_rejects_nan() {
        assert!(validate_positive(f64::NAN, "x").is_err());
    }

    #[test]
    fn positive_rejects_positive_inf() {
        assert!(validate_positive(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn positive_rejects_negative_inf() {
        assert!(validate_positive(f64::NEG_INFINITY, "x").is_err());
    }

    // validate_non_negative

    #[test]
    fn non_negative_accepts_zero() {
        assert_eq!(validate_non_negative(0.0, "x").unwrap(), 0.0);
    }

    #[test]
    fn non_negative_accepts_positive() {
        assert_eq!(validate_non_negative(1.0, "x").unwrap(), 1.0);
    }

    #[test]
    fn non_negative_rejects_negative() {
        assert!(validate_non_negative(-0.001, "x").is_err());
    }

    #[test]
    fn non_negative_rejects_nan() {
        assert!(validate_non_negative(f64::NAN, "x").is_err());
    }

    #[test]
    fn non_negative_rejects_positive_inf() {
        assert!(validate_non_negative(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn non_negative_rejects_negative_inf() {
        assert!(validate_non_negative(f64::NEG_INFINITY, "x").is_err());
    }

    // validate_finite

    #[test]
    fn finite_accepts_zero() {
        assert_eq!(validate_finite(0.0, "x").unwrap(), 0.0);
    }

    #[test]
    fn finite_accepts_negative() {
        assert_eq!(validate_finite(-100.0, "x").unwrap(), -100.0);
    }

    #[test]
    fn finite_accepts_positive() {
        assert_eq!(validate_finite(1e300, "x").unwrap(), 1e300);
    }

    #[test]
    fn finite_rejects_nan() {
        assert!(validate_finite(f64::NAN, "x").is_err());
    }

    #[test]
    fn finite_rejects_positive_inf() {
        assert!(validate_finite(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn finite_rejects_negative_inf() {
        assert!(validate_finite(f64::NEG_INFINITY, "x").is_err());
    }

    // Error messages include the field name

    #[test]
    fn error_message_includes_field_name() {
        let err = validate_positive(-1.0, "my_field").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("my_field"), "error should include field name: {msg}");
    }
}
