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

/// Validate every element of a named collection as positive and finite.
///
/// The error format intentionally matches the surface constructors' historic
/// messages because those diagnostics are part of the crate's observable API.
pub(crate) fn validate_positive_slice(values: &[f64], name: &str) -> crate::error::Result<()> {
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(VolSurfError::InvalidInput {
                message: format!("{name} must be positive and finite, got {name}[{i}]={value}"),
            });
        }
    }
    Ok(())
}

/// Validate that a named collection is strictly increasing.
pub(crate) fn validate_strictly_increasing(values: &[f64], name: &str) -> crate::error::Result<()> {
    for pair in values.windows(2) {
        if pair[1] <= pair[0] {
            return Err(VolSurfError::InvalidInput {
                message: format!(
                    "{name} must be strictly increasing, but {} >= {}",
                    pair[0], pair[1]
                ),
            });
        }
    }
    Ok(())
}

/// Validate the common tenor/forward/theta grid used by parametric surfaces.
pub(crate) fn validate_surface_grid(
    tenors: &[f64],
    forwards: &[f64],
    thetas: &[f64],
) -> crate::error::Result<()> {
    if tenors.is_empty() {
        return Err(VolSurfError::InvalidInput {
            message: "at least one tenor is required".into(),
        });
    }
    if tenors.len() != forwards.len() {
        return Err(VolSurfError::InvalidInput {
            message: format!(
                "tenors and forwards must have the same length, got {} and {}",
                tenors.len(),
                forwards.len()
            ),
        });
    }
    if tenors.len() != thetas.len() {
        return Err(VolSurfError::InvalidInput {
            message: format!(
                "tenors and thetas must have the same length, got {} and {}",
                tenors.len(),
                thetas.len()
            ),
        });
    }

    validate_positive_slice(tenors, "tenors")?;
    validate_positive_slice(forwards, "forwards")?;
    validate_positive_slice(thetas, "thetas")?;
    validate_strictly_increasing(tenors, "tenors")?;
    validate_strictly_increasing(thetas, "thetas")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gap #6: validate edge cases

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
        assert!(
            msg.contains("my_field"),
            "error should include field name: {msg}"
        );
    }

    #[test]
    fn surface_grid_preserves_indexed_validation_messages() {
        let err = validate_surface_grid(&[0.5], &[f64::NAN], &[0.04]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid input: forwards must be positive and finite, got forwards[0]=NaN"
        );
    }

    #[test]
    fn strictly_increasing_reports_adjacent_values() {
        let err = validate_strictly_increasing(&[0.5, 0.5], "tenors").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid input: tenors must be strictly increasing, but 0.5 >= 0.5"
        );
    }
}
