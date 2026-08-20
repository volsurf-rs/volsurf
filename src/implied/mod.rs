//! Implied volatility extraction from option prices.
//!
//! Provides three models for extracting implied volatility:
//!
//! - [`BlackImpliedVol`] — Standard Black (lognormal) model via Jäckel's algorithm
//! - [`NormalImpliedVol`] — Bachelier (normal) model for fixed income / short-dated FX
//! - [`DisplacedImpliedVol`] — Displaced diffusion hybrid (interpolates normal ↔ Black)

pub mod black;
pub mod displaced;
pub mod normal;

pub use black::{BlackImpliedVol, black_price};
pub use displaced::{DisplacedImpliedVol, displaced_price};
pub use normal::{NormalImpliedVol, normal_price};

use crate::error::{self, VolSurfError};
use crate::types::OptionType;
use crate::validate::{validate_finite, validate_non_negative, validate_positive};

#[derive(Clone, Copy)]
pub(crate) enum PriceDomain {
    Positive,
    Finite,
}

pub(crate) fn validate_implied_inputs(
    option_price: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    domain: PriceDomain,
) -> error::Result<()> {
    validate_non_negative(option_price, "option_price")?;
    validate_forward_strike(forward, strike, domain)?;
    validate_positive(expiry, "expiry")?;
    Ok(())
}

pub(crate) fn validate_pricing_inputs(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    domain: PriceDomain,
) -> error::Result<()> {
    validate_forward_strike(forward, strike, domain)?;
    validate_non_negative(vol, "volatility")?;
    validate_non_negative(expiry, "expiry")?;
    Ok(())
}

fn validate_forward_strike(forward: f64, strike: f64, domain: PriceDomain) -> error::Result<()> {
    match domain {
        PriceDomain::Positive => {
            validate_positive(forward, "forward")?;
            validate_positive(strike, "strike")?;
        }
        PriceDomain::Finite => {
            validate_finite(forward, "forward")?;
            validate_finite(strike, "strike")?;
        }
    }
    Ok(())
}

pub(crate) fn is_call(option_type: OptionType) -> bool {
    matches!(option_type, OptionType::Call)
}

pub(crate) fn finish_implied<T, F>(built: Option<T>, calculate: F) -> error::Result<f64>
where
    F: FnOnce(T) -> Option<f64>,
{
    let calculator = built.ok_or_else(|| VolSurfError::InvalidInput {
        message: "implied-vol rejected inputs as outside model domain".into(),
    })?;
    calculate(calculator).ok_or_else(|| VolSurfError::NumericalError {
        message: "option price is outside the attainable range".into(),
    })
}

pub(crate) fn finish_price<T, F>(built: Option<T>, calculate: F) -> error::Result<f64>
where
    F: FnOnce(T) -> f64,
{
    let calculator = built.ok_or_else(|| VolSurfError::InvalidInput {
        message: "implied-vol rejected pricing inputs as outside model domain".into(),
    })?;
    Ok(calculate(calculator))
}
