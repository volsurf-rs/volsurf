//! eSSVI (Extended SSVI) surface and slice.
//!
//! eSSVI extends Gatheral-Jacquier's SSVI by making the correlation parameter ρ
//! depend on maturity (through the ATM total variance θ). At the slice level the
//! formula is identical to SSVI — the difference is purely in how ρ is chosen.
//!
//! [`EssviSlice`] is a single-tenor slice implementing [`SmileSection`].
//! [`EssviSurface`] is the multi-tenor surface (stub, implemented in issue #3).
//!
//! # References
//! - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::smile::arbitrage::ArbitrageReport;
use crate::surface::VolSurface;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::ssvi::SsviSlice;
use crate::types::{Variance, Vol};

/// A single-tenor slice through an eSSVI surface.
///
/// Evaluates the eSSVI total variance formula at a fixed ATM total variance θ,
/// providing a [`SmileSection`] interface for surface queries and downstream
/// consumers expecting a per-tenor smile.
///
/// At the slice level eSSVI and SSVI share the same formula — this is a thin
/// newtype over [`SsviSlice`]. The per-tenor ρ(θ) is evaluated by
/// [`EssviSurface`] at construction time and baked into the slice.
///
/// # Formula
///
/// At log-moneyness `k = ln(K/F)`:
///
/// ```text
/// w(k, θ) = (θ/2) · [1 + ρ(θ)·φ(θ)·k + √((φ(θ)·k + ρ(θ))² + (1 − ρ(θ)²))]
/// ```
///
/// where `φ(θ) = η / θ^γ` and `ρ(θ)` is a parametric family managed by
/// [`EssviSurface`].
///
/// # References
/// - Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019), Eq. 2.2
/// - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "EssviSliceRaw", into = "EssviSliceRaw")]
pub struct EssviSlice(SsviSlice);

#[derive(Serialize, Deserialize)]
struct EssviSliceRaw {
    forward: f64,
    expiry: f64,
    rho: f64,
    eta: f64,
    gamma: f64,
    theta: f64,
}

impl TryFrom<EssviSliceRaw> for EssviSlice {
    type Error = VolSurfError;
    fn try_from(raw: EssviSliceRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.forward,
            raw.expiry,
            raw.rho,
            raw.eta,
            raw.gamma,
            raw.theta,
        )
    }
}

impl From<EssviSlice> for EssviSliceRaw {
    fn from(s: EssviSlice) -> Self {
        Self {
            forward: s.0.forward(),
            expiry: s.0.expiry(),
            rho: s.rho(),
            eta: s.eta(),
            gamma: s.gamma(),
            theta: s.0.theta(),
        }
    }
}

impl EssviSlice {
    /// Create an eSSVI slice at a fixed tenor.
    ///
    /// The `rho` parameter is the maturity-dependent correlation ρ(θ),
    /// typically computed by [`EssviSurface`] from its parametric family.
    /// All validation is delegated to [`SsviSlice::new()`].
    ///
    /// # Arguments
    /// * `forward` — Forward price, must be positive
    /// * `expiry` — Time to expiry in years, must be positive
    /// * `rho` — Maturity-dependent skew ρ(θ), must be in (−1, 1)
    /// * `eta` — Smile amplitude, must be positive
    /// * `gamma` — Term structure decay, must be in \[0, 1\]
    /// * `theta` — ATM total variance σ²T, must be positive
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use volsurf::surface::EssviSlice;
    /// use volsurf::SmileSection;
    ///
    /// let slice = EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.04)?;
    /// let vol = slice.vol(100.0)?;
    /// assert!((vol.0 - 0.20).abs() < 0.01);
    /// # Ok::<(), volsurf::VolSurfError>(())
    /// ```
    pub fn new(
        forward: f64,
        expiry: f64,
        rho: f64,
        eta: f64,
        gamma: f64,
        theta: f64,
    ) -> error::Result<Self> {
        SsviSlice::new(forward, expiry, rho, eta, gamma, theta).map(EssviSlice)
    }

    /// ATM total variance θ at this tenor.
    pub fn theta(&self) -> f64 {
        self.0.theta()
    }

    /// Skew parameter ρ(θ) at this tenor.
    pub fn rho(&self) -> f64 {
        self.0.rho()
    }

    /// Smile amplitude η.
    pub fn eta(&self) -> f64 {
        self.0.eta()
    }

    /// Term structure decay γ.
    pub fn gamma(&self) -> f64 {
        self.0.gamma()
    }
}

impl SmileSection for EssviSlice {
    fn vol(&self, strike: f64) -> error::Result<Vol> {
        self.0.vol(strike)
    }

    fn variance(&self, strike: f64) -> error::Result<Variance> {
        self.0.variance(strike)
    }

    fn forward(&self) -> f64 {
        self.0.forward()
    }

    fn expiry(&self) -> f64 {
        self.0.expiry()
    }

    fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
        self.0.is_arbitrage_free()
    }
}

/// Extended SSVI surface with calendar-spread no-arbitrage guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssviSurface {
    _placeholder: (),
}

impl EssviSurface {
    /// Calibrate an eSSVI surface from market data.
    ///
    /// # Errors
    /// Returns [`VolSurfError::CalibrationError`] if calibration fails.
    pub fn calibrate(
        _market_data: &[Vec<(f64, f64)>],
        _tenors: &[f64],
        _forwards: &[f64],
    ) -> error::Result<Self> {
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }
}

impl VolSurface for EssviSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> error::Result<Variance> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        let _ = expiry;
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        Err(VolSurfError::NumericalError {
            message: "not yet implemented".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn equity_slice() -> EssviSlice {
        EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap()
    }

    #[test]
    fn new_valid_params() {
        let s = equity_slice();
        assert_eq!(s.forward(), 100.0);
        assert_eq!(s.expiry(), 1.0);
        assert_eq!(s.theta(), 0.16);
        assert_eq!(s.rho(), -0.3);
        assert_eq!(s.eta(), 0.5);
        assert_eq!(s.gamma(), 0.5);
    }

    #[test]
    fn new_rejects_invalid_params() {
        assert!(EssviSlice::new(0.0, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(-1.0, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 0.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, -1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, 1.0, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -1.0, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.0, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, -0.1, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, -0.1, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 1.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.0).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, -0.01).is_err());
    }

    #[test]
    fn new_rejects_nan_and_inf() {
        assert!(EssviSlice::new(f64::NAN, 1.0, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, f64::INFINITY, -0.3, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, f64::NAN, 0.5, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, f64::NEG_INFINITY, 0.5, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, f64::NAN, 0.16).is_err());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, f64::INFINITY).is_err());
    }

    #[test]
    fn new_gamma_boundaries_accepted() {
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 0.0, 0.16).is_ok());
        assert!(EssviSlice::new(100.0, 1.0, -0.3, 0.5, 1.0, 0.16).is_ok());
    }

    #[test]
    fn vol_atm_equals_sqrt_theta_over_t() {
        // ATM: k = 0, w(0) = theta, vol = sqrt(theta / T) = sqrt(0.16) = 0.4
        let s = equity_slice();
        let vol = s.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn variance_atm_equals_theta() {
        let s = equity_slice();
        let var = s.variance(100.0).unwrap();
        assert_abs_diff_eq!(var.0, 0.16, epsilon = 1e-14);
    }

    #[test]
    fn vol_variance_consistency() {
        let s = equity_slice();
        for &strike in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = s.vol(strike).unwrap();
            let var = s.variance(strike).unwrap();
            assert_abs_diff_eq!(var.0, vol.0 * vol.0 * s.expiry(), epsilon = 1e-14);
        }
    }

    #[test]
    fn matches_ssvi_slice() {
        // Same params should produce identical vols — the newtype is transparent.
        let essvi = equity_slice();
        let ssvi = SsviSlice::new(100.0, 1.0, -0.3, 0.5, 0.5, 0.16).unwrap();
        for &strike in &[70.0, 85.0, 100.0, 115.0, 130.0] {
            let v_essvi = essvi.vol(strike).unwrap().0;
            let v_ssvi = ssvi.vol(strike).unwrap().0;
            assert_eq!(v_essvi.to_bits(), v_ssvi.to_bits(), "strike={strike}");
        }
    }

    #[test]
    fn vol_rejects_non_positive_strikes() {
        let s = equity_slice();
        assert!(s.vol(0.0).is_err());
        assert!(s.vol(-100.0).is_err());
        assert!(s.variance(0.0).is_err());
    }

    #[test]
    fn skew_direction() {
        let s = equity_slice();
        let vol_put = s.vol(80.0).unwrap().0;
        let vol_call = s.vol(120.0).unwrap().0;
        assert!(vol_put > vol_call, "negative rho should produce put skew");
    }

    #[test]
    fn density_positive_near_atm() {
        let s = equity_slice();
        for &strike in &[90.0, 95.0, 100.0, 105.0, 110.0] {
            let d = s.density(strike).unwrap();
            assert!(d > 0.0, "density({strike}) = {d}");
        }
    }

    #[test]
    fn arb_free_conservative_params() {
        let s = equity_slice();
        let report = s.is_arbitrage_free().unwrap();
        assert!(report.is_free);
        assert!(report.butterfly_violations.is_empty());
    }

    #[test]
    fn arb_detected_extreme_params() {
        // eta * (1 + |rho|) = 3.0 * 1.95 = 5.85 >> 2
        let s = EssviSlice::new(100.0, 1.0, -0.95, 3.0, 0.5, 0.16).unwrap();
        let report = s.is_arbitrage_free().unwrap();
        assert!(!report.is_free);
        assert!(!report.butterfly_violations.is_empty());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EssviSlice>();
    }

    #[test]
    fn serde_round_trip() {
        let s = equity_slice();
        let json = serde_json::to_string(&s).unwrap();
        let s2: EssviSlice = serde_json::from_str(&json).unwrap();
        assert_eq!(s.forward(), s2.forward());
        assert_eq!(s.expiry(), s2.expiry());
        assert_eq!(s.theta(), s2.theta());
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.eta(), s2.eta());
        assert_eq!(s.gamma(), s2.gamma());
        assert_abs_diff_eq!(
            s.vol(90.0).unwrap().0,
            s2.vol(90.0).unwrap().0,
            epsilon = 1e-14
        );
    }

    #[test]
    fn serde_rejects_invalid_rho() {
        let json = r#"{"forward":100,"expiry":1,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_forward() {
        let json = r#"{"forward":-100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_expiry() {
        let json = r#"{"forward":100,"expiry":0,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_negative_eta() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":-0.1,"gamma":0.5,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_gamma_out_of_range() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":2.0,"theta":0.04}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_rejects_zero_theta() {
        let json = r#"{"forward":100,"expiry":1,"rho":-0.3,"eta":0.5,"gamma":0.5,"theta":0.0}"#;
        assert!(serde_json::from_str::<EssviSlice>(json).is_err());
    }

    #[test]
    fn serde_error_contains_validation_message() {
        let json = r#"{"forward":100,"expiry":1,"rho":1.0,"eta":0.5,"gamma":0.5,"theta":0.04}"#;
        let err = serde_json::from_str::<EssviSlice>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rho"),
            "should contain domain message, got: {msg}"
        );
    }

    #[test]
    fn short_expiry_slice() {
        // 1-week tenor: T=7/365, theta=0.04^2 * 7/365 ≈ 3.07e-5
        let t = 7.0 / 365.0;
        let theta = 0.04 * t;
        let s = EssviSlice::new(100.0, t, -0.5, 0.8, 0.4, theta).unwrap();
        let vol = s.vol(100.0).unwrap();
        assert!(vol.0 > 0.0 && vol.0 < 2.0);
    }

    #[test]
    fn long_expiry_slice() {
        // 5-year tenor
        let s = EssviSlice::new(100.0, 5.0, -0.2, 0.3, 0.6, 0.50).unwrap();
        let vol = s.vol(100.0).unwrap();
        assert_abs_diff_eq!(vol.0, (0.50 / 5.0_f64).sqrt(), epsilon = 1e-10);
    }
}
