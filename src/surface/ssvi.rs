//! SSVI (Surface SVI) global parameterization.
//!
//! SSVI parameterizes the entire volatility surface (not just a single tenor)
//! as a function of total variance at-the-money and log-moneyness. The core
//! formula for total implied variance is:
//!
//! ```text
//! w(k, θ) = (θ/2) · [1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²))]
//! ```
//!
//! where `φ(θ) = η / θ^γ` is the power-law mixing function.
//!
//! The three global parameters `(ρ, η, γ)` control:
//! - `ρ ∈ (−1, 1)`: skew direction and magnitude
//! - `η > 0`: smile amplitude
//! - `γ ∈ [0, 1]`: term structure decay of the smile
//!
//! Per-tenor ATM total variances `θ_i = σ²_ATM,i · T_i` anchor the surface
//! to market-observed ATM levels.
//!
//! # No-Arbitrage Conditions
//!
//! The power-law φ satisfies the sufficient condition for no butterfly arbitrage
//! (Theorem 4.1) when `η·(1 + |ρ|) ≤ 2`.
//!
//! # References
//! - Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)

use serde::{Deserialize, Serialize};

use crate::error::{self, VolSurfError};
use crate::smile::SmileSection;
use crate::surface::arbitrage::SurfaceDiagnostics;
use crate::surface::VolSurface;
use crate::types::{Variance, Vol};
use crate::validate::validate_positive;

/// SSVI volatility surface parameterized by Gatheral-Jacquier (2014).
///
/// Stores three global parameters `(ρ, η, γ)` plus per-tenor data
/// `(T_i, F_i, θ_i)`. The surface is fully determined by these values
/// and can evaluate the SSVI total variance formula at any `(T, K)`.
///
/// # Construction
///
/// ```
/// use volsurf::surface::SsviSurface;
///
/// let surface = SsviSurface::new(
///     -0.3,                              // rho: skew
///     0.5,                               // eta: smile amplitude
///     0.5,                               // gamma: term decay
///     vec![0.25, 0.5, 1.0],             // tenors (years)
///     vec![100.0, 100.0, 100.0],        // forward prices
///     vec![0.04, 0.08, 0.16],           // ATM total variances (theta)
/// ).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsviSurface {
    /// Global skew parameter ρ ∈ (−1, 1).
    rho: f64,
    /// Smile amplitude parameter η > 0.
    eta: f64,
    /// Term structure decay parameter γ ∈ [0, 1].
    gamma: f64,
    /// Per-tenor expiries, strictly increasing and positive.
    tenors: Vec<f64>,
    /// Forward prices at each tenor, all positive.
    forwards: Vec<f64>,
    /// ATM total variance θ_i = σ²_ATM,i · T_i at each tenor, strictly increasing.
    thetas: Vec<f64>,
    /// Precomputed 1 − ρ² for the hot path.
    one_minus_rho_sq: f64,
}

impl SsviSurface {
    /// Create an SSVI surface from global parameters and per-tenor data.
    ///
    /// # Arguments
    /// * `rho` — Global skew, must be in (−1, 1)
    /// * `eta` — Smile amplitude, must be positive
    /// * `gamma` — Term structure decay, must be in \[0, 1\]
    /// * `tenors` — Expiries in years, strictly increasing, all positive
    /// * `forwards` — Forward prices at each tenor, all positive
    /// * `thetas` — ATM total variances θ_i, strictly increasing, all positive
    ///
    /// # Errors
    /// Returns [`VolSurfError::InvalidInput`] if any parameter is invalid.
    pub fn new(
        rho: f64,
        eta: f64,
        gamma: f64,
        tenors: Vec<f64>,
        forwards: Vec<f64>,
        thetas: Vec<f64>,
    ) -> error::Result<Self> {
        // --- Scalar validation ---
        if rho.abs() >= 1.0 || rho.is_nan() {
            return Err(VolSurfError::InvalidInput {
                message: format!("|rho| must be less than 1, got {rho}"),
            });
        }
        validate_positive(eta, "eta")?;
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(VolSurfError::InvalidInput {
                message: format!("gamma must be in [0, 1], got {gamma}"),
            });
        }

        // --- Vector length checks ---
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

        // --- Element-wise validation ---
        for (i, &t) in tenors.iter().enumerate() {
            if !t.is_finite() || t <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "tenors must be positive and finite, got tenors[{i}]={t}"
                    ),
                });
            }
        }
        for (i, &f) in forwards.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "forwards must be positive and finite, got forwards[{i}]={f}"
                    ),
                });
            }
        }
        for (i, &th) in thetas.iter().enumerate() {
            if !th.is_finite() || th <= 0.0 {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "thetas must be positive and finite, got thetas[{i}]={th}"
                    ),
                });
            }
        }

        // --- Monotonicity checks ---
        for w in tenors.windows(2) {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "tenors must be strictly increasing, but {} >= {}",
                        w[0], w[1]
                    ),
                });
            }
        }
        for w in thetas.windows(2) {
            if w[1] <= w[0] {
                return Err(VolSurfError::InvalidInput {
                    message: format!(
                        "thetas must be strictly increasing, but {} >= {}",
                        w[0], w[1]
                    ),
                });
            }
        }

        Ok(Self {
            one_minus_rho_sq: 1.0 - rho * rho,
            rho,
            eta,
            gamma,
            tenors,
            forwards,
            thetas,
        })
    }

    /// Global skew parameter ρ.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Smile amplitude parameter η.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Term structure decay parameter γ.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Per-tenor expiries.
    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    /// Forward prices at each tenor.
    pub fn forwards(&self) -> &[f64] {
        &self.forwards
    }

    /// ATM total variances θ_i at each tenor.
    pub fn thetas(&self) -> &[f64] {
        &self.thetas
    }

    /// Evaluate the SSVI total variance formula at `(θ, k)`.
    ///
    /// ```text
    /// w(k, θ) = (θ/2) · [1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²))]
    /// ```
    ///
    /// where `φ(θ) = η / θ^γ`.
    ///
    /// This is the hot path — no allocations, no branching.
    #[allow(dead_code)] // Used by T07 (SsviSlice) and T08 (VolSurface impl)
    pub(crate) fn total_variance_at(&self, theta: f64, k: f64) -> f64 {
        let phi = self.eta / theta.powf(self.gamma);
        let phi_k = phi * k;
        (theta / 2.0)
            * (1.0 + self.rho * phi_k
                + ((phi_k + self.rho).powi(2) + self.one_minus_rho_sq).sqrt())
    }

    /// Evaluate the power-law mixing function φ(θ) = η / θ^γ.
    #[allow(dead_code)] // Used by T07 (SsviSlice)
    pub(crate) fn phi(&self, theta: f64) -> f64 {
        self.eta / theta.powf(self.gamma)
    }

    /// Interpolate `(θ, F)` at an arbitrary expiry.
    ///
    /// - **Exact match** (within 1e-10): uses stored values directly.
    /// - **Between tenors**: linear interpolation in θ (total variance) space,
    ///   linear interpolation for forward price.
    /// - **Before first tenor**: flat vol extrapolation `θ(T) = θ₀ · T/T₀`.
    /// - **After last tenor**: flat vol extrapolation `θ(T) = θₙ · T/Tₙ`.
    #[allow(dead_code)] // Used by T08 (VolSurface impl)
    pub(crate) fn theta_and_forward_at(&self, expiry: f64) -> (f64, f64) {
        let n = self.tenors.len();

        // Check exact match
        for (i, &t) in self.tenors.iter().enumerate() {
            if (expiry - t).abs() < 1e-10 {
                return (self.thetas[i], self.forwards[i]);
            }
        }

        if expiry < self.tenors[0] {
            // Flat vol extrapolation before first tenor
            let theta = self.thetas[0] * expiry / self.tenors[0];
            return (theta, self.forwards[0]);
        }

        if expiry > self.tenors[n - 1] {
            // Flat vol extrapolation after last tenor
            let theta = self.thetas[n - 1] * expiry / self.tenors[n - 1];
            return (theta, self.forwards[n - 1]);
        }

        // Between tenors: linear interpolation
        let right = self.tenors.partition_point(|&t| t < expiry);
        let left = right - 1;
        let alpha = (expiry - self.tenors[left]) / (self.tenors[right] - self.tenors[left]);
        let theta = (1.0 - alpha) * self.thetas[left] + alpha * self.thetas[right];
        let forward = (1.0 - alpha) * self.forwards[left] + alpha * self.forwards[right];
        (theta, forward)
    }

    /// Calibrate an SSVI surface from market data.
    ///
    /// # Errors
    /// Returns [`VolSurfError::CalibrationError`] if calibration fails.
    #[allow(dead_code)]
    pub fn calibrate(
        _market_data: &[Vec<(f64, f64)>],
        _tenors: &[f64],
        _forwards: &[f64],
    ) -> error::Result<Self> {
        Err(VolSurfError::NumericalError {
            message: "SSVI calibration not yet implemented".to_string(),
        })
    }
}

impl VolSurface for SsviSurface {
    fn black_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError {
            message: "SSVI VolSurface not yet implemented (T08)".to_string(),
        })
    }

    fn black_variance(&self, expiry: f64, strike: f64) -> error::Result<Variance> {
        let _ = (expiry, strike);
        Err(VolSurfError::NumericalError {
            message: "SSVI VolSurface not yet implemented (T08)".to_string(),
        })
    }

    fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
        let _ = expiry;
        Err(VolSurfError::NumericalError {
            message: "SSVI VolSurface not yet implemented (T08)".to_string(),
        })
    }

    fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
        Err(VolSurfError::NumericalError {
            message: "SSVI VolSurface not yet implemented (T08)".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Canonical SSVI parameters for a typical equity surface.
    fn equity_surface() -> SsviSurface {
        SsviSurface::new(
            -0.3,                         // rho
            0.5,                          // eta
            0.5,                          // gamma
            vec![0.25, 0.5, 1.0, 2.0],   // tenors
            vec![100.0, 100.0, 100.0, 100.0], // forwards
            vec![0.04, 0.08, 0.16, 0.32], // thetas (σ_ATM=40% flat)
        )
        .unwrap()
    }

    // ========== Constructor validation ==========

    #[test]
    fn new_valid_params() {
        let s = equity_surface();
        assert_eq!(s.rho(), -0.3);
        assert_eq!(s.eta(), 0.5);
        assert_eq!(s.gamma(), 0.5);
        assert_eq!(s.tenors().len(), 4);
        assert_eq!(s.forwards().len(), 4);
        assert_eq!(s.thetas().len(), 4);
    }

    #[test]
    fn new_single_tenor() {
        let s = SsviSurface::new(
            0.0, 1.0, 0.5,
            vec![1.0], vec![100.0], vec![0.04],
        );
        assert!(s.is_ok());
    }

    #[test]
    fn new_rho_at_boundary_rejected() {
        assert!(SsviSurface::new(1.0, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-1.0, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_rho_nan_rejected() {
        assert!(SsviSurface::new(f64::NAN, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_rho_inf_rejected() {
        assert!(SsviSurface::new(f64::INFINITY, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_eta_zero_rejected() {
        assert!(SsviSurface::new(-0.3, 0.0, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_eta_negative_rejected() {
        assert!(SsviSurface::new(-0.3, -0.5, 0.5, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_gamma_out_of_range_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, -0.1, vec![1.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 1.1, vec![1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_gamma_boundaries_accepted() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.0, vec![1.0], vec![100.0], vec![0.04]).is_ok());
        assert!(SsviSurface::new(-0.3, 0.5, 1.0, vec![1.0], vec![100.0], vec![0.04]).is_ok());
    }

    #[test]
    fn new_empty_tenors_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn new_length_mismatch_rejected() {
        // tenors vs forwards mismatch
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![0.5, 1.0], vec![100.0], vec![0.04, 0.08],
        ).is_err());
        // tenors vs thetas mismatch
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![0.5, 1.0], vec![100.0, 100.0], vec![0.04],
        ).is_err());
    }

    #[test]
    fn new_non_positive_tenor_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![0.0], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![-1.0], vec![100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_non_positive_forward_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![0.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![-100.0], vec![0.04]).is_err());
    }

    #[test]
    fn new_non_positive_theta_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.0]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![-0.04]).is_err());
    }

    #[test]
    fn new_tenors_not_increasing_rejected() {
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![1.0, 0.5], vec![100.0, 100.0], vec![0.04, 0.08],
        ).is_err());
        // Equal tenors also rejected
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![1.0, 1.0], vec![100.0, 100.0], vec![0.04, 0.08],
        ).is_err());
    }

    #[test]
    fn new_thetas_not_increasing_rejected() {
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![0.5, 1.0], vec![100.0, 100.0], vec![0.08, 0.04],
        ).is_err());
        // Equal thetas also rejected
        assert!(SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![0.5, 1.0], vec![100.0, 100.0], vec![0.04, 0.04],
        ).is_err());
    }

    #[test]
    fn new_nan_in_vectors_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![f64::NAN], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![f64::NAN], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![f64::NAN]).is_err());
    }

    #[test]
    fn new_inf_in_vectors_rejected() {
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![f64::INFINITY], vec![100.0], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![f64::INFINITY], vec![0.04]).is_err());
        assert!(SsviSurface::new(-0.3, 0.5, 0.5, vec![1.0], vec![100.0], vec![f64::INFINITY]).is_err());
    }

    // ========== SSVI formula ==========

    #[test]
    fn total_variance_atm_equals_theta() {
        // Gatheral-Jacquier (2014): w(0, θ) = θ for any θ.
        let s = equity_surface();
        for &theta in s.thetas() {
            assert_abs_diff_eq!(s.total_variance_at(theta, 0.0), theta, epsilon = 1e-14);
        }
    }

    #[test]
    fn total_variance_symmetric_when_rho_zero() {
        // When ρ = 0, the formula is symmetric in k: w(k, θ) = w(-k, θ).
        let s = SsviSurface::new(
            0.0, 0.5, 0.5,
            vec![1.0], vec![100.0], vec![0.16],
        ).unwrap();
        let theta = 0.16;
        for &k in &[-0.5, -0.2, -0.1, 0.1, 0.2, 0.5] {
            assert_abs_diff_eq!(
                s.total_variance_at(theta, k),
                s.total_variance_at(theta, -k),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn total_variance_positive_for_all_strikes() {
        let s = equity_surface();
        let theta = 0.16;
        for k in (-30..=30).map(|i| i as f64 * 0.1) {
            let w = s.total_variance_at(theta, k);
            assert!(w > 0.0, "w({k}, {theta}) = {w} must be positive");
        }
    }

    #[test]
    fn total_variance_known_value() {
        // Manual computation for rho=-0.3, eta=0.5, gamma=0.5, theta=0.16, k=0.1:
        // phi = 0.5 / 0.16^0.5 = 0.5 / 0.4 = 1.25
        // phi*k = 1.25 * 0.1 = 0.125
        // (phi*k + rho)^2 = (0.125 + (-0.3))^2 = (-0.175)^2 = 0.030625
        // 1 - rho^2 = 1 - 0.09 = 0.91
        // sqrt(0.030625 + 0.91) = sqrt(0.940625) = 0.969858...
        // w = (0.16/2) * (1 + (-0.3)*0.125 + 0.969858...)
        //   = 0.08 * (1 - 0.0375 + 0.969858...)
        //   = 0.08 * 1.932358...
        //   = 0.154589...
        let s = equity_surface();
        let w = s.total_variance_at(0.16, 0.1);
        assert_abs_diff_eq!(w, 0.154589, epsilon = 1e-5);
    }

    #[test]
    fn total_variance_skew_direction() {
        // With rho < 0 (equity-like skew), OTM puts (k < 0) have higher
        // implied variance than OTM calls (k > 0) at same |k|.
        let s = equity_surface();
        let theta = 0.16;
        let w_put = s.total_variance_at(theta, -0.2);
        let w_call = s.total_variance_at(theta, 0.2);
        assert!(w_put > w_call, "negative rho should produce put skew");
    }

    // ========== Phi function ==========

    #[test]
    fn phi_known_values() {
        let s = equity_surface();
        // phi(theta) = eta / theta^gamma = 0.5 / theta^0.5
        assert_abs_diff_eq!(s.phi(0.04), 0.5 / 0.04_f64.sqrt(), epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(0.16), 0.5 / 0.4, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn phi_gamma_zero_is_constant() {
        // gamma = 0 => phi(theta) = eta / theta^0 = eta for all theta.
        let s = SsviSurface::new(
            -0.3, 0.5, 0.0,
            vec![1.0], vec![100.0], vec![0.16],
        ).unwrap();
        assert_abs_diff_eq!(s.phi(0.04), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(10.0), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn phi_gamma_one_is_inverse() {
        // gamma = 1 => phi(theta) = eta / theta.
        let s = SsviSurface::new(
            -0.3, 0.5, 1.0,
            vec![1.0], vec![100.0], vec![0.16],
        ).unwrap();
        assert_abs_diff_eq!(s.phi(0.04), 0.5 / 0.04, epsilon = 1e-14);
        assert_abs_diff_eq!(s.phi(1.0), 0.5, epsilon = 1e-14);
    }

    // ========== Theta and forward interpolation ==========

    #[test]
    fn theta_forward_exact_match() {
        let s = equity_surface();
        // Exact tenor should return stored values.
        let (theta, fwd) = s.theta_and_forward_at(0.5);
        assert_abs_diff_eq!(theta, 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_between_tenors() {
        let s = equity_surface();
        // Midpoint between T=0.5 (theta=0.08) and T=1.0 (theta=0.16).
        let (theta, fwd) = s.theta_and_forward_at(0.75);
        // alpha = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // theta = 0.5 * 0.08 + 0.5 * 0.16 = 0.12
        assert_abs_diff_eq!(theta, 0.12, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_before_first_tenor() {
        let s = equity_surface();
        // T=0.1 < T_0=0.25. Flat vol: theta(0.1) = 0.04 * 0.1 / 0.25 = 0.016
        let (theta, fwd) = s.theta_and_forward_at(0.1);
        assert_abs_diff_eq!(theta, 0.016, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_after_last_tenor() {
        let s = equity_surface();
        // T=3.0 > T_n=2.0. Flat vol: theta(3.0) = 0.32 * 3.0 / 2.0 = 0.48
        let (theta, fwd) = s.theta_and_forward_at(3.0);
        assert_abs_diff_eq!(theta, 0.48, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 100.0, epsilon = 1e-14);
    }

    #[test]
    fn theta_forward_interpolation_with_different_forwards() {
        let s = SsviSurface::new(
            -0.3, 0.5, 0.5,
            vec![0.5, 1.0],
            vec![100.0, 105.0],
            vec![0.08, 0.16],
        ).unwrap();
        // At T=0.75: alpha = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        let (theta, fwd) = s.theta_and_forward_at(0.75);
        assert_abs_diff_eq!(theta, 0.12, epsilon = 1e-14);
        assert_abs_diff_eq!(fwd, 102.5, epsilon = 1e-14);
    }

    // ========== Serde round-trip ==========

    #[test]
    fn serde_round_trip() {
        let s = equity_surface();
        let json = serde_json::to_string(&s).unwrap();
        let s2: SsviSurface = serde_json::from_str(&json).unwrap();
        assert_eq!(s.rho(), s2.rho());
        assert_eq!(s.eta(), s2.eta());
        assert_eq!(s.gamma(), s2.gamma());
        assert_eq!(s.tenors(), s2.tenors());
        assert_eq!(s.forwards(), s2.forwards());
        assert_eq!(s.thetas(), s2.thetas());
    }

    // ========== Send + Sync ==========

    #[test]
    fn ssvi_surface_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SsviSurface>();
    }

    // ========== VolSurface stubs still return errors ==========

    #[test]
    fn volsurface_stubs_return_error() {
        let s = equity_surface();
        assert!(s.black_vol(1.0, 100.0).is_err());
        assert!(s.black_variance(1.0, 100.0).is_err());
        assert!(s.smile_at(1.0).is_err());
        assert!(s.diagnostics().is_err());
    }

    #[test]
    fn calibrate_stub_returns_error() {
        let result = SsviSurface::calibrate(&[], &[], &[]);
        assert!(result.is_err());
    }

    // ========== Error type checks ==========

    #[test]
    fn validation_errors_are_invalid_input() {
        let err = SsviSurface::new(
            1.5, 0.5, 0.5, vec![1.0], vec![100.0], vec![0.04],
        ).unwrap_err();
        assert!(matches!(err, VolSurfError::InvalidInput { .. }));
    }
}
