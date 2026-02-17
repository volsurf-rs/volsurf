//! Dupire local volatility extraction.
//!
//! Computes local volatility from the implied volatility surface using:
//!
//! ```text
//! σ²_loc(T, K) = ∂w/∂T / [1 − (k/w)·∂w/∂k + ¼(−¼ − 1/w + k²/w²)·(∂w/∂k)² + ½·∂²w/∂k²]
//! ```
//!
//! where w(T, k) is total implied variance and k = ln(K/F).
//!
//! # References
//! - Dupire, B. "Pricing with a Smile" (1994)
//! - Gatheral, J. "The Volatility Surface: A Practitioner's Guide" (2006), Ch. 2

use std::fmt;
use std::sync::Arc;

use crate::conventions::log_moneyness;
use crate::error::{self, VolSurfError};
use crate::surface::VolSurface;
use crate::types::Vol;
use crate::validate::validate_positive;

use super::LocalVol;

/// Dupire local volatility derived from an implied vol surface.
#[derive(Clone)]
pub struct DupireLocalVol {
    surface: Arc<dyn VolSurface>,
    bump_size: f64,
}

impl fmt::Debug for DupireLocalVol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DupireLocalVol")
            .field("surface", &self.surface)
            .field("bump_size", &self.bump_size)
            .finish()
    }
}

impl DupireLocalVol {
    /// Create a Dupire local vol from an implied vol surface.
    ///
    /// Uses a default finite-difference step of 1% (`bump_size = 0.01`).
    pub fn new(surface: Arc<dyn VolSurface>) -> Self {
        Self {
            surface,
            bump_size: 0.01,
        }
    }

    /// Override the finite-difference step size.
    ///
    /// # Errors
    ///
    /// Returns [`VolSurfError::InvalidInput`] if `bump_size` is zero, negative,
    /// NaN, or infinite.
    pub fn with_bump_size(mut self, bump_size: f64) -> error::Result<Self> {
        validate_positive(bump_size, "bump_size")?;
        self.bump_size = bump_size;
        Ok(self)
    }
}

impl LocalVol for DupireLocalVol {
    /// Compute local volatility at (expiry, strike) via Gatheral (2006) Eq (1.10).
    ///
    /// Uses central finite differences for ∂w/∂y, ∂²w/∂y², and ∂w/∂T.
    /// The time derivative is taken at constant log-moneyness y = ln(K/F),
    /// adjusting the strike at each bumped tenor to account for the
    /// changing forward.
    fn local_vol(&self, expiry: f64, strike: f64) -> error::Result<Vol> {
        validate_positive(expiry, "expiry")?;
        validate_positive(strike, "strike")?;

        let h = self.bump_size;

        let smile = self.surface.smile_at(expiry)?;
        let fwd = smile.forward();
        let y = log_moneyness(strike, fwd);

        let w = self.surface.black_variance(expiry, strike)?.0;
        if w <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "non-positive total variance {w} at T={expiry}, K={strike}"
                ),
            });
        }

        // Strike-space derivatives at fixed T (central differences in log-moneyness)
        let k_up = fwd * (y + h).exp();
        let k_dn = fwd * (y - h).exp();
        let w_up = self.surface.black_variance(expiry, k_up)?.0;
        let w_dn = self.surface.black_variance(expiry, k_dn)?.0;

        let dw_dy = (w_up - w_dn) / (2.0 * h);
        let d2w_dy2 = (w_up - 2.0 * w + w_dn) / (h * h);

        // Time derivative at constant y (forward adjustment at bumped tenors)
        let dw_dt = if expiry > 2.0 * h {
            let smile_up = self.surface.smile_at(expiry + h)?;
            let smile_dn = self.surface.smile_at(expiry - h)?;
            let w_t_up = self.surface.black_variance(
                expiry + h,
                smile_up.forward() * y.exp(),
            )?.0;
            let w_t_dn = self.surface.black_variance(
                expiry - h,
                smile_dn.forward() * y.exp(),
            )?.0;
            (w_t_up - w_t_dn) / (2.0 * h)
        } else {
            // Forward difference for short expiry where T - h would be non-positive
            let smile_up = self.surface.smile_at(expiry + h)?;
            let w_t_up = self.surface.black_variance(
                expiry + h,
                smile_up.forward() * y.exp(),
            )?.0;
            (w_t_up - w) / h
        };

        // Gatheral (1.10) denominator
        let denom = 1.0
            - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;

        if denom <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "non-positive denominator {denom} at T={expiry}, K={strike} \
                     (butterfly arbitrage)"
                ),
            });
        }

        let v_local = dw_dt / denom;
        if v_local < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "negative local variance {v_local} at T={expiry}, K={strike} \
                     (calendar arbitrage)"
                ),
            });
        }

        Ok(Vol(v_local.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::arbitrage::ArbitrageReport;
    use crate::smile::SmileSection;
    use crate::surface::arbitrage::SurfaceDiagnostics;
    use crate::types::Variance;

    #[derive(Debug)]
    struct StubSurface;

    impl VolSurface for StubSurface {
        fn black_vol(&self, _: f64, _: f64) -> error::Result<Vol> {
            Ok(Vol(0.2))
        }
        fn black_variance(&self, _: f64, _: f64) -> error::Result<Variance> {
            Ok(Variance(0.04))
        }
        fn smile_at(&self, _: f64) -> error::Result<Box<dyn SmileSection>> {
            unimplemented!()
        }
        fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
    }

    fn stub_surface() -> Arc<dyn VolSurface> {
        Arc::new(StubSurface)
    }

    // Flat vol surface: w(T, K) = σ² · T for all K, forward = 100
    #[derive(Debug)]
    struct FlatVolSurface {
        sigma: f64,
        fwd: f64,
    }

    #[derive(Debug)]
    struct FlatSmile {
        sigma: f64,
        fwd: f64,
        expiry: f64,
    }

    impl SmileSection for FlatSmile {
        fn vol(&self, _: f64) -> error::Result<Vol> {
            Ok(Vol(self.sigma))
        }
        fn variance(&self, _: f64) -> error::Result<Variance> {
            Ok(Variance(self.sigma * self.sigma * self.expiry))
        }
        fn forward(&self) -> f64 {
            self.fwd
        }
        fn expiry(&self) -> f64 {
            self.expiry
        }
        fn density(&self, _: f64) -> error::Result<f64> {
            unimplemented!()
        }
        fn is_arbitrage_free(&self) -> error::Result<ArbitrageReport> {
            unimplemented!()
        }
    }

    impl VolSurface for FlatVolSurface {
        fn black_vol(&self, _: f64, _: f64) -> error::Result<Vol> {
            Ok(Vol(self.sigma))
        }
        fn black_variance(&self, expiry: f64, _: f64) -> error::Result<Variance> {
            Ok(Variance(self.sigma * self.sigma * expiry))
        }
        fn smile_at(&self, expiry: f64) -> error::Result<Box<dyn SmileSection>> {
            Ok(Box::new(FlatSmile {
                sigma: self.sigma,
                fwd: self.fwd,
                expiry,
            }))
        }
        fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
    }

    fn flat_surface(sigma: f64) -> Arc<dyn VolSurface> {
        Arc::new(FlatVolSurface { sigma, fwd: 100.0 })
    }

    #[test]
    fn default_bump_size() {
        let lv = DupireLocalVol::new(stub_surface());
        assert_eq!(lv.bump_size, 0.01);
    }

    #[test]
    fn with_bump_size_valid() {
        let lv = DupireLocalVol::new(stub_surface())
            .with_bump_size(0.005)
            .unwrap();
        assert_eq!(lv.bump_size, 0.005);
    }

    #[test]
    fn with_bump_size_rejects_zero() {
        assert!(DupireLocalVol::new(stub_surface()).with_bump_size(0.0).is_err());
    }

    #[test]
    fn with_bump_size_rejects_negative() {
        assert!(DupireLocalVol::new(stub_surface()).with_bump_size(-0.01).is_err());
    }

    #[test]
    fn with_bump_size_rejects_nan() {
        assert!(DupireLocalVol::new(stub_surface()).with_bump_size(f64::NAN).is_err());
    }

    #[test]
    fn with_bump_size_rejects_inf() {
        assert!(DupireLocalVol::new(stub_surface()).with_bump_size(f64::INFINITY).is_err());
        assert!(DupireLocalVol::new(stub_surface()).with_bump_size(f64::NEG_INFINITY).is_err());
    }

    // Flat vol: σ_loc should equal σ everywhere (Gatheral Ch.1 trivial case)
    #[test]
    fn flat_vol_returns_input_vol() {
        let sigma = 0.25;
        let lv = DupireLocalVol::new(flat_surface(sigma));

        // ATM
        let v = lv.local_vol(0.5, 100.0).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "ATM: got {}", v.0);

        // OTM
        let v = lv.local_vol(1.0, 120.0).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "OTM: got {}", v.0);

        // ITM
        let v = lv.local_vol(0.25, 80.0).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "ITM: got {}", v.0);
    }

    #[test]
    fn rejects_zero_expiry() {
        let lv = DupireLocalVol::new(flat_surface(0.2));
        assert!(lv.local_vol(0.0, 100.0).is_err());
    }

    #[test]
    fn rejects_zero_strike() {
        let lv = DupireLocalVol::new(flat_surface(0.2));
        assert!(lv.local_vol(0.5, 0.0).is_err());
    }

    // Gatheral (1.10) at y=0: denominator simplifies because (y/w)*dw_dy = 0
    // and y²/w² = 0. Denominator becomes:
    //   1 + 0.25*(-0.25 - 1/w)*(dw_dy)^2 + 0.5*d2w_dy2
    #[test]
    fn atm_matches_simplified_formula() {
        use crate::smile::SviSmile;
        use crate::surface::PiecewiseSurface;

        let fwd = 100.0;
        let (t1, t2) = (0.5, 1.5);
        let (a1, a2) = (0.02, 0.06);
        let (b, rho, m, sigma) = (0.3, -0.3, 0.0, 0.15);

        let s1 = Box::new(SviSmile::new(fwd, t1, a1, b, rho, m, sigma).unwrap())
            as Box<dyn SmileSection>;
        let s2 = Box::new(SviSmile::new(fwd, t2, a2, b, rho, m, sigma).unwrap())
            as Box<dyn SmileSection>;
        let surface = Arc::new(
            PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap(),
        );

        let dupire = DupireLocalVol::new(surface);
        let v = dupire.local_vol(1.0, 100.0).unwrap();

        // Analytical at ATM (y=0, m=0):
        // w = 0.5*(a1 + b*sigma) + 0.5*(a2 + b*sigma)
        let w_atm = 0.5 * (a1 + b * sigma) + 0.5 * (a2 + b * sigma);
        // dw/dy = b*rho (exact at y=0 with m=0, symmetric sqrt terms cancel)
        let dw_dy = b * rho;
        // dw/dT = (w2_ATM - w1_ATM) / (T2 - T1) = (a2 - a1) / (T2 - T1)
        let dw_dt = (a2 - a1) / (t2 - t1);
        // d2w/dy2 ≈ b/sigma (analytical), FD gives ~1.998 with h=0.01
        // Use analytical for the expected value; tolerance covers FD error
        let d2w_dy2 = b / sigma;

        let denom = 1.0
            + 0.25 * (-0.25 - 1.0 / w_atm) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;
        let expected = (dw_dt / denom).sqrt();

        assert!(
            (v.0 - expected).abs() < 5e-4,
            "ATM local vol: got {}, expected {} (diff {})",
            v.0, expected, (v.0 - expected).abs()
        );
    }

    // Compare Dupire local vol against analytical SVI derivatives at OTM point.
    // SVI w'(k) = b*[rho + (k-m)/sqrt((k-m)^2 + sigma^2)]
    // SVI w''(k) = b*sigma^2/((k-m)^2 + sigma^2)^(3/2)
    // Gatheral (2006), Eq (1.10)
    #[test]
    fn svi_analytical_otm() {
        use crate::smile::SviSmile;
        use crate::surface::PiecewiseSurface;

        let fwd = 100.0;
        let (t1, t2) = (0.5, 1.5);
        let (a1, a2) = (0.02, 0.06);
        let (b, rho, m, sigma) = (0.3, -0.3, 0.0, 0.15);

        let s1 = Box::new(SviSmile::new(fwd, t1, a1, b, rho, m, sigma).unwrap())
            as Box<dyn SmileSection>;
        let s2 = Box::new(SviSmile::new(fwd, t2, a2, b, rho, m, sigma).unwrap())
            as Box<dyn SmileSection>;
        let surface = Arc::new(
            PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap(),
        );

        // Test at OTM: K=110, T=1.0
        let k = 110.0_f64;
        let y = (k / fwd).ln();
        let dupire = DupireLocalVol::new(surface);
        let v = dupire.local_vol(1.0, k).unwrap();

        // SVI analytical derivatives at y = ln(110/100) ≈ 0.09531
        let dk = y - m;
        let r = (dk * dk + sigma * sigma).sqrt();
        let dw_dy = b * (rho + dk / r);
        let d2w_dy2 = b * sigma * sigma / (r * r * r);

        // w at midpoint: 0.5*w1(y) + 0.5*w2(y)
        let w_svi = |a_val: f64| a_val + b * (rho * dk + r);
        let w = 0.5 * w_svi(a1) + 0.5 * w_svi(a2);

        // dw/dT: same as ATM since both SVI shapes are identical
        let dw_dt = (a2 - a1) / (t2 - t1);

        let denom = 1.0
            - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;
        let expected = (dw_dt / denom).sqrt();

        // Tolerance accounts for O(h²) finite difference error in strike derivatives
        // and SplineSmile approximation in smile_at() for time derivative
        assert!(
            (v.0 - expected).abs() < 2e-3,
            "OTM local vol at K={}: got {}, expected {} (diff {})",
            k, v.0, expected, (v.0 - expected).abs()
        );
    }

    // Multiple expiries on flat vol surface (Gatheral Ch.1)
    #[test]
    fn flat_vol_short_expiry_forward_difference() {
        // T=0.015 < 2*h (h=0.01), uses forward difference path
        let sigma = 0.20;
        let lv = DupireLocalVol::new(flat_surface(sigma));
        let v = lv.local_vol(0.015, 100.0).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "short T: got {}", v.0);
    }
}
