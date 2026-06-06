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
use crate::types::{Strike, Tenor, Vol};
use crate::validate::{validate_non_negative, validate_positive};

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

    /// Wrap this Dupire local vol in a [`BoundaryLocalVol`] that handles the
    /// small-time boundary, using the finite-difference [`bump_size`](Self::with_bump_size)
    /// as the floor.
    ///
    /// For a query at `t ≤ bump_size` the wrapper evaluates `σ_loc` at
    /// `t = bump_size` instead of erroring; for `t > bump_size` it delegates to
    /// the strict [`local_vol`](LocalVol::local_vol) unchanged. This is the
    /// opt-in way to keep total variance `w = σ²·T` bounded away from the
    /// singular `1/w` and `k²/w²` terms in the Gatheral (2006) denominator as
    /// `T → 0` (see [`BoundaryLocalVol`]).
    pub fn with_boundary(self) -> BoundaryLocalVol<Self> {
        // `bump_size` is validated positive on construction (default 0.01,
        // `with_bump_size` rejects non-positive), so it is a valid floor.
        let floor = self.bump_size;
        BoundaryLocalVol { inner: self, floor }
    }
}

impl LocalVol for DupireLocalVol {
    /// Compute local volatility at (expiry, strike) via Gatheral (2006) Eq (1.10).
    ///
    /// Uses central finite differences for ∂w/∂y, ∂²w/∂y², and ∂w/∂T.
    /// The time derivative is taken at constant log-moneyness y = ln(K/F),
    /// adjusting the strike at each bumped tenor to account for the
    /// changing forward.
    fn local_vol(&self, expiry: Tenor, strike: Strike) -> error::Result<Vol> {
        validate_positive(expiry.0, "expiry")?;
        validate_positive(strike.0, "strike")?;

        let h = self.bump_size;
        let t = expiry.0;
        let k = strike.0;

        let smile = self.surface.smile_at(expiry)?;
        let fwd = smile.forward();
        let y = log_moneyness(k, fwd)?;

        let w = self.surface.black_variance(expiry, strike)?.0;
        if w <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!("non-positive total variance {w} at T={t}, K={k}"),
            });
        }

        // Strike-space derivatives at fixed T (central differences in log-moneyness)
        let k_up = fwd * (y + h).exp();
        let k_dn = fwd * (y - h).exp();
        let w_up = self.surface.black_variance(expiry, Strike(k_up))?.0;
        let w_dn = self.surface.black_variance(expiry, Strike(k_dn))?.0;

        let dw_dy = (w_up - w_dn) / (2.0 * h);
        let d2w_dy2 = (w_up - 2.0 * w + w_dn) / (h * h);

        // Time derivative at constant y (forward adjustment at bumped tenors)
        let dw_dt = if t > 2.0 * h {
            let smile_up = self.surface.smile_at(Tenor(t + h))?;
            let smile_dn = self.surface.smile_at(Tenor(t - h))?;
            let w_t_up = self
                .surface
                .black_variance(Tenor(t + h), Strike(smile_up.forward() * y.exp()))?
                .0;
            let w_t_dn = self
                .surface
                .black_variance(Tenor(t - h), Strike(smile_dn.forward() * y.exp()))?
                .0;
            (w_t_up - w_t_dn) / (2.0 * h)
        } else {
            // Forward difference for short expiry where T - h would be non-positive
            let smile_up = self.surface.smile_at(Tenor(t + h))?;
            let w_t_up = self
                .surface
                .black_variance(Tenor(t + h), Strike(smile_up.forward() * y.exp()))?
                .0;
            (w_t_up - w) / h
        };

        // Gatheral (1.10) denominator
        let denom = 1.0 - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;

        if denom <= 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "non-positive denominator {denom} at T={t}, K={k} \
                     (butterfly arbitrage)"
                ),
            });
        }

        let v_local = dw_dt / denom;
        if v_local < 0.0 {
            return Err(VolSurfError::NumericalError {
                message: format!(
                    "negative local variance {v_local} at T={t}, K={k} \
                     (calendar arbitrage)"
                ),
            });
        }

        Ok(Vol(v_local.sqrt()))
    }
}

/// A [`LocalVol`] adapter that smooths the small-time boundary.
///
/// The Dupire formula (Gatheral 2006, Eq. 1.10) divides by total implied
/// variance `w = σ²·T`; its denominator carries `−1/w` and `k²/w²` terms that
/// are ill-conditioned as `T → 0`, and the strict
/// [`DupireLocalVol`] rejects `T = 0` outright. Consumers that evaluate local
/// vol at the calendar-time boundary — a backward PDE march whose last step is
/// `t = 0`, or a Monte-Carlo path's first step — therefore have to clamp their
/// queries away from zero by hand.
///
/// `BoundaryLocalVol` encapsulates that clamp: for a query at `T ≤ floor` it
/// evaluates the inner local vol at `T = floor`; for `T > floor` it delegates
/// **exactly** to the inner implementation. Evaluating at `floor` keeps `w`
/// bounded away from zero, sidestepping the singular denominator entirely. On a
/// flat surface the result is exact (`σ_loc ≡ σ`); on a smile it is an `O(floor)`
/// short-maturity extrapolation.
///
/// This is opt-in and composes around any `L: LocalVol` (mirroring the
/// `std::io::BufReader<R>` / `Take<R>` adapter pattern); the strict
/// [`DupireLocalVol::local_vol`] path is left unchanged. The usual entry point is
/// [`DupireLocalVol::with_boundary`], which sets `floor` to the Dupire bump size.
///
/// # References
/// - Gatheral, J. "The Volatility Surface: A Practitioner's Guide" (2006), Ch. 2
#[derive(Debug, Clone)]
pub struct BoundaryLocalVol<L: LocalVol> {
    inner: L,
    floor: f64,
}

impl<L: LocalVol> BoundaryLocalVol<L> {
    /// Wrap `inner`, treating any query at `expiry ≤ floor` as a query at
    /// `expiry = floor`.
    ///
    /// # Errors
    ///
    /// Returns [`VolSurfError::InvalidInput`] if `floor` is zero, negative, NaN,
    /// or infinite.
    pub fn new(inner: L, floor: f64) -> error::Result<Self> {
        validate_positive(floor, "floor")?;
        Ok(Self { inner, floor })
    }

    /// Consume the wrapper and return the inner [`LocalVol`].
    pub fn into_inner(self) -> L {
        self.inner
    }
}

impl<L: LocalVol> LocalVol for BoundaryLocalVol<L> {
    /// Local volatility with the small-time boundary handled.
    ///
    /// For `expiry ≤ floor` the inner local vol is evaluated at
    /// `expiry = floor`; for `expiry > floor` the call delegates unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`VolSurfError::InvalidInput`] if `strike` is non-positive or
    /// `expiry` is negative, NaN, or infinite (`expiry = 0` is valid and maps to
    /// `floor`). Otherwise propagates any error from the inner implementation.
    fn local_vol(&self, expiry: Tenor, strike: Strike) -> error::Result<Vol> {
        validate_positive(strike.0, "strike")?;
        validate_non_negative(expiry.0, "expiry")?;

        if expiry.0 <= self.floor {
            self.inner.local_vol(Tenor(self.floor), strike)
        } else {
            self.inner.local_vol(expiry, strike)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smile::SmileSection;
    use crate::surface::arbitrage::SurfaceDiagnostics;
    use crate::types::{Strike, Tenor, Variance};

    #[derive(Debug)]
    struct StubSurface;

    impl VolSurface for StubSurface {
        fn black_vol(&self, _: Tenor, _: Strike) -> error::Result<Vol> {
            Ok(Vol(0.2))
        }
        fn black_variance(&self, _: Tenor, _: Strike) -> error::Result<Variance> {
            Ok(Variance(0.04))
        }
        fn smile_at(&self, _: Tenor) -> error::Result<Box<dyn SmileSection>> {
            unimplemented!()
        }
        fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
        fn diagnostics_with(
            &self,
            _: &crate::smile::ArbitrageScanConfig,
        ) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
        fn tenors(&self) -> &[f64] {
            &[]
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
        fn vol(&self, _: Strike) -> error::Result<Vol> {
            Ok(Vol(self.sigma))
        }
        fn variance(&self, _: Strike) -> error::Result<Variance> {
            Ok(Variance(self.sigma * self.sigma * self.expiry))
        }
        fn forward(&self) -> f64 {
            self.fwd
        }
        fn expiry(&self) -> f64 {
            self.expiry
        }
        fn model_name(&self) -> &'static str {
            "Flat"
        }
        fn density(&self, _: Strike) -> error::Result<f64> {
            unimplemented!()
        }
    }

    impl VolSurface for FlatVolSurface {
        fn black_vol(&self, _: Tenor, _: Strike) -> error::Result<Vol> {
            Ok(Vol(self.sigma))
        }
        fn black_variance(&self, expiry: Tenor, _: Strike) -> error::Result<Variance> {
            Ok(Variance(self.sigma * self.sigma * expiry.0))
        }
        fn smile_at(&self, expiry: Tenor) -> error::Result<Box<dyn SmileSection>> {
            Ok(Box::new(FlatSmile {
                sigma: self.sigma,
                fwd: self.fwd,
                expiry: expiry.0,
            }))
        }
        fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
        fn diagnostics_with(
            &self,
            _: &crate::smile::ArbitrageScanConfig,
        ) -> error::Result<SurfaceDiagnostics> {
            unimplemented!()
        }
        fn tenors(&self) -> &[f64] {
            &[]
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
        assert!(
            DupireLocalVol::new(stub_surface())
                .with_bump_size(0.0)
                .is_err()
        );
    }

    #[test]
    fn with_bump_size_rejects_negative() {
        assert!(
            DupireLocalVol::new(stub_surface())
                .with_bump_size(-0.01)
                .is_err()
        );
    }

    #[test]
    fn with_bump_size_rejects_nan() {
        assert!(
            DupireLocalVol::new(stub_surface())
                .with_bump_size(f64::NAN)
                .is_err()
        );
    }

    #[test]
    fn with_bump_size_rejects_inf() {
        assert!(
            DupireLocalVol::new(stub_surface())
                .with_bump_size(f64::INFINITY)
                .is_err()
        );
        assert!(
            DupireLocalVol::new(stub_surface())
                .with_bump_size(f64::NEG_INFINITY)
                .is_err()
        );
    }

    // Flat vol: σ_loc should equal σ everywhere (Gatheral Ch.1 trivial case)
    #[test]
    fn flat_vol_returns_input_vol() {
        let sigma = 0.25;
        let lv = DupireLocalVol::new(flat_surface(sigma));

        // ATM
        let v = lv.local_vol(Tenor(0.5), Strike(100.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "ATM: got {}", v.0);

        // OTM
        let v = lv.local_vol(Tenor(1.0), Strike(120.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "OTM: got {}", v.0);

        // ITM
        let v = lv.local_vol(Tenor(0.25), Strike(80.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "ITM: got {}", v.0);
    }

    #[test]
    fn rejects_zero_expiry() {
        let lv = DupireLocalVol::new(flat_surface(0.2));
        assert!(lv.local_vol(Tenor(0.0), Strike(100.0)).is_err());
    }

    #[test]
    fn rejects_zero_strike() {
        let lv = DupireLocalVol::new(flat_surface(0.2));
        assert!(lv.local_vol(Tenor(0.5), Strike(0.0)).is_err());
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
        let surface = Arc::new(PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap());

        let dupire = DupireLocalVol::new(surface);
        let v = dupire.local_vol(Tenor(1.0), Strike(100.0)).unwrap();

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

        let denom = 1.0 + 0.25 * (-0.25 - 1.0 / w_atm) * dw_dy * dw_dy + 0.5 * d2w_dy2;
        let expected = (dw_dt / denom).sqrt();

        assert!(
            (v.0 - expected).abs() < 5e-4,
            "ATM local vol: got {}, expected {} (diff {})",
            v.0,
            expected,
            (v.0 - expected).abs()
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
        let surface = Arc::new(PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap());

        // Test at OTM: K=110, T=1.0
        let k = 110.0_f64;
        let y = (k / fwd).ln();
        let dupire = DupireLocalVol::new(surface);
        let v = dupire.local_vol(Tenor(1.0), Strike(k)).unwrap();

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

        let denom = 1.0 - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;
        let expected = (dw_dt / denom).sqrt();

        // Tolerance accounts for O(h²) finite difference error in strike derivatives
        // and SplineSmile approximation in smile_at() for time derivative
        assert!(
            (v.0 - expected).abs() < 2e-3,
            "OTM local vol at K={}: got {}, expected {} (diff {})",
            k,
            v.0,
            expected,
            (v.0 - expected).abs()
        );
    }

    // Multiple expiries on flat vol surface (Gatheral Ch.1)
    #[test]
    fn propagates_error_from_zero_forward() {
        #[derive(Debug)]
        struct ZeroFwdSurface;

        #[derive(Debug)]
        struct ZeroFwdSmile;

        impl SmileSection for ZeroFwdSmile {
            fn vol(&self, _: Strike) -> error::Result<Vol> {
                Ok(Vol(0.2))
            }
            fn variance(&self, _: Strike) -> error::Result<Variance> {
                Ok(Variance(0.01))
            }
            fn forward(&self) -> f64 {
                0.0
            }
            fn expiry(&self) -> f64 {
                0.5
            }
            fn model_name(&self) -> &'static str {
                "ZeroFwd"
            }
            fn density(&self, _: Strike) -> error::Result<f64> {
                unimplemented!()
            }
        }

        impl VolSurface for ZeroFwdSurface {
            fn black_vol(&self, _: Tenor, _: Strike) -> error::Result<Vol> {
                Ok(Vol(0.2))
            }
            fn black_variance(&self, _: Tenor, _: Strike) -> error::Result<Variance> {
                Ok(Variance(0.01))
            }
            fn smile_at(&self, _: Tenor) -> error::Result<Box<dyn SmileSection>> {
                Ok(Box::new(ZeroFwdSmile))
            }
            fn diagnostics(&self) -> error::Result<SurfaceDiagnostics> {
                unimplemented!()
            }
            fn diagnostics_with(
                &self,
                _: &crate::smile::ArbitrageScanConfig,
            ) -> error::Result<SurfaceDiagnostics> {
                unimplemented!()
            }
            fn tenors(&self) -> &[f64] {
                &[]
            }
        }

        let dupire = DupireLocalVol::new(Arc::new(ZeroFwdSurface));
        let err = dupire.local_vol(Tenor(0.5), Strike(100.0)).unwrap_err();
        assert!(
            matches!(err, VolSurfError::InvalidInput { .. }),
            "expected InvalidInput from log_moneyness, got {err:?}"
        );
    }

    #[test]
    fn flat_vol_short_expiry_forward_difference() {
        // T=0.015 < 2*h (h=0.01), uses forward difference path
        let sigma = 0.20;
        let lv = DupireLocalVol::new(flat_surface(sigma));
        let v = lv.local_vol(Tenor(0.015), Strike(100.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "short T: got {}", v.0);
    }

    // ----- BoundaryLocalVol (PAN-25) -----

    // SVI PiecewiseSurface fixture shared by the boundary tests below — same
    // arb-free parameters as atm_matches_simplified_formula / svi_analytical_otm.
    fn svi_surface() -> Arc<dyn VolSurface> {
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
        Arc::new(PiecewiseSurface::new(vec![t1, t2], vec![s1, s2]).unwrap())
    }

    // Flat surface: sigma_loc == sigma exactly, so the clamped boundary value at
    // t = 0, 0 < t < floor, and t == floor must all return sigma.
    #[test]
    fn boundary_flat_exact_at_and_below_floor() {
        let sigma = 0.25;
        let adapter = DupireLocalVol::new(flat_surface(sigma)).with_boundary();

        // t = 0 is rejected by the strict path but rescued to floor here.
        let v = adapter.local_vol(Tenor(0.0), Strike(100.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "t=0: got {}", v.0);

        // 0 < t < floor (= 0.01)
        let v = adapter.local_vol(Tenor(0.005), Strike(120.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "0<t<floor: got {}", v.0);

        // exactly at the floor
        let v = adapter.local_vol(Tenor(0.01), Strike(80.0)).unwrap();
        assert!((v.0 - sigma).abs() < 1e-10, "t=floor: got {}", v.0);
    }

    // For t > floor the adapter must delegate to DupireLocalVol::local_vol with
    // the original tenor: same code path, so the result is bit-identical.
    #[test]
    fn boundary_delegates_bit_identical_above_floor() {
        let surf = svi_surface();
        let raw = DupireLocalVol::new(surf.clone());
        let adapter = DupireLocalVol::new(surf).with_boundary();

        for (t, k) in [(0.5, 100.0), (1.0, 110.0), (1.2, 90.0)] {
            let r = raw.local_vol(Tenor(t), Strike(k)).unwrap();
            let a = adapter.local_vol(Tenor(t), Strike(k)).unwrap();
            assert_eq!(r.0, a.0, "delegation drift at t={t}, K={k}");
        }
    }

    // The boundary the adapter exists to fix: an OTM query at t=0 errors on the
    // strict path (InvalidInput) but succeeds on the adapter (clamped to floor).
    // NOTE: on the arb-free flat-extrapolation fixture a NumericalError baseline
    // is not reproducible (w scales t/t1, so smaller t is more stable); the
    // t=0 -> InvalidInput rescue is the real boundary (requirements FR-1/FR-5).
    #[test]
    fn boundary_succeeds_where_raw_errors_at_t0() {
        let surf = svi_surface();
        let raw = DupireLocalVol::new(surf.clone());
        let adapter = DupireLocalVol::new(surf).with_boundary();
        let k = Strike(110.0); // OTM

        assert!(
            raw.local_vol(Tenor(0.0), k).is_err(),
            "strict path should reject t=0"
        );
        assert!(
            adapter.local_vol(Tenor(0.0), k).is_ok(),
            "adapter should rescue t=0 via the floor"
        );
    }

    // Strike stays strict; t < 0 / NaN / inf still rejected (t = 0 is allowed).
    #[test]
    fn boundary_rejects_invalid_inputs() {
        let a = DupireLocalVol::new(flat_surface(0.2)).with_boundary();
        assert!(matches!(
            a.local_vol(Tenor(-1.0), Strike(100.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            a.local_vol(Tenor(f64::NAN), Strike(100.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            a.local_vol(Tenor(f64::INFINITY), Strike(100.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            a.local_vol(Tenor(0.5), Strike(0.0)),
            Err(VolSurfError::InvalidInput { .. })
        ));
        assert!(matches!(
            a.local_vol(Tenor(0.5), Strike(f64::NAN)),
            Err(VolSurfError::InvalidInput { .. })
        ));
    }

    // with_boundary()'s floor is the Dupire bump_size (single source of truth):
    // default 0.01, and a custom bump propagates.
    #[test]
    fn boundary_floor_defaults_to_bump_size() {
        let a = DupireLocalVol::new(flat_surface(0.2)).with_boundary();
        assert_eq!(a.floor, 0.01);

        let a2 = DupireLocalVol::new(flat_surface(0.2))
            .with_bump_size(0.005)
            .unwrap()
            .with_boundary();
        assert_eq!(a2.floor, 0.005);
    }

    // BoundaryLocalVol::new rejects a non-positive / non-finite floor, honors a
    // valid custom floor, and into_inner() returns the wrapped LocalVol.
    #[test]
    fn boundary_new_validates_floor_and_into_inner() {
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                BoundaryLocalVol::new(DupireLocalVol::new(stub_surface()), bad).is_err(),
                "floor {bad} should be rejected"
            );
        }

        let b = BoundaryLocalVol::new(DupireLocalVol::new(stub_surface()), 0.02).unwrap();
        assert_eq!(b.floor, 0.02, "custom floor honored");
        let _inner: DupireLocalVol = b.into_inner();
    }
}
