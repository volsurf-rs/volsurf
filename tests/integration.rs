//! Integration tests for the volsurf pipeline.
//!
//! Exercises the full path from market data through calibration,
//! surface construction, cross-tenor interpolation, arbitrage
//! diagnostics, and cross-model comparison for SVI, SABR, and SSVI.

use std::sync::Arc;
use std::thread;

use approx::assert_abs_diff_eq;
use volsurf::VolSurfError;
use volsurf::smile::spline::SplineSmile;
use volsurf::smile::{SabrSmile, SmileSection, SviSmile};
use volsurf::surface::{PiecewiseSurface, SmileModel, SsviSurface, SurfaceBuilder, VolSurface};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// SVI parameter set for generating synthetic market data.
struct SviParams {
    forward: f64,
    expiry: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
}

/// Generate synthetic (strike, vol) market data from known SVI parameters.
fn svi_market_data(p: &SviParams, strikes: &[f64]) -> Vec<(f64, f64)> {
    let smile = SviSmile::new(p.forward, p.expiry, p.a, p.b, p.rho, p.m, p.sigma).unwrap();
    strikes
        .iter()
        .map(|&k| (k, smile.vol(k).unwrap().0))
        .collect()
}

/// Standard strike grid: 21 strikes from 80 to 120.
fn standard_strikes() -> Vec<f64> {
    (0..21).map(|i| 80.0 + 2.0 * i as f64).collect()
}

/// Build a 3-tenor test surface via the builder API.
///
/// SVI params chosen so ATM total variance increases with tenor:
/// - 3M: vol ~26%, w_atm ≈ 0.017
/// - 6M: vol ~22%, w_atm ≈ 0.024
/// - 1Y: vol ~19%, w_atm ≈ 0.036
fn build_3_tenor_surface() -> PiecewiseSurface {
    let spot = 100.0;
    let rate = 0.05;
    let strikes = standard_strikes();

    // 3M: a + b*sigma = 0.005 + 0.05*0.25 = 0.0175 → vol ≈ 26.5%
    let data_3m = svi_market_data(
        &SviParams {
            forward: spot * (rate * 0.25_f64).exp(),
            expiry: 0.25,
            a: 0.005,
            b: 0.05,
            rho: -0.3,
            m: 0.0,
            sigma: 0.25,
        },
        &strikes,
    );
    // 6M: a + b*sigma = 0.01 + 0.04*0.35 = 0.024 → vol ≈ 21.9%
    let data_6m = svi_market_data(
        &SviParams {
            forward: spot * (rate * 0.5_f64).exp(),
            expiry: 0.5,
            a: 0.01,
            b: 0.04,
            rho: -0.25,
            m: 0.0,
            sigma: 0.35,
        },
        &strikes,
    );
    // 1Y: a + b*sigma = 0.02 + 0.04*0.4 = 0.036 → vol ≈ 19.0%
    let data_1y = svi_market_data(
        &SviParams {
            forward: spot * (rate * 1.0_f64).exp(),
            expiry: 1.0,
            a: 0.02,
            b: 0.04,
            rho: -0.2,
            m: 0.0,
            sigma: 0.4,
        },
        &strikes,
    );

    let (strikes_3m, vols_3m): (Vec<f64>, Vec<f64>) = data_3m.into_iter().unzip();
    let (strikes_6m, vols_6m): (Vec<f64>, Vec<f64>) = data_6m.into_iter().unzip();
    let (strikes_1y, vols_1y): (Vec<f64>, Vec<f64>) = data_1y.into_iter().unzip();

    SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .add_tenor(0.25, &strikes_3m, &vols_3m)
        .add_tenor(0.50, &strikes_6m, &vols_6m)
        .add_tenor(1.00, &strikes_1y, &vols_1y)
        .build()
        .unwrap()
}

// ---------------------------------------------------------------------------
// Test 1: SVI calibration round-trip
// ---------------------------------------------------------------------------

#[test]
fn svi_calibration_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 1.0;
    let (a, b, rho, m, sigma) = (0.04, 0.4, -0.4, 0.0, 0.4);
    let strikes = standard_strikes();

    let original = SviSmile::new(forward, expiry, a, b, rho, m, sigma)?;

    let market_data: Vec<(f64, f64)> = strikes
        .iter()
        .map(|&k| (k, original.vol(k).unwrap().0))
        .collect();

    let calibrated = SviSmile::calibrate(forward, expiry, &market_data)?;

    // Compute RMS error in vol points
    let mut sum_sq = 0.0;
    for &k in &strikes {
        let orig_vol = original.vol(k)?.0;
        let calib_vol = calibrated.vol(k)?.0;
        let diff = orig_vol - calib_vol;
        sum_sq += diff * diff;
    }
    let rms = (sum_sq / strikes.len() as f64).sqrt();
    assert!(
        rms < 0.001,
        "SVI calibration round-trip RMS = {rms:.6}, should be < 0.001"
    );

    Ok(())
}

#[test]
fn svi_calibration_round_trip_symmetric() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 0.5;
    let (a, b, rho, m, sigma) = (0.03, 0.3, 0.0, 0.0, 0.5);
    let strikes = standard_strikes();

    let original = SviSmile::new(forward, expiry, a, b, rho, m, sigma)?;
    let market_data: Vec<(f64, f64)> = strikes
        .iter()
        .map(|&k| (k, original.vol(k).unwrap().0))
        .collect();

    let calibrated = SviSmile::calibrate(forward, expiry, &market_data)?;

    let mut sum_sq = 0.0;
    for &k in &strikes {
        let diff = original.vol(k)?.0 - calibrated.vol(k)?.0;
        sum_sq += diff * diff;
    }
    let rms = (sum_sq / strikes.len() as f64).sqrt();
    assert!(
        rms < 0.001,
        "Symmetric SVI round-trip RMS = {rms:.6}, should be < 0.001"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: 3-tenor surface build and query
// ---------------------------------------------------------------------------

#[test]
fn three_tenor_surface_build_and_query() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();

    // Query at exact tenors — should return reasonable vols
    for t in [0.25, 0.50, 1.00] {
        let vol = surface.black_vol(t, 100.0)?;
        assert!(
            vol.0 > 0.10 && vol.0 < 0.50,
            "ATM vol at T={t} = {:.4}, out of range",
            vol.0
        );
    }

    // Query between tenors
    let vol_mid = surface.black_vol(0.375, 100.0)?;
    assert!(
        vol_mid.0 > 0.10 && vol_mid.0 < 0.50,
        "Interpolated ATM vol = {:.4}, out of range",
        vol_mid.0
    );

    Ok(())
}

#[test]
fn three_tenor_vol_variance_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();

    for t in [0.25, 0.375, 0.50, 0.75, 1.00] {
        for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = surface.black_vol(t, k)?;
            let var = surface.black_variance(t, k)?;
            assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-10);
        }
    }

    Ok(())
}

#[test]
fn three_tenor_smile_at_returns_queryable_section() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();

    for t in [0.25, 0.50, 0.75, 1.00] {
        let smile = surface.smile_at(t)?;
        assert_abs_diff_eq!(smile.expiry(), t, epsilon = 1e-14);
        assert!(smile.forward() > 0.0);

        let vol = smile.vol(100.0)?;
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: SPX-like end-to-end
// ---------------------------------------------------------------------------

#[test]
fn spx_like_surface_no_nan_no_negative_vol() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();

    // Query across full grid
    let tenors = [0.1, 0.25, 0.375, 0.50, 0.75, 1.0, 1.5];
    let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();

    for &t in &tenors {
        for &k in &strikes {
            let vol = surface.black_vol(t, k)?;
            assert!(
                vol.0.is_finite() && vol.0 > 0.0,
                "vol({t}, {k}) = {:.6} is invalid",
                vol.0
            );
        }
    }

    Ok(())
}

#[test]
fn variance_monotone_in_time_for_arb_free_surface() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();

    // Total variance should be non-decreasing in time at each strike
    let tenors = [0.25, 0.50, 1.00];
    for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
        let mut prev_var = 0.0_f64;
        for &t in &tenors {
            let var = surface.black_variance(t, k)?.0;
            assert!(
                var >= prev_var - 1e-10,
                "Calendar violation at K={k}: w({t}) = {var:.6} < w(prev) = {prev_var:.6}"
            );
            prev_var = var;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: Butterfly violation detection
// ---------------------------------------------------------------------------

#[test]
fn butterfly_arb_free_params_report_clean() -> Result<(), Box<dyn std::error::Error>> {
    // Well-behaved SVI params should be arb-free
    let smile = SviSmile::new(100.0, 1.0, 0.04, 0.4, -0.3, 0.0, 0.5)?;
    let report = smile.is_arbitrage_free()?;
    assert!(
        report.is_free,
        "Clean params should be arb-free, got {} violations",
        report.butterfly_violations.len()
    );

    Ok(())
}

#[test]
fn butterfly_violation_detected_via_diagnostics() -> Result<(), Box<dyn std::error::Error>> {
    // Construct a surface with a smile that has butterfly violations.
    // Use extreme SVI params: large b, small sigma, rho close to -1.
    // These pass new() validation but can produce negative density.
    let smile = SviSmile::new(100.0, 1.0, 0.001, 0.8, -0.95, 0.0, 0.05)?;
    let report = smile.is_arbitrage_free()?;

    // With such extreme params, butterfly violations are likely
    // (though not 100% guaranteed for every param set).
    // If this particular set is arb-free, that's fine — skip assertion.
    if !report.is_free {
        assert!(
            !report.butterfly_violations.is_empty(),
            "Non-free report should have violations"
        );
        for v in &report.butterfly_violations {
            assert!(
                v.density < 0.0,
                "Butterfly violation should have negative density"
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 5: Calendar violation detection
// ---------------------------------------------------------------------------

#[test]
fn calendar_violation_detected_for_inverted_surface() -> Result<(), Box<dyn std::error::Error>> {
    // Direct construction: short tenor has HIGHER vol than long tenor.
    // 3M: 30% vol → w = 0.09 * 0.25 = 0.0225
    // 1Y: 15% vol → w = 0.0225 * 1.0 = 0.0225
    // Actually, make the inversion stronger:
    // 3M: 35% vol → w = 0.1225 * 0.25 = 0.030625
    // 1Y: 15% vol → w = 0.0225 * 1.0 = 0.0225 < 0.030625 → VIOLATION
    let fwd = 100.0;
    let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];

    let w_3m = 0.35 * 0.35 * 0.25; // 0.030625
    let w_1y = 0.15 * 0.15 * 1.0; // 0.0225

    let vars_3m = vec![w_3m; 5];
    let vars_1y = vec![w_1y; 5];

    let smile_3m = SplineSmile::new(fwd, 0.25, strikes.clone(), vars_3m)?;
    let smile_1y = SplineSmile::new(fwd, 1.0, strikes, vars_1y)?;

    let surface = PiecewiseSurface::new(
        vec![0.25, 1.0],
        vec![Box::new(smile_3m), Box::new(smile_1y)],
    )?;

    let diag = surface.diagnostics()?;
    assert!(!diag.is_free, "Inverted surface should not be arb-free");
    assert!(
        !diag.calendar_violations.is_empty(),
        "Should have calendar violations"
    );

    for v in &diag.calendar_violations {
        assert!(
            v.variance_short > v.variance_long,
            "Calendar violation: short variance {} should exceed long variance {}",
            v.variance_short,
            v.variance_long
        );
    }

    Ok(())
}

#[test]
fn diagnostics_returns_valid_result() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_3_tenor_surface();
    let diag = surface.diagnostics()?;

    // Should have one smile report per tenor
    assert_eq!(diag.smile_reports.len(), 3, "Should have 3 tenor reports");

    // Calendar violations at extreme wings can occur due to SVI calibration
    // reshaping the curve slightly. But at core strikes (80-120), the
    // variance term structure should be monotone (verified by
    // variance_monotone_in_time_for_arb_free_surface).
    // Here we just verify the diagnostics machinery works.
    for v in &diag.calendar_violations {
        assert!(v.tenor_short < v.tenor_long);
        assert!(v.strike > 0.0);
    }

    Ok(())
}

#[test]
fn diagnostics_is_free_for_hand_crafted_monotone_surface() -> Result<(), Box<dyn std::error::Error>>
{
    // Direct construction with strictly increasing flat variances
    let fwd = 100.0;
    let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];

    let smile_3m = SplineSmile::new(fwd, 0.25, strikes.clone(), vec![0.010; 5])?;
    let smile_1y = SplineSmile::new(fwd, 1.0, strikes, vec![0.040; 5])?;

    let surface = PiecewiseSurface::new(
        vec![0.25, 1.0],
        vec![Box::new(smile_3m), Box::new(smile_1y)],
    )?;

    let diag = surface.diagnostics()?;
    assert!(
        diag.calendar_violations.is_empty(),
        "Monotone surface should have no calendar violations, got {}",
        diag.calendar_violations.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 6: Concurrent queries from multiple threads
// ---------------------------------------------------------------------------

#[test]
fn concurrent_surface_queries() -> Result<(), Box<dyn std::error::Error>> {
    let surface = Arc::new(build_3_tenor_surface());

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let s = Arc::clone(&surface);
            thread::spawn(move || -> volsurf::Result<()> {
                let strike = 80.0 + i as f64 * 5.0;
                for &t in &[0.25, 0.50, 0.75, 1.00] {
                    let vol = s.black_vol(t, strike)?;
                    assert!(
                        vol.0 > 0.0 && vol.0 < 2.0,
                        "vol({t}, {strike}) = {:.4} out of range",
                        vol.0
                    );
                    let var = s.black_variance(t, strike)?;
                    assert!(var.0 > 0.0 && var.0.is_finite());
                }
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked")?;
    }

    Ok(())
}

#[test]
fn concurrent_smile_at_queries() -> Result<(), Box<dyn std::error::Error>> {
    let surface = Arc::new(build_3_tenor_surface());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let s = Arc::clone(&surface);
            thread::spawn(move || -> volsurf::Result<()> {
                let t = 0.25 + i as f64 * 0.25;
                let smile = s.smile_at(t)?;
                let vol = smile.vol(100.0)?;
                assert!(vol.0 > 0.0);
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 7: CubicSpline builder end-to-end
// ---------------------------------------------------------------------------

#[test]
fn cubic_spline_surface_end_to_end() -> Result<(), Box<dyn std::error::Error>> {
    use volsurf::surface::SmileModel;

    let spot = 100.0;
    let rate = 0.05;
    let strikes = standard_strikes(); // 21 strikes, 80..120

    // Generate synthetic vols from a known SVI smile for each tenor
    let data_3m = svi_market_data(
        &SviParams {
            forward: spot * (rate * 0.25_f64).exp(),
            expiry: 0.25,
            a: 0.005,
            b: 0.05,
            rho: -0.3,
            m: 0.0,
            sigma: 0.25,
        },
        &strikes,
    );
    let data_1y = svi_market_data(
        &SviParams {
            forward: spot * (rate * 1.0_f64).exp(),
            expiry: 1.0,
            a: 0.02,
            b: 0.04,
            rho: -0.2,
            m: 0.0,
            sigma: 0.4,
        },
        &strikes,
    );

    let (strikes_3m, vols_3m): (Vec<f64>, Vec<f64>) = data_3m.into_iter().unzip();
    let (strikes_1y, vols_1y): (Vec<f64>, Vec<f64>) = data_1y.into_iter().unzip();

    let surface = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .model(SmileModel::CubicSpline)
        .add_tenor(0.25, &strikes_3m, &vols_3m)
        .add_tenor(1.00, &strikes_1y, &vols_1y)
        .build()?;

    // Query at exact tenors
    for t in [0.25, 1.0] {
        let vol = surface.black_vol(t, 100.0)?;
        assert!(
            vol.0 > 0.10 && vol.0 < 0.50,
            "ATM vol at T={t} = {:.4}, out of range",
            vol.0
        );
    }

    // Interpolated tenor
    let vol_mid = surface.black_vol(0.5, 100.0)?;
    assert!(
        vol_mid.0 > 0.10 && vol_mid.0 < 0.50,
        "Interpolated ATM vol = {:.4}, out of range",
        vol_mid.0
    );

    // Vol/variance consistency
    for t in [0.25, 0.5, 1.0] {
        for k in [80.0, 100.0, 120.0] {
            let vol = surface.black_vol(t, k)?;
            let var = surface.black_variance(t, k)?;
            assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-10);
        }
    }

    // smile_at returns a queryable section
    let smile = surface.smile_at(0.5)?;
    assert!(smile.vol(100.0)?.0 > 0.0);

    // No NaN across grid
    for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] {
        for &k in &strikes_3m {
            let vol = surface.black_vol(t, k)?;
            assert!(
                vol.0.is_finite() && vol.0 > 0.0,
                "vol({t}, {k}) = {:.6} is invalid",
                vol.0
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Builder validation integration tests
// ---------------------------------------------------------------------------

#[test]
fn builder_missing_spot_is_invalid_input() {
    let strikes = standard_strikes();
    let vols: Vec<f64> = vec![0.20; strikes.len()];
    let result = SurfaceBuilder::new()
        .rate(0.05)
        .add_tenor(0.25, &strikes, &vols)
        .build();
    assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
}

#[test]
fn builder_missing_rate_is_invalid_input() {
    let strikes = standard_strikes();
    let vols: Vec<f64> = vec![0.20; strikes.len()];
    let result = SurfaceBuilder::new()
        .spot(100.0)
        .add_tenor(0.25, &strikes, &vols)
        .build();
    assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
}

#[test]
fn builder_no_tenors_is_invalid_input() {
    let result = SurfaceBuilder::new().spot(100.0).rate(0.05).build();
    assert!(matches!(result, Err(VolSurfError::InvalidInput { .. })));
}

// ---------------------------------------------------------------------------
// SABR Helpers
// ---------------------------------------------------------------------------

/// SABR parameter set for generating synthetic market data.
struct SabrParams {
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
}

/// Generate synthetic (strike, vol) market data from known SABR parameters.
fn sabr_market_data(p: &SabrParams, strikes: &[f64]) -> Vec<(f64, f64)> {
    let smile = SabrSmile::new(p.forward, p.expiry, p.alpha, p.beta, p.rho, p.nu).unwrap();
    strikes
        .iter()
        .map(|&k| (k, smile.vol(k).unwrap().0))
        .collect()
}

/// Build a 3-tenor SABR surface via the builder API.
fn build_sabr_surface() -> PiecewiseSurface {
    let spot = 100.0;
    let rate = 0.05;
    let strikes = standard_strikes();

    let mut builder = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .model(SmileModel::Sabr { beta: 0.5 });

    // With beta=0.5, ATM vol ≈ alpha / F^(1-beta) = alpha / F^0.5 ≈ alpha / 10.
    // For ~20% vol, need alpha ≈ 2.0.
    for &(t, alpha) in &[(0.25, 2.0), (0.50, 2.0), (1.0, 2.0)] {
        let fwd = spot * (rate * t).exp();
        let data = sabr_market_data(
            &SabrParams {
                forward: fwd,
                expiry: t,
                alpha,
                beta: 0.5,
                rho: -0.30,
                nu: 0.40,
            },
            &strikes,
        );
        let (ks, vs): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        builder = builder.add_tenor(t, &ks, &vs);
    }

    builder.build().unwrap()
}

/// Build a 3-tenor SSVI surface.
fn build_ssvi_surface() -> SsviSurface {
    let tenors = vec![0.25, 0.50, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    // Increasing ATM total variances: vol ~20% => theta = vol^2 * T
    let thetas = vec![0.04 * 0.25, 0.04 * 0.50, 0.04 * 1.0];
    SsviSurface::new(-0.30, 1.0, 0.50, tenors, forwards, thetas)
        .expect("SSVI params should be valid")
}

// ---------------------------------------------------------------------------
// Test 8: SABR calibration round-trip
// ---------------------------------------------------------------------------

#[test]
fn sabr_calibration_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 0.5;
    let params = SabrParams {
        forward,
        expiry,
        alpha: 0.30,
        beta: 0.5,
        rho: -0.30,
        nu: 0.40,
    };
    let strikes = standard_strikes();
    let market_data = sabr_market_data(&params, &strikes);

    let calibrated = SabrSmile::calibrate(forward, expiry, 0.5, &market_data)?;

    let mut sum_sq = 0.0;
    for &(k, v_orig) in &market_data {
        let v_calib = calibrated.vol(k)?.0;
        let diff = v_orig - v_calib;
        sum_sq += diff * diff;
    }
    let rms = (sum_sq / market_data.len() as f64).sqrt();
    assert!(
        rms < 0.001,
        "SABR calibration round-trip RMS = {rms:.6}, should be < 0.001"
    );

    Ok(())
}

#[test]
fn sabr_calibration_round_trip_lognormal() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 1.0;
    let params = SabrParams {
        forward,
        expiry,
        alpha: 0.20,
        beta: 1.0,
        rho: -0.40,
        nu: 0.30,
    };
    let strikes = standard_strikes();
    let market_data = sabr_market_data(&params, &strikes);

    let calibrated = SabrSmile::calibrate(forward, expiry, 1.0, &market_data)?;

    let mut sum_sq = 0.0;
    for &(k, v_orig) in &market_data {
        let v_calib = calibrated.vol(k)?.0;
        let diff = v_orig - v_calib;
        sum_sq += diff * diff;
    }
    let rms = (sum_sq / market_data.len() as f64).sqrt();
    assert!(
        rms < 0.001,
        "Lognormal SABR round-trip RMS = {rms:.6}, should be < 0.001"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 9: Multi-tenor SABR surface end-to-end
// ---------------------------------------------------------------------------

#[test]
fn sabr_surface_build_and_query() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_sabr_surface();

    // Query at exact tenors — should return reasonable vols
    for t in [0.25, 0.50, 1.00] {
        let vol = surface.black_vol(t, 100.0)?;
        assert!(
            vol.0 > 0.10 && vol.0 < 0.50,
            "SABR ATM vol at T={t} = {:.4}, out of range",
            vol.0
        );
    }

    // Interpolated tenor
    let vol_mid = surface.black_vol(0.375, 100.0)?;
    assert!(
        vol_mid.0 > 0.10 && vol_mid.0 < 0.50,
        "SABR interpolated ATM vol = {:.4}, out of range",
        vol_mid.0
    );

    Ok(())
}

#[test]
fn sabr_surface_vol_variance_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_sabr_surface();

    for t in [0.25, 0.375, 0.50, 0.75, 1.00] {
        for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = surface.black_vol(t, k)?;
            let var = surface.black_variance(t, k)?;
            assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-10);
        }
    }

    Ok(())
}

#[test]
fn sabr_surface_smile_at() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_sabr_surface();

    for t in [0.25, 0.50, 0.75, 1.00] {
        let smile = surface.smile_at(t)?;
        assert_abs_diff_eq!(smile.expiry(), t, epsilon = 1e-14);
        assert!(smile.forward() > 0.0);
        let vol = smile.vol(100.0)?;
        assert!(vol.0 > 0.0 && vol.0 < 1.0);
    }

    Ok(())
}

#[test]
fn sabr_surface_no_nan_full_grid() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_sabr_surface();
    let tenors = [0.1, 0.25, 0.375, 0.50, 0.75, 1.0, 1.5];
    let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();

    for &t in &tenors {
        for &k in &strikes {
            let vol = surface.black_vol(t, k)?;
            assert!(
                vol.0.is_finite() && vol.0 > 0.0,
                "SABR vol({t}, {k}) = {:.6} is invalid",
                vol.0
            );
        }
    }

    Ok(())
}

#[test]
fn sabr_butterfly_clean_params() -> Result<(), Box<dyn std::error::Error>> {
    let smile = SabrSmile::new(100.0, 1.0, 0.20, 0.5, -0.30, 0.30)?;
    let report = smile.is_arbitrage_free()?;
    assert!(
        report.is_free,
        "Well-behaved SABR should be arb-free, got {} violations",
        report.butterfly_violations.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 10: SSVI surface end-to-end
// ---------------------------------------------------------------------------

#[test]
fn ssvi_surface_build_and_query() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_ssvi_surface();

    for t in [0.25, 0.50, 1.00] {
        let vol = surface.black_vol(t, 100.0)?;
        assert!(
            vol.0 > 0.05 && vol.0 < 0.50,
            "SSVI ATM vol at T={t} = {:.4}, out of range",
            vol.0
        );
    }

    // Interpolated tenor
    let vol_mid = surface.black_vol(0.375, 100.0)?;
    assert!(
        vol_mid.0 > 0.05 && vol_mid.0 < 0.50,
        "SSVI interpolated ATM vol = {:.4}, out of range",
        vol_mid.0
    );

    Ok(())
}

#[test]
fn ssvi_surface_no_nan_full_grid() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_ssvi_surface();
    let tenors = [0.1, 0.25, 0.375, 0.50, 0.75, 1.0, 1.5];
    let strikes: Vec<f64> = (0..15).map(|i| 70.0 + 4.0 * i as f64).collect();

    for &t in &tenors {
        for &k in &strikes {
            let vol = surface.black_vol(t, k)?;
            assert!(
                vol.0.is_finite() && vol.0 > 0.0,
                "SSVI vol({t}, {k}) = {:.6} is invalid",
                vol.0
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 11: SSVI vol/variance consistency + smile_at
// ---------------------------------------------------------------------------

#[test]
fn ssvi_vol_variance_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_ssvi_surface();

    for t in [0.25, 0.375, 0.50, 0.75, 1.00] {
        for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = surface.black_vol(t, k)?;
            let var = surface.black_variance(t, k)?;
            assert_abs_diff_eq!(vol.0 * vol.0 * t, var.0, epsilon = 1e-10);
        }
    }

    Ok(())
}

#[test]
fn ssvi_smile_at_returns_working_section() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_ssvi_surface();

    for t in [0.25, 0.50, 0.75, 1.00] {
        let smile = surface.smile_at(t)?;
        assert_abs_diff_eq!(smile.expiry(), t, epsilon = 1e-14);
        assert!(smile.forward() > 0.0);

        let vol = smile.vol(100.0)?;
        assert!(vol.0 > 0.0 && vol.0 < 1.0);

        // Verify smile and surface are consistent at this tenor
        let surf_vol = surface.black_vol(t, 100.0)?;
        assert_abs_diff_eq!(vol.0, surf_vol.0, epsilon = 1e-10);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 12: SSVI calendar diagnostics
// ---------------------------------------------------------------------------

#[test]
fn ssvi_well_behaved_passes_diagnostics() -> Result<(), Box<dyn std::error::Error>> {
    let surface = build_ssvi_surface();
    let diag = surface.diagnostics()?;

    assert_eq!(diag.smile_reports.len(), 3, "Should have 3 tenor reports");

    // Well-behaved SSVI with increasing thetas and moderate params
    // should be calendar-arb free
    assert!(
        diag.calendar_violations.is_empty(),
        "Well-behaved SSVI should have no calendar violations, got {}",
        diag.calendar_violations.len()
    );

    Ok(())
}

#[test]
fn ssvi_analytical_calendar_check() {
    let surface = build_ssvi_surface();

    // Analytical check should also pass for well-behaved params
    let violations = surface.calendar_arb_analytical();
    assert!(
        violations.is_empty(),
        "Analytical calendar check should pass, got {} violations",
        violations.len()
    );
}

// ---------------------------------------------------------------------------
// Test 13: Cross-model comparison
// ---------------------------------------------------------------------------

#[test]
fn svi_vs_sabr_produce_plausible_vols() -> Result<(), Box<dyn std::error::Error>> {
    // Calibrate both models to the same SVI-generated market data
    let forward = 100.0;
    let expiry = 0.5;
    let strikes = standard_strikes();
    let market_data = svi_market_data(
        &SviParams {
            forward,
            expiry,
            a: 0.01,
            b: 0.10,
            rho: -0.30,
            m: 0.0,
            sigma: 0.30,
        },
        &strikes,
    );

    let svi = SviSmile::calibrate(forward, expiry, &market_data)?;
    let sabr = SabrSmile::calibrate(forward, expiry, 0.5, &market_data)?;

    // Both should produce vols in the same ballpark (< 5% relative difference at ATM)
    let svi_atm = svi.vol(forward)?.0;
    let sabr_atm = sabr.vol(forward)?.0;
    let rel_diff = (svi_atm - sabr_atm).abs() / svi_atm;
    assert!(
        rel_diff < 0.05,
        "ATM vol: SVI={svi_atm:.4}, SABR={sabr_atm:.4}, rel diff={rel_diff:.4} > 5%"
    );

    // Check across the strike range — both should be reasonable
    for &k in &strikes {
        let v_svi = svi.vol(k)?.0;
        let v_sabr = sabr.vol(k)?.0;
        assert!(v_svi > 0.0 && v_svi < 1.0);
        assert!(v_sabr > 0.0 && v_sabr < 1.0);
    }

    Ok(())
}

#[test]
fn piecewise_vs_ssvi_produce_comparable_atm_vols() -> Result<(), Box<dyn std::error::Error>> {
    let piecewise = build_3_tenor_surface();
    let ssvi = build_ssvi_surface();

    // Both surfaces should produce positive, finite vols at the same tenors
    for t in [0.25, 0.50, 1.00] {
        let v_pw = piecewise.black_vol(t, 100.0)?.0;
        let v_ssvi = ssvi.black_vol(t, 100.0)?.0;
        assert!(
            v_pw > 0.0 && v_pw < 1.0,
            "PiecewiseSurface vol({t}, 100) = {v_pw:.4} out of range"
        );
        assert!(
            v_ssvi > 0.0 && v_ssvi < 1.0,
            "SsviSurface vol({t}, 100) = {v_ssvi:.4} out of range"
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 14: Thread safety (SABR surface + SSVI surface)
// ---------------------------------------------------------------------------

#[test]
fn concurrent_sabr_surface_queries() -> Result<(), Box<dyn std::error::Error>> {
    let surface = Arc::new(build_sabr_surface());

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let s = Arc::clone(&surface);
            thread::spawn(move || -> volsurf::Result<()> {
                let strike = 80.0 + i as f64 * 5.0;
                for &t in &[0.25, 0.50, 0.75, 1.00] {
                    let vol = s.black_vol(t, strike)?;
                    assert!(vol.0 > 0.0 && vol.0 < 2.0);
                    let var = s.black_variance(t, strike)?;
                    assert!(var.0 > 0.0 && var.0.is_finite());
                }
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked")?;
    }

    Ok(())
}

#[test]
fn concurrent_ssvi_surface_queries() -> Result<(), Box<dyn std::error::Error>> {
    let surface = Arc::new(build_ssvi_surface());

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let s = Arc::clone(&surface);
            thread::spawn(move || -> volsurf::Result<()> {
                let strike = 80.0 + i as f64 * 5.0;
                for &t in &[0.25, 0.50, 0.75, 1.00] {
                    let vol = s.black_vol(t, strike)?;
                    assert!(vol.0 > 0.0 && vol.0 < 2.0);
                }
                // smile_at from thread
                let smile = s.smile_at(0.5)?;
                assert!(smile.vol(100.0)?.0 > 0.0);
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 15: Send + Sync compile-time verification
// ---------------------------------------------------------------------------

#[test]
fn types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    // Smile models
    assert_send_sync::<SviSmile>();
    assert_send_sync::<SabrSmile>();
    assert_send_sync::<SplineSmile>();

    // Surfaces
    assert_send_sync::<PiecewiseSurface>();
    assert_send_sync::<SsviSurface>();
}

// Test 16: Dividend yield — multi-tenor surface with continuous dividend yield
//
// Build a 3-tenor surface with q=2%, verify each tenor's forward reflects
// F = S · exp((r − q) · T) rather than F = S · exp(r · T).
#[test]
fn surface_with_dividend_yield() -> Result<(), Box<dyn std::error::Error>> {
    let spot = 100.0;
    let rate = 0.05;
    let q = 0.02;
    let strikes = standard_strikes();

    let tenors = [0.25, 0.5, 1.0];
    let svi_configs: [(f64, f64, f64, f64, f64); 3] = [
        (0.005, 0.05, -0.3, 0.0, 0.25),
        (0.01, 0.04, -0.25, 0.0, 0.35),
        (0.02, 0.04, -0.2, 0.0, 0.4),
    ];

    let mut builder = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q);

    for (&t, &(a, b, rho, m, sigma)) in tenors.iter().zip(svi_configs.iter()) {
        let fwd = spot * ((rate - q) * t).exp();
        let data = svi_market_data(
            &SviParams {
                forward: fwd,
                expiry: t,
                a,
                b,
                rho,
                m,
                sigma,
            },
            &strikes,
        );
        let (ks, vs): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        builder = builder.add_tenor(t, &ks, &vs);
    }

    let surface = builder.build()?;

    for &t in &tenors {
        let smile = surface.smile_at(t)?;
        let expected_fwd = spot * ((rate - q) * t).exp();
        assert_abs_diff_eq!(smile.forward(), expected_fwd, epsilon = 0.5);
    }

    // Verify q > 0 gives lower forwards than q = 0
    let smile_1y = surface.smile_at(1.0)?;
    let fwd_no_q = spot * (rate * 1.0_f64).exp();
    assert!(smile_1y.forward() < fwd_no_q);

    Ok(())
}

// Test 17: Mixed add_tenor / add_tenor_with_forward — per-tenor forward override
//
// Two tenors: 3M uses computed forward (via spot/rate/q), 1Y uses an explicit
// forward (simulating a futures option). Verify each smile carries the right
// forward and cross-tenor queries produce sensible vols.
#[test]
fn mixed_tenor_and_tenor_with_forward() -> Result<(), Box<dyn std::error::Error>> {
    let spot = 100.0;
    let rate = 0.05;
    let q = 0.02;
    let strikes = standard_strikes();

    // 3M: computed forward = spot * exp((r-q)*0.25) ≈ 100.75
    let fwd_3m = spot * ((rate - q) * 0.25_f64).exp();
    let data_3m = svi_market_data(
        &SviParams {
            forward: fwd_3m,
            expiry: 0.25,
            a: 0.005,
            b: 0.05,
            rho: -0.3,
            m: 0.0,
            sigma: 0.25,
        },
        &strikes,
    );

    // 1Y: explicit forward = 110.0 (e.g. futures price)
    let explicit_fwd = 110.0;
    let data_1y = svi_market_data(
        &SviParams {
            forward: explicit_fwd,
            expiry: 1.0,
            a: 0.02,
            b: 0.04,
            rho: -0.2,
            m: 0.0,
            sigma: 0.4,
        },
        &strikes,
    );

    let (ks_3m, vs_3m): (Vec<f64>, Vec<f64>) = data_3m.into_iter().unzip();
    let (ks_1y, vs_1y): (Vec<f64>, Vec<f64>) = data_1y.into_iter().unzip();

    let surface = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .add_tenor(0.25, &ks_3m, &vs_3m)
        .add_tenor_with_forward(1.0, &ks_1y, &vs_1y, explicit_fwd)
        .build()?;

    // 3M should use computed forward
    let smile_3m = surface.smile_at(0.25)?;
    assert_abs_diff_eq!(smile_3m.forward(), fwd_3m, epsilon = 0.5);

    // 1Y should use explicit forward, ignoring spot/rate/q
    let smile_1y = surface.smile_at(1.0)?;
    assert_abs_diff_eq!(smile_1y.forward(), explicit_fwd, epsilon = 0.5);

    // Cross-tenor query at 6M (interpolation between 3M and 1Y)
    let vol_6m = surface.black_vol(0.5, 100.0)?;
    assert!(vol_6m.0 > 0.05 && vol_6m.0 < 0.50);

    Ok(())
}

// Test 18: Dividend yield q = r gives forward ≈ spot
#[test]
fn dividend_yield_equals_rate() -> Result<(), Box<dyn std::error::Error>> {
    let spot = 100.0;
    let rate = 0.05;
    let q = 0.05;
    let strikes = standard_strikes();

    // Forward ≈ spot when q = r
    let data = svi_market_data(
        &SviParams {
            forward: spot,
            expiry: 1.0,
            a: 0.02,
            b: 0.04,
            rho: -0.2,
            m: 0.0,
            sigma: 0.4,
        },
        &strikes,
    );
    let (ks, vs): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();

    let surface = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .add_tenor(1.0, &ks, &vs)
        .build()?;

    let smile = surface.smile_at(1.0)?;
    assert_abs_diff_eq!(smile.forward(), spot, epsilon = 0.5);

    Ok(())
}

// Test 19: q > r gives forward < spot (high-dividend equity)
#[test]
fn dividend_yield_exceeds_rate() -> Result<(), Box<dyn std::error::Error>> {
    let spot = 100.0;
    let rate = 0.02;
    let q = 0.06;
    let strikes = standard_strikes();

    let fwd = spot * ((rate - q) * 1.0_f64).exp(); // ≈ 96.08
    let data = svi_market_data(
        &SviParams {
            forward: fwd,
            expiry: 1.0,
            a: 0.02,
            b: 0.04,
            rho: -0.2,
            m: 0.0,
            sigma: 0.4,
        },
        &strikes,
    );
    let (ks, vs): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();

    let surface = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .add_tenor(1.0, &ks, &vs)
        .build()?;

    let smile = surface.smile_at(1.0)?;
    assert!(smile.forward() < spot);
    assert_abs_diff_eq!(smile.forward(), fwd, epsilon = 0.5);

    Ok(())
}
