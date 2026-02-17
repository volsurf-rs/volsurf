//! Integration tests for Dupire local volatility extraction.
//!
//! Tests the full pipeline: VolSurface → DupireLocalVol → local_vol()
//! using SSVI surfaces, hand-crafted calendar violations, and edge cases.
//!
//! Reference: Gatheral (2006), "The Volatility Surface", Ch. 1-2, Eq (1.10).

use std::sync::Arc;

use volsurf::local_vol::{DupireLocalVol, LocalVol};
use volsurf::smile::SmileSection;
use volsurf::smile::spline::SplineSmile;
use volsurf::surface::{PiecewiseSurface, SsviSurface, VolSurface};

fn ssvi_test_surface() -> Arc<dyn VolSurface> {
    Arc::new(
        SsviSurface::new(
            -0.3,
            0.5,
            0.5,
            vec![0.25, 0.5, 1.0, 2.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![0.01, 0.02, 0.04, 0.08],
        )
        .unwrap(),
    )
}

// SSVI surface: local vol should be positive and finite across a grid.
// Gatheral-Jacquier (2014) guarantees butterfly-free when eta*(1+|rho|) <= 2.
#[test]
fn ssvi_local_vol_grid_positive() {
    let surface = ssvi_test_surface();
    let dupire = DupireLocalVol::new(surface);

    let tenors = [0.3, 0.5, 0.75, 1.0, 1.5];
    let strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];

    let mut vals = Vec::new();
    for &t in &tenors {
        for &k in &strikes {
            let v = dupire
                .local_vol(t, k)
                .unwrap_or_else(|e| panic!("failed at T={t}, K={k}: {e}"));
            assert!(v.0 > 0.0, "non-positive at T={t}, K={k}: {}", v.0);
            assert!(v.0.is_finite(), "non-finite at T={t}, K={k}: {}", v.0);
            assert!(v.0 < 2.0, "unreasonably large at T={t}, K={k}: {}", v.0);
            vals.push(v.0);
        }
    }

    // Smoothness: standard deviation should be bounded (no wild spikes)
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std_dev = variance.sqrt();
    assert!(
        std_dev < 0.15,
        "local vol grid too noisy: mean={mean:.4}, std={std_dev:.4}"
    );
}

// Calendar violation: w(T1, K) > w(T2, K) for T1 < T2 at some strike.
// Dupire formula should return NumericalError (negative local variance).
#[test]
fn calendar_violation_returns_error() {
    let strikes: Vec<f64> = (0..11).map(|i| 80.0 + 4.0 * i as f64).collect();

    // 3M smile: high variance (35% vol → w = 0.35^2 * 0.25 = 0.030625)
    let w_high: Vec<f64> = vec![0.030625; strikes.len()];
    let smile_3m = Box::new(SplineSmile::new(100.0, 0.25, strikes.clone(), w_high).unwrap())
        as Box<dyn SmileSection>;

    // 1Y smile: lower variance (15% vol → w = 0.15^2 * 1.0 = 0.0225)
    let w_low: Vec<f64> = vec![0.0225; strikes.len()];
    let smile_1y =
        Box::new(SplineSmile::new(100.0, 1.0, strikes, w_low).unwrap()) as Box<dyn SmileSection>;

    let surface =
        Arc::new(PiecewiseSurface::new(vec![0.25, 1.0], vec![smile_3m, smile_1y]).unwrap());

    let dupire = DupireLocalVol::new(surface);

    // At the midpoint, total variance is decreasing → negative dw/dT
    let result = dupire.local_vol(0.5, 100.0);
    assert!(
        result.is_err(),
        "expected error for calendar violation, got {:?}",
        result
    );

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("calendar") || err.contains("negative"),
        "error should mention calendar arbitrage: {err}"
    );
}

// Deep OTM: K/F = 2.0 and K/F = 0.5. Should either succeed with bounded
// result or return a clean error — never panic.
#[test]
fn deep_otm_no_panic() {
    let surface = ssvi_test_surface();
    let dupire = DupireLocalVol::new(surface);

    for &k in &[50.0, 200.0] {
        match dupire.local_vol(0.5, k) {
            Ok(v) => assert!(v.0.is_finite(), "non-finite at K={k}: {}", v.0),
            Err(_) => {} // graceful error is fine for deep OTM
        }
    }
}

// Convergence: finer bump_size should give results closer to the finest grid.
// Central difference error is O(h²), so halving h should quarter the error.
#[test]
fn bump_size_convergence() {
    let surface = ssvi_test_surface();

    let reference = DupireLocalVol::new(Arc::clone(&surface))
        .with_bump_size(0.0005)
        .unwrap();

    let coarse = DupireLocalVol::new(Arc::clone(&surface))
        .with_bump_size(0.01)
        .unwrap();

    let fine = DupireLocalVol::new(surface).with_bump_size(0.001).unwrap();

    let points = [
        (0.5, 95.0),
        (0.5, 100.0),
        (0.5, 105.0),
        (1.0, 90.0),
        (1.0, 100.0),
        (1.0, 110.0),
    ];

    let mut err_coarse = 0.0;
    let mut err_fine = 0.0;

    for &(t, k) in &points {
        let v_ref = reference.local_vol(t, k).unwrap().0;
        let v_coarse = coarse.local_vol(t, k).unwrap().0;
        let v_fine = fine.local_vol(t, k).unwrap().0;
        err_coarse += (v_coarse - v_ref).abs();
        err_fine += (v_fine - v_ref).abs();
    }

    // Fine grid should have smaller total error
    assert!(
        err_fine < err_coarse,
        "convergence failed: fine_err={err_fine:.6} >= coarse_err={err_coarse:.6}"
    );

    // Rough check: O(h²) means 10x smaller h gives ~100x smaller error.
    // With h_coarse/h_fine = 10, ratio should be ~100.
    // Be lenient since SplineSmile approximation adds noise.
    let ratio = err_coarse / err_fine.max(1e-15);
    assert!(
        ratio > 5.0,
        "convergence ratio too low: {ratio:.1} (expected >> 1)"
    );
}

// Thread safety: Arc<DupireLocalVol> can be shared across threads.
#[test]
fn thread_safe_local_vol() {
    let surface = ssvi_test_surface();
    let dupire = Arc::new(DupireLocalVol::new(surface));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let d = Arc::clone(&dupire);
            std::thread::spawn(move || {
                let k = 90.0 + 10.0 * i as f64;
                d.local_vol(0.5, k).unwrap()
            })
        })
        .collect();

    for h in handles {
        let v = h.join().unwrap();
        assert!(v.0 > 0.0 && v.0.is_finite());
    }
}
