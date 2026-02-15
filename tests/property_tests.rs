//! Property-based tests using proptest.
//!
//! These tests verify invariant properties across random inputs rather than
//! testing fixed examples. They help catch edge cases and ensure robustness.

use proptest::prelude::*;
use volsurf::smile::{SmileSection, SplineSmile, SviSmile};
use volsurf::surface::{PiecewiseSurface, VolSurface};

// --- Property Test 1: SVI vol non-negativity ---

proptest! {
    /// SVI smile should always return non-negative volatilities for valid
    /// parameters and positive strikes.
    ///
    /// Generates random valid SVI parameters and verifies that vol(K) >= 0
    /// for strikes in the range [80, 120] around forward=100.
    #[test]
    fn svi_vol_is_non_negative(
        a in 0.01_f64..0.10,
        b in 0.05_f64..0.30,
        rho in -0.9_f64..0.9,
        m in -0.5_f64..0.5,
        sigma in 0.05_f64..0.5,
    ) {
        let forward = 100.0;
        let expiry = 0.25;

        // Try to create SVI smile (may fail for some random param combos)
        let smile_result = SviSmile::new(forward, expiry, a, b, rho, m, sigma);

        // If construction succeeds, verify vol is non-negative
        if let Ok(smile) = smile_result {
            for strike in 80..=120 {
                let k = strike as f64;
                let vol = smile.vol(k).unwrap();
                prop_assert!(
                    vol.0 >= 0.0,
                    "vol should be non-negative, got {} at strike {}",
                    vol.0,
                    k
                );
            }
        }
        // If construction fails, that's fine (invalid params), just skip
    }
}

// --- Property Test 2: SVI variance-vol consistency ---

proptest! {
    /// SVI variance should equal vol^2 * T for all valid parameter sets.
    ///
    /// This is a fundamental relationship that must hold everywhere.
    #[test]
    fn svi_variance_equals_vol_squared_times_t(
        a in 0.01_f64..0.10,
        b in 0.05_f64..0.30,
        rho in -0.9_f64..0.9,
        m in -0.5_f64..0.5,
        sigma in 0.05_f64..0.5,
    ) {
        let forward = 100.0;
        let expiry = 0.25;

        // Skip invalid param combinations
        let smile_result = SviSmile::new(forward, expiry, a, b, rho, m, sigma);
        prop_assume!(smile_result.is_ok());

        let smile = smile_result.unwrap();

        for strike in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = smile.vol(strike).unwrap();
            let var = smile.variance(strike).unwrap();
            let expected_var = vol.0 * vol.0 * expiry;

            prop_assert!(
                (var.0 - expected_var).abs() < 1e-12,
                "variance should equal vol^2 * T, got var={} vs expected={}",
                var.0,
                expected_var
            );
        }
    }
}

// --- Property Test 3: SplineSmile passes through knots ---

proptest! {
    /// SplineSmile should exactly interpolate through its knot points.
    ///
    /// Creates a spline with 5 knots at variance=0.04 and verifies that
    /// evaluating at each knot strike returns the exact variance.
    #[test]
    fn spline_passes_through_knots(
        _seed in 0..100_u32, // Just to run test multiple times
    ) {
        let forward = 100.0;
        let expiry = 1.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let variances = vec![0.04, 0.04, 0.04, 0.04, 0.04];

        let smile = SplineSmile::new(forward, expiry, strikes.clone(), variances.clone())
            .unwrap();

        for (strike, expected_var) in strikes.iter().zip(variances.iter()) {
            let vol = smile.vol(*strike).unwrap();
            let actual_vol = (expected_var / expiry).sqrt();
            prop_assert!(
                (vol.0 - actual_vol).abs() < 1e-12,
                "spline should pass through knot at K={}, got vol={} vs expected={}",
                strike,
                vol.0,
                actual_vol
            );
        }
    }
}

// --- Property Test 4: PiecewiseSurface at exact tenor matches stored smile ---

proptest! {
    /// When querying a PiecewiseSurface at an exact tenor, the result should
    /// match the stored smile at that tenor.
    #[test]
    fn piecewise_surface_exact_tenor_matches_smile(
        a in 0.02_f64..0.08,
        b in 0.1_f64..0.4,
        rho in -0.8_f64..0.8,
        m in -0.3_f64..0.3,
        sigma in 0.1_f64..0.3,
    ) {
        let forward = 100.0;
        let expiry = 0.5;

        // Try to create SVI smile
        let smile_result = SviSmile::new(forward, expiry, a, b, rho, m, sigma);
        prop_assume!(smile_result.is_ok());

        let smile = smile_result.unwrap();

        // Build a 1-tenor surface
        let surface = PiecewiseSurface::new(
            vec![expiry],
            vec![Box::new(smile.clone())],
        ).unwrap();

        // Query at the exact tenor for various strikes
        for strike in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol_from_surface = surface.black_vol(expiry, strike).unwrap();
            let vol_from_smile = smile.vol(strike).unwrap();

            prop_assert!(
                (vol_from_surface.0 - vol_from_smile.0).abs() < 1e-10,
                "surface vol at exact tenor should match smile, got {} vs {} at strike {}",
                vol_from_surface.0,
                vol_from_smile.0,
                strike
            );
        }
    }
}
