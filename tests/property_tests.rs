//! Property-based tests using proptest.
//!
//! These tests verify invariant properties across random inputs rather than
//! testing fixed examples. They help catch edge cases and ensure robustness.

use proptest::prelude::*;
use volsurf::conventions;
use volsurf::smile::{SabrSmile, SmileSection, SplineSmile, SviSmile};
use volsurf::surface::{PiecewiseSurface, SsviSurface, SurfaceBuilder, VolSurface};

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
    /// Generates random variance values at fixed strikes and verifies that
    /// evaluating at each knot strike returns the exact knot variance.
    #[test]
    fn spline_passes_through_knots(
        v0 in 0.01_f64..0.20,
        v1 in 0.01_f64..0.20,
        v2 in 0.01_f64..0.20,
        v3 in 0.01_f64..0.20,
    ) {
        let forward = 100.0;
        let expiry = 1.0;
        // Variances must be positive (enforced by range).
        // Strikes must be strictly increasing.
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        // Use sorted variances to keep shape well-behaved, but randomize values.
        let variances = vec![v0, v1, v2, v3, v0]; // Symmetric with random levels

        let smile = SplineSmile::new(forward, expiry, strikes.clone(), variances.clone())
            .unwrap();

        for (strike, expected_var) in strikes.iter().zip(variances.iter()) {
            let var = smile.variance(*strike).unwrap();
            prop_assert!(
                (var.0 - expected_var).abs() < 1e-12,
                "spline should pass through knot at K={}, got var={} vs expected={}",
                strike,
                var.0,
                expected_var
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

// --- Property Test 5: SABR vol non-negativity ---

proptest! {
    /// SABR smile should always return non-negative volatilities for valid
    /// parameters and positive strikes.
    ///
    /// Generates random valid SABR parameters and verifies that vol(K) >= 0
    /// for strikes in [80, 120]. SABR vol() can return Err in deep wings
    /// (Hagan approximation breakdown), so we skip errors gracefully.
    #[test]
    fn sabr_vol_is_non_negative(
        alpha in 0.05_f64..0.5,
        beta in 0.0_f64..1.0,
        rho in -0.9_f64..0.9,
        nu in 0.01_f64..1.0,
    ) {
        let forward = 100.0;
        let expiry = 0.25;

        let sabr_result = SabrSmile::new(forward, expiry, alpha, beta, rho, nu);
        prop_assume!(sabr_result.is_ok());

        let sabr = sabr_result.unwrap();

        for strike in 80..=120 {
            let k = strike as f64;
            if let Ok(v) = sabr.vol(k) {
                prop_assert!(
                    v.0 >= 0.0,
                    "SABR vol should be non-negative, got {} at strike {}",
                    v.0,
                    k
                );
            }
            // If vol() returns Err, that's acceptable (Hagan wing breakdown)
        }
    }
}

// --- Property Test 6: SABR density integrates approximately to one ---

proptest! {
    /// The risk-neutral density from Breeden-Litzenberger (d²C/dK²) should
    /// integrate to approximately 1 over a sufficiently wide strike range.
    ///
    /// Uses a generous tolerance [0.5, 1.5] because:
    /// - Breeden-Litzenberger is a finite-difference approximation
    /// - We integrate over a truncated domain [50, 200]
    /// - Some SABR parameter combos have fat tails
    ///
    /// Parameter ranges are restricted to lognormal-like regimes (beta >= 0.5)
    /// to ensure the density is well-spread over [50, 200]. With beta near 0
    /// (normal model) and low vol-of-vol, the density concentrates too narrowly.
    #[test]
    fn sabr_density_integrates_approximately_to_one(
        alpha in 0.1_f64..0.5,
        beta in 0.5_f64..1.0,
        rho in -0.9_f64..0.9,
        nu in 0.1_f64..1.0,
    ) {
        let forward = 100.0;
        let expiry = 0.25;

        let sabr_result = SabrSmile::new(forward, expiry, alpha, beta, rho, nu);
        prop_assume!(sabr_result.is_ok());

        let sabr = sabr_result.unwrap();

        // Trapezoidal integration of density from 50 to 200
        let n_steps = 200;
        let k_lo = 50.0_f64;
        let k_hi = 200.0_f64;
        let dk = (k_hi - k_lo) / n_steps as f64;

        let mut integral = 0.0;
        let mut any_err = false;

        for i in 0..=n_steps {
            let k = k_lo + i as f64 * dk;
            match sabr.density(k) {
                Ok(d) => {
                    let weight = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
                    integral += weight * d * dk;
                }
                Err(_) => {
                    any_err = true;
                    break;
                }
            }
        }

        // Skip if any density call failed (Hagan wing breakdown)
        prop_assume!(!any_err);

        prop_assert!(
            integral > 0.5 && integral < 1.5,
            "density integral should be approximately 1, got {}",
            integral
        );
    }
}

// --- Property Test 7: SSVI total variance is positive ---

proptest! {
    /// SSVI total variance w(k, theta) should always be positive for valid
    /// parameters. This is a fundamental requirement: variance cannot be negative.
    #[test]
    fn ssvi_total_variance_positive(
        rho in -0.9_f64..0.9,
        eta in 0.1_f64..2.0,
        gamma in 0.0_f64..1.0,
        theta1 in 0.01_f64..0.1,
    ) {
        let surface_result = SsviSurface::new(
            rho, eta, gamma,
            vec![0.25],
            vec![100.0],
            vec![theta1],
        );
        prop_assume!(surface_result.is_ok());

        let surface = surface_result.unwrap();

        for strike in (60..=140).step_by(10) {
            let k = strike as f64;
            let var = surface.black_variance(0.25, k).unwrap();
            prop_assert!(
                var.0 > 0.0,
                "SSVI variance should be positive, got {} at strike {}",
                var.0,
                k
            );
        }
    }
}

// --- Property Test 8: SSVI variance monotone in time ---

proptest! {
    /// Total variance must be non-decreasing in time (no calendar arbitrage).
    ///
    /// For any fixed strike K, w(T2, K) >= w(T1, K) when T2 > T1. This
    /// follows from the requirement that call prices are non-decreasing in
    /// maturity.
    #[test]
    fn ssvi_variance_monotone_in_time(
        rho in -0.9_f64..0.9,
        eta in 0.1_f64..2.0,
        gamma in 0.0_f64..1.0,
        theta1 in 0.01_f64..0.1,
        theta_incr in 0.01_f64..0.3,
    ) {
        let theta2 = theta1 + theta_incr;

        let surface_result = SsviSurface::new(
            rho, eta, gamma,
            vec![0.25, 1.0],
            vec![100.0, 100.0],
            vec![theta1, theta2],
        );
        prop_assume!(surface_result.is_ok());

        let surface = surface_result.unwrap();

        for strike in (70..=130).step_by(10) {
            let k = strike as f64;
            let var_short = surface.black_variance(0.25, k).unwrap();
            let var_long = surface.black_variance(1.0, k).unwrap();
            prop_assert!(
                var_long.0 >= var_short.0,
                "variance at T=1.0 ({}) should be >= variance at T=0.25 ({}) at strike {}",
                var_long.0,
                var_short.0,
                k
            );
        }
    }
}

// --- Property Test 9: SSVI calendar arbitrage analytical check clean ---

proptest! {
    /// SSVI surfaces with strictly increasing thetas should be free of
    /// calendar arbitrage as detected by the analytical g-function check
    /// (Gatheral-Jacquier Theorem 4.2).
    #[test]
    fn ssvi_calendar_arb_analytical_empty(
        rho in -0.9_f64..0.9,
        eta in 0.1_f64..2.0,
        gamma in 0.0_f64..1.0,
        theta1 in 0.01_f64..0.1,
        theta_incr in 0.01_f64..0.3,
    ) {
        let theta2 = theta1 + theta_incr;

        let surface_result = SsviSurface::new(
            rho, eta, gamma,
            vec![0.25, 1.0],
            vec![100.0, 100.0],
            vec![theta1, theta2],
        );
        prop_assume!(surface_result.is_ok());

        let surface = surface_result.unwrap();
        let violations = surface.calendar_arb_analytical();

        prop_assert!(
            violations.is_empty(),
            "valid SSVI should be calendar-arb-free, got {} violations",
            violations.len()
        );
    }
}

// --- Property Test 10: SurfaceBuilder round-trip vol close ---

proptest! {
    /// Volatilities queried from a built surface should be close to the
    /// input vols used to construct it. The tolerance of 0.02 (2 vol points)
    /// accounts for calibration imperfection in the parametric fit.
    #[test]
    fn builder_round_trip_vol_close(
        atm_vol in 0.15_f64..0.35,
        skew in -0.05_f64..0.05,
    ) {
        let spot = 100.0;
        let rate = 0.05;
        let expiry = 0.25;

        let strikes = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
        let vols: Vec<f64> = strikes
            .iter()
            .map(|&k| atm_vol + skew * (k - 100.0) / 100.0)
            .collect();

        let build_result = SurfaceBuilder::new()
            .spot(spot)
            .rate(rate)
            .add_tenor(expiry, &strikes, &vols)
            .build();
        prop_assume!(build_result.is_ok());

        let surface = build_result.unwrap();

        for (i, &k) in strikes.iter().enumerate() {
            let queried = surface.black_vol(expiry, k).unwrap();
            prop_assert!(
                (queried.0 - vols[i]).abs() < 0.02,
                "round-trip vol at K={} should be within 0.02, got {} vs input {}",
                k,
                queried.0,
                vols[i]
            );
        }
    }
}

// --- Property Test 11: log_moneyness / moneyness consistency ---

proptest! {
    /// exp(log_moneyness(K, F)) should equal moneyness(K, F) for all valid inputs.
    #[test]
    fn log_moneyness_exp_equals_moneyness(
        strike in 0.01_f64..1e6,
        forward in 0.01_f64..1e6,
    ) {
        let k = conventions::log_moneyness(strike, forward).unwrap();
        let m = conventions::moneyness(strike, forward).unwrap();
        prop_assert!(
            (k.exp() - m).abs() < 1e-10,
            "exp(log_moneyness) should equal moneyness: exp({}) = {} vs {}",
            k, k.exp(), m
        );
    }

    #[test]
    fn moneyness_positive_inputs_always_succeed(
        strike in 0.01_f64..1e6,
        forward in 0.01_f64..1e6,
    ) {
        let k = conventions::log_moneyness(strike, forward);
        let m = conventions::moneyness(strike, forward);
        prop_assert!(k.is_ok());
        prop_assert!(m.is_ok());
        prop_assert!(m.unwrap() > 0.0);
    }

    #[test]
    fn forward_price_positive_inputs_succeed(
        spot in 0.01_f64..1e4,
        rate in -0.1_f64..0.2,
        q in -0.05_f64..0.1,
        expiry in 0.01_f64..30.0,
    ) {
        let f = conventions::forward_price(spot, rate, q, expiry);
        prop_assert!(f.is_ok());
        let fwd = f.unwrap();
        prop_assert!(fwd > 0.0);
        prop_assert!(fwd.is_finite());
    }
}
