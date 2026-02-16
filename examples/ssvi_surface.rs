//! SSVI surface: construction, calibration, and diagnostics.
//!
//! Demonstrates the SSVI global parameterization (Gatheral-Jacquier, 2014):
//!   1. Construct an SSVI surface from known parameters
//!   2. Query vol/variance across a (T, K) grid
//!   3. Extract smile sections at various tenors
//!   4. Run butterfly and calendar arbitrage diagnostics
//!   5. Calibrate SSVI from multi-tenor market data
//!   6. Compare with a PiecewiseSurface approach
//!
//! Run with: `cargo run --example ssvi_surface`

use volsurf::surface::{SsviSurface, SurfaceBuilder, VolSurface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------
    // 1. Construct an SSVI surface from known parameters
    // ---------------------------------------------------------------

    // Three global parameters control the entire surface
    let rho = -0.3;   // skew: negative for equity
    let eta = 0.5;    // smile amplitude
    let gamma = 0.5;  // term structure decay

    // Per-tenor data: expiries, forwards, ATM total variances
    let tenors = vec![0.25, 0.50, 1.0, 2.0];
    let forwards = vec![100.0, 100.0, 100.0, 100.0];
    let thetas = vec![0.01, 0.02, 0.04, 0.08]; // θ_i = σ²_ATM,i · T_i

    let surface = SsviSurface::new(
        rho, eta, gamma,
        tenors.clone(),
        forwards.clone(),
        thetas.clone(),
    )?;

    println!("SSVI surface: rho={:.2}, eta={:.2}, gamma={:.2}", rho, eta, gamma);
    println!("Tenors: {:?}", tenors);
    println!("Thetas: {:?}\n", thetas);

    // ---------------------------------------------------------------
    // 2. Query vol across the surface
    // ---------------------------------------------------------------

    println!("--- Vol grid (expiry x strike) ---\n");

    let query_tenors = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00];
    let query_strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

    print!("{:>8}", "T\\K");
    for &k in &query_strikes {
        print!("{k:>8.0}");
    }
    println!();
    println!("{}", "-".repeat(64));

    for &t in &query_tenors {
        print!("{t:>8.2}");
        for &k in &query_strikes {
            let vol = surface.black_vol(t, k)?;
            print!("{:>8.2}%", vol.0 * 100.0);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // 3. Vol/variance consistency
    // ---------------------------------------------------------------

    println!("\n--- Vol/Variance consistency ---\n");
    let t = 0.5;
    let k = 100.0;
    let vol = surface.black_vol(t, k)?;
    let var = surface.black_variance(t, k)?;
    println!("At T={t}, K={k}: vol={:.6}, var={:.6}, vol^2*T={:.6}",
        vol.0, var.0, vol.0 * vol.0 * t);

    // ---------------------------------------------------------------
    // 4. Extract smile sections at various tenors
    // ---------------------------------------------------------------

    println!("\n--- Smile sections ---\n");

    for &t in &[0.25, 1.0, 2.0] {
        let smile = surface.smile_at(t)?;
        let atm_vol = smile.vol(smile.forward())?;
        let density = smile.density(smile.forward())?;
        println!(
            "T={t:.2}: forward={:.2}, ATM vol={:.4}%, density={:.6}",
            smile.forward(),
            atm_vol.0 * 100.0,
            density
        );
    }

    // Interpolated tenor (not in original data)
    let smile_075 = surface.smile_at(0.75)?;
    println!(
        "\nInterpolated T=0.75: forward={:.2}, ATM vol={:.4}%",
        smile_075.forward(),
        smile_075.vol(smile_075.forward())?.0 * 100.0
    );

    // ---------------------------------------------------------------
    // 5. Arbitrage diagnostics
    // ---------------------------------------------------------------

    println!("\n--- Arbitrage diagnostics ---\n");

    let diag = surface.diagnostics()?;
    println!("Butterfly violations per tenor:");
    for (i, report) in diag.smile_reports.iter().enumerate() {
        println!(
            "  Tenor {}: {} violations, arb-free={}",
            i + 1,
            report.butterfly_violations.len(),
            report.is_free
        );
    }
    println!("Calendar violations: {}", diag.calendar_violations.len());
    println!("Surface arb-free: {}", diag.is_free);

    // Analytical calendar check (SSVI-specific)
    let cal_violations = surface.calendar_arb_analytical();
    println!("Analytical calendar violations: {}", cal_violations.len());

    // ---------------------------------------------------------------
    // 6. Calibrate SSVI from market data
    // ---------------------------------------------------------------

    println!("\n--- Calibration ---\n");

    // Generate synthetic market data from the known surface
    let cal_strikes: Vec<f64> = (70..=130).step_by(5).map(|k| k as f64).collect();

    let market_data: Vec<Vec<(f64, f64)>> = tenors
        .iter()
        .map(|&t| {
            cal_strikes
                .iter()
                .map(|&k| Ok((k, surface.black_vol(t, k)?.0)))
                .collect::<Result<Vec<_>, volsurf::VolSurfError>>()
        })
        .collect::<Result<_, _>>()?;

    let calibrated = SsviSurface::calibrate(&market_data, &tenors, &forwards)?;

    println!(
        "Original:   rho={:.4}, eta={:.4}, gamma={:.4}",
        surface.rho(), surface.eta(), surface.gamma()
    );
    println!(
        "Calibrated: rho={:.4}, eta={:.4}, gamma={:.4}",
        calibrated.rho(), calibrated.eta(), calibrated.gamma()
    );

    // Compare at a few points
    let mut max_err: f64 = 0.0;
    for &t in &tenors {
        for &k in &cal_strikes {
            let orig = surface.black_vol(t, k)?.0;
            let cal = calibrated.black_vol(t, k)?.0;
            max_err = max_err.max((orig - cal).abs());
        }
    }
    println!("Max vol error: {max_err:.2e}");

    // ---------------------------------------------------------------
    // 7. Compare with PiecewiseSurface (SVI per-tenor)
    // ---------------------------------------------------------------

    println!("\n--- SSVI vs PiecewiseSurface (SVI) ---\n");

    let strikes_7 = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

    let mut builder = SurfaceBuilder::new().spot(100.0).rate(0.0);

    for &t in &tenors {
        let vols: Vec<f64> = strikes_7
            .iter()
            .map(|&k| surface.black_vol(t, k).unwrap().0)
            .collect();
        builder = builder.add_tenor(t, &strikes_7, &vols);
    }

    let piecewise = builder.build()?;

    println!("{:>8} {:>10} {:>10} {:>10}", "T,K", "SSVI", "Piecewise", "Diff(bp)");
    println!("{}", "-".repeat(42));

    for &t in &[0.25, 1.0, 2.0] {
        for &k in &[90.0, 100.0, 110.0] {
            let ssvi_vol = surface.black_vol(t, k)?.0;
            let pw_vol = piecewise.black_vol(t, k)?.0;
            let diff_bp = (ssvi_vol - pw_vol) * 10_000.0;
            println!(
                "{t:.2},{k:.0} {:>9.4}% {:>9.4}% {:>9.1}",
                ssvi_vol * 100.0,
                pw_vol * 100.0,
                diff_bp
            );
        }
    }

    Ok(())
}
