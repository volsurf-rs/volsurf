//! SABR smile model: construction, calibration, and analysis.
//!
//! Demonstrates the SABR stochastic volatility model (Hagan et al., 2002):
//!   1. Construct a SABR smile from known parameters
//!   2. Query implied vol across strikes — see the characteristic skew
//!   3. Check butterfly arbitrage quality
//!   4. Calibrate SABR from market data
//!   5. Build a multi-tenor SABR surface via SurfaceBuilder
//!
//! Run with: `cargo run --example sabr_smile`

use volsurf::smile::{SabrSmile, SmileSection, SviSmile};
use volsurf::surface::{SmileModel, SurfaceBuilder, VolSurface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------
    // 1. Construct a SABR smile with equity parameters
    // ---------------------------------------------------------------

    let forward = 100.0;
    let expiry = 1.0;

    // Typical equity parameters: beta=0.5 (CIR-like), negative skew
    let smile = SabrSmile::new(
        forward, expiry, 0.3,  // alpha: ATM vol scale
        1.0,  // beta: lognormal backbone (simplest case)
        -0.3, // rho: negative skew (equity)
        0.4,  // nu: vol-of-vol (smile curvature)
    )?;

    println!(
        "SABR smile: alpha={:.2}, beta={:.1}, rho={:.2}, nu={:.2}",
        smile.alpha(),
        smile.beta(),
        smile.rho(),
        smile.nu()
    );
    println!("Forward={forward}, Expiry={expiry}y\n");

    // ---------------------------------------------------------------
    // 2. Query vol across strikes
    // ---------------------------------------------------------------

    println!("{:>8} {:>10} {:>12}", "Strike", "Vol", "Density");
    println!("{}", "-".repeat(32));

    let strikes = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];
    for &k in &strikes {
        let vol = smile.vol(k)?;
        let dens = smile.density(k)?;
        println!("{k:>8.0} {:>9.4}% {dens:>12.6}", vol.0 * 100.0);
    }

    // ---------------------------------------------------------------
    // 3. Butterfly arbitrage check
    // ---------------------------------------------------------------

    println!("\n--- Arbitrage check ---\n");
    let report = smile.is_arbitrage_free()?;
    println!(
        "Butterfly violations: {}, arb-free: {}",
        report.butterfly_violations.len(),
        report.is_free
    );

    // ---------------------------------------------------------------
    // 4. Calibrate SABR from market data
    // ---------------------------------------------------------------

    println!("\n--- Calibration round-trip ---\n");

    // Generate synthetic market data from the known smile
    let market_vols: Vec<(f64, f64)> = strikes
        .iter()
        .map(|&k| Ok((k, smile.vol(k)?.0)))
        .collect::<Result<_, volsurf::VolSurfError>>()?;

    // Calibrate with beta=1.0 (same as original)
    let calibrated = SabrSmile::calibrate(forward, expiry, 1.0, &market_vols)?;

    println!(
        "Original:   alpha={:.6}, rho={:.6}, nu={:.6}",
        smile.alpha(),
        smile.rho(),
        smile.nu()
    );
    println!(
        "Calibrated: alpha={:.6}, rho={:.6}, nu={:.6}",
        calibrated.alpha(),
        calibrated.rho(),
        calibrated.nu()
    );

    // Compare vols
    let mut max_err: f64 = 0.0;
    for &(k, orig_vol) in &market_vols {
        let cal_vol = calibrated.vol(k)?.0;
        max_err = max_err.max((orig_vol - cal_vol).abs());
    }
    println!("Max vol error: {max_err:.2e}");

    // ---------------------------------------------------------------
    // 5. Compare SABR with SVI on the same data
    // ---------------------------------------------------------------

    println!("\n--- SABR vs SVI comparison ---\n");

    let svi = SviSmile::calibrate(forward, expiry, &market_vols)?;

    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "Strike", "SABR", "SVI", "Diff(bp)"
    );
    println!("{}", "-".repeat(42));

    for &(k, _) in &market_vols {
        let sabr_vol = smile.vol(k)?.0;
        let svi_vol = svi.vol(k)?.0;
        let diff_bp = (sabr_vol - svi_vol) * 10_000.0;
        println!(
            "{k:>8.0} {:>9.4}% {:>9.4}% {:>9.1}",
            sabr_vol * 100.0,
            svi_vol * 100.0,
            diff_bp
        );
    }

    // ---------------------------------------------------------------
    // 6. Build a multi-tenor SABR surface
    // ---------------------------------------------------------------

    println!("\n--- Multi-tenor SABR surface ---\n");

    let spot = 100.0;
    let rate = 0.05;
    let strikes_wide = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

    // Generate realistic smiles at three tenors
    let vols_3m: Vec<f64> = strikes_wide
        .iter()
        .map(|&k| {
            SabrSmile::new(forward, 0.25, 0.3, 1.0, -0.3, 0.4)
                .unwrap()
                .vol(k)
                .unwrap()
                .0
        })
        .collect();

    let vols_1y: Vec<f64> = strikes_wide
        .iter()
        .map(|&k| {
            SabrSmile::new(forward, 1.0, 0.25, 1.0, -0.25, 0.35)
                .unwrap()
                .vol(k)
                .unwrap()
                .0
        })
        .collect();

    let vols_2y: Vec<f64> = strikes_wide
        .iter()
        .map(|&k| {
            SabrSmile::new(forward, 2.0, 0.22, 1.0, -0.20, 0.30)
                .unwrap()
                .vol(k)
                .unwrap()
                .0
        })
        .collect();

    let surface = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .model(SmileModel::Sabr { beta: 1.0 })
        .add_tenor(0.25, &strikes_wide, &vols_3m)
        .add_tenor(1.00, &strikes_wide, &vols_1y)
        .add_tenor(2.00, &strikes_wide, &vols_2y)
        .build()?;

    println!("Surface built: 3 tenors, SABR model (beta=1.0)\n");

    // Query across the surface
    print!("{:>8}", "T\\K");
    for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
        print!("{k:>10.0}");
    }
    println!();
    println!("{}", "-".repeat(58));

    for &t in &[0.25, 0.50, 1.00, 1.50, 2.00] {
        print!("{t:>8.2}");
        for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let vol = surface.black_vol(t, k)?;
            print!("{:>10.2}%", vol.0 * 100.0);
        }
        println!();
    }

    // Surface diagnostics
    let diag = surface.diagnostics()?;
    println!("\nSurface arb-free: {}", diag.is_free);
    println!("Calendar violations: {}", diag.calendar_violations.len());

    Ok(())
}
