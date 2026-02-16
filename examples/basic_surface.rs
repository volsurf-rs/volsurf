//! Build a volatility surface from market quotes and query it.
//!
//! Demonstrates the core v0.1 workflow:
//!   1. Define per-tenor (strike, vol) market data
//!   2. Construct a surface via SurfaceBuilder
//!   3. Query vol/variance at arbitrary (expiry, strike) points
//!   4. Extract a smile section for a specific tenor
//!   5. Run arbitrage diagnostics
//!
//! Run with: `cargo run --example basic_surface`

use volsurf::surface::{SurfaceBuilder, VolSurface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------
    // 1. Market data: SPX-like equity smile across three tenors
    // ---------------------------------------------------------------

    // Strikes are absolute (not delta or moneyness)
    let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

    // 3-month smile: pronounced skew, higher ATM
    let vols_3m = vec![0.30, 0.26, 0.24, 0.22, 0.23, 0.25, 0.29];

    // 6-month smile: moderate skew, slightly lower ATM
    let vols_6m = vec![0.28, 0.25, 0.23, 0.21, 0.22, 0.24, 0.27];

    // 1-year smile: flatter, lower ATM (term structure decay)
    let vols_1y = vec![0.26, 0.23, 0.22, 0.20, 0.21, 0.23, 0.26];

    // ---------------------------------------------------------------
    // 2. Build the surface
    // ---------------------------------------------------------------

    let surface = SurfaceBuilder::new()
        .spot(100.0)
        .rate(0.05) // risk-free rate
        .add_tenor(0.25, &strikes, &vols_3m) // 3M
        .add_tenor(0.50, &strikes, &vols_6m) // 6M
        .add_tenor(1.00, &strikes, &vols_1y) // 1Y
        .build()?;

    println!("Surface built: 3 tenors x 7 strikes\n");

    // ---------------------------------------------------------------
    // 3. Query vol and variance across the surface
    // ---------------------------------------------------------------

    println!("--- Vol grid (expiry x strike) ---\n");
    let query_tenors = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50];
    let query_strikes = [80.0, 90.0, 100.0, 110.0, 120.0];

    // Header
    print!("{:>8}", "T\\K");
    for &k in &query_strikes {
        print!("{k:>10.0}");
    }
    println!();
    println!("{}", "-".repeat(58));

    for &t in &query_tenors {
        print!("{t:>8.2}");
        for &k in &query_strikes {
            let vol = surface.black_vol(t, k)?;
            print!("{:>10.2}%", vol.0 * 100.0);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // 4. Vol vs variance consistency
    // ---------------------------------------------------------------

    println!("\n--- Vol/Variance consistency check ---\n");
    let t = 0.5;
    let k = 100.0;
    let vol = surface.black_vol(t, k)?;
    let var = surface.black_variance(t, k)?;
    println!("At T={t}, K={k}:");
    println!("  vol      = {:.6}", vol.0);
    println!("  variance = {:.6}", var.0);
    println!("  vol^2*T  = {:.6}", vol.0 * vol.0 * t);
    println!("  match    = {}", (vol.0 * vol.0 * t - var.0).abs() < 1e-12);

    // ---------------------------------------------------------------
    // 5. Extract a smile section at an interpolated tenor
    // ---------------------------------------------------------------

    println!("\n--- Smile section at T=0.75 (interpolated) ---\n");
    let smile = surface.smile_at(0.75)?;
    println!("Forward: {:.2}", smile.forward());
    println!("Expiry:  {:.2}", smile.expiry());
    for &k in &query_strikes {
        let v = smile.vol(k)?;
        let d = smile.density(k)?;
        println!("  K={k:>6.0}  vol={:.4}  density={:.6}", v.0, d);
    }

    // ---------------------------------------------------------------
    // 6. Arbitrage diagnostics
    // ---------------------------------------------------------------

    println!("\n--- Arbitrage diagnostics ---\n");
    let diag = surface.diagnostics()?;
    println!("Butterfly violations per tenor:");
    for (i, report) in diag.smile_reports.iter().enumerate() {
        println!(
            "  Tenor {}: {} violations",
            i + 1,
            report.butterfly_violations.len()
        );
    }
    println!(
        "Calendar violations: {}",
        diag.calendar_violations.len()
    );
    println!(
        "Surface is arbitrage-free: {}",
        diag.is_free
    );

    Ok(())
}
