//! Compare SVI and CubicSpline smile models side by side.
//!
//! Shows how to:
//!   - Calibrate an SVI smile from market data
//!   - Build a SplineSmile from variance knots
//!   - Compare vol output and arbitrage quality
//!
//! Run with: `cargo run --example smile_models`

use volsurf::smile::{SmileSection, SplineSmile, SviSmile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 1.0;

    // Market data: (strike, implied vol)
    let market_data: Vec<(f64, f64)> = vec![
        (70.0, 0.32),
        (80.0, 0.28),
        (90.0, 0.24),
        (95.0, 0.22),
        (100.0, 0.20),
        (105.0, 0.21),
        (110.0, 0.23),
        (120.0, 0.27),
        (130.0, 0.31),
    ];

    // ---------------------------------------------------------------
    // 1. SVI calibration
    // ---------------------------------------------------------------

    let svi = SviSmile::calibrate(forward, expiry, &market_data)?;
    println!("SVI calibrated from {} market points\n", market_data.len());

    // ---------------------------------------------------------------
    // 2. CubicSpline construction
    // ---------------------------------------------------------------

    let (strikes, variances): (Vec<f64>, Vec<f64>) = market_data
        .iter()
        .map(|&(k, v)| (k, v * v * expiry)) // convert vol → total variance
        .unzip();
    let spline = SplineSmile::new(forward, expiry, strikes, variances)?;
    println!("SplineSmile built from {} knot points\n", market_data.len());

    // ---------------------------------------------------------------
    // 3. Compare outputs
    // ---------------------------------------------------------------

    println!("{:>8} {:>12} {:>12} {:>12}", "Strike", "Market", "SVI", "Spline");
    println!("{}", "-".repeat(48));

    for &(k, market_vol) in &market_data {
        let svi_vol = svi.vol(k)?.0;
        let spline_vol = spline.vol(k)?.0;
        println!(
            "{k:>8.0} {market_vol:>11.4}% {svi_vol:>11.4}% {spline_vol:>11.4}%",
        );
    }

    // Also query at off-grid strikes
    println!("\n{:>8} {:>12} {:>12}", "Strike", "SVI", "Spline");
    println!("{}", "-".repeat(35));
    for k in [75.0, 85.0, 97.5, 102.5, 115.0, 125.0] {
        let svi_vol = svi.vol(k)?.0;
        let spline_vol = spline.vol(k)?.0;
        println!("{k:>8.1} {svi_vol:>11.4}% {spline_vol:>11.4}%");
    }

    // ---------------------------------------------------------------
    // 4. Arbitrage quality
    // ---------------------------------------------------------------

    println!("\n--- Arbitrage check ---\n");
    let svi_report = svi.is_arbitrage_free()?;
    let spline_report = spline.is_arbitrage_free()?;
    println!(
        "SVI:    {} butterfly violations, arb-free={}",
        svi_report.butterfly_violations.len(),
        svi_report.is_free
    );
    println!(
        "Spline: {} butterfly violations, arb-free={}",
        spline_report.butterfly_violations.len(),
        spline_report.is_free
    );

    // ---------------------------------------------------------------
    // 5. Density comparison at ATM
    // ---------------------------------------------------------------

    println!("\n--- Risk-neutral density at ATM ---\n");
    let svi_d = svi.density(100.0)?;
    let spline_d = spline.density(100.0)?;
    println!("SVI density:    {svi_d:.6}");
    println!("Spline density: {spline_d:.6}");

    Ok(())
}
