//! Configurable calibration: filtering, weighting, and warm-starting.
//!
//! Shows how to:
//!   - Filter noisy wing data with DataFilter
//!   - Choose vega vs uniform weighting
//!   - Warm-start from a previous calibration (skip grid search)
//!   - Pass config through SurfaceBuilder
//!
//! Run with: `cargo run --example calibration_config`

use std::time::Instant;

use volsurf::calibration::{DataFilter, WeightingScheme};
use volsurf::smile::{SabrSmile, SmileSection, SviSmile};
use volsurf::surface::{SmileModel, SurfaceBuilder, VolSurface};
use volsurf::types::{Strike, Tenor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let expiry = 0.25;

    // Synthetic market data with noisy far-wing points
    let mut market: Vec<(f64, f64)> = vec![
        (85.0, 0.28),
        (90.0, 0.24),
        (95.0, 0.22),
        (100.0, 0.20),
        (105.0, 0.21),
        (110.0, 0.23),
        (115.0, 0.27),
    ];
    // Add garbage wing quotes (cabinet-level OTM)
    market.push((50.0, 0.05));
    market.push((200.0, 0.03));

    println!(
        "Market data: {} points (7 clean + 2 garbage wings)\n",
        market.len()
    );

    // ---------------------------------------------------------------
    // 1. DataFilter — exclude far wings by moneyness
    // ---------------------------------------------------------------

    let filter = DataFilter {
        max_log_moneyness: Some(0.3), // |ln(K/F)| <= 0.3
        min_vol: Some(0.05),          // exclude < 5% vol
        ..Default::default()
    };

    let filtered = SviSmile::calibrate_with_config(
        forward,
        expiry,
        &market,
        &filter,
        &WeightingScheme::default(),
        None,
    )?;
    let unfiltered = SviSmile::calibrate(forward, expiry, &market)?;

    println!("--- DataFilter effect ---");
    println!(
        "ATM vol (filtered):   {:.4}%",
        filtered.vol(Strike(100.0))?.0 * 100.0
    );
    println!(
        "ATM vol (unfiltered): {:.4}%",
        unfiltered.vol(Strike(100.0))?.0 * 100.0
    );

    // ---------------------------------------------------------------
    // 2. WeightingScheme — vega vs uniform
    // ---------------------------------------------------------------

    let clean: Vec<(f64, f64)> = market[..7].to_vec();

    let vega = SviSmile::calibrate_with_config(
        forward,
        expiry,
        &clean,
        &DataFilter::default(),
        &WeightingScheme::Vega,
        None,
    )?;
    let uniform = SviSmile::calibrate_with_config(
        forward,
        expiry,
        &clean,
        &DataFilter::default(),
        &WeightingScheme::Uniform,
        None,
    )?;

    println!("\n--- WeightingScheme effect ---");
    println!("{:>8} {:>12} {:>12}", "Strike", "Vega", "Uniform");
    println!("{}", "-".repeat(35));
    for k in [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0] {
        println!(
            "{k:>8.0} {v:>10.4}% {u:>10.4}%",
            v = vega.vol(Strike(k))?.0 * 100.0,
            u = uniform.vol(Strike(k))?.0 * 100.0,
        );
    }

    // ---------------------------------------------------------------
    // 3. Warm-starting — seed from previous calibration
    // ---------------------------------------------------------------

    let t_cold = Instant::now();
    let cold = SviSmile::calibrate(forward, expiry, &clean)?;
    let cold_ms = t_cold.elapsed();

    let t_warm = Instant::now();
    let warm = SviSmile::calibrate_with_config(
        forward,
        expiry,
        &clean,
        &DataFilter::default(),
        &WeightingScheme::default(),
        Some(&cold),
    )?;
    let warm_ms = t_warm.elapsed();

    println!("\n--- Warm-start speedup ---");
    println!("Cold start: {:?}", cold_ms);
    println!("Warm start: {:?}", warm_ms);
    println!(
        "ATM match:  cold={:.6}, warm={:.6}",
        cold.vol(Strike(100.0))?.0,
        warm.vol(Strike(100.0))?.0,
    );

    // ---------------------------------------------------------------
    // 4. SurfaceBuilder with config
    // ---------------------------------------------------------------

    let strikes: Vec<f64> = clean.iter().map(|&(k, _)| k).collect();
    let vols: Vec<f64> = clean.iter().map(|&(_, v)| v).collect();

    let surface = SurfaceBuilder::new()
        .spot(100.0)
        .rate(0.05)
        .model(SmileModel::Svi)
        .data_filter(DataFilter {
            max_log_moneyness: Some(0.5),
            ..Default::default()
        })
        .weighting(WeightingScheme::Vega)
        .add_tenor(0.25, &strikes, &vols)
        .add_tenor(1.0, &strikes, &vols)
        .build()?;

    println!("\n--- SurfaceBuilder with config ---");
    println!(
        "3m ATM vol: {:.4}%",
        surface.black_vol(Tenor(0.25), Strike(100.0))?.0 * 100.0
    );
    println!(
        "1y ATM vol: {:.4}%",
        surface.black_vol(Tenor(1.0), Strike(100.0))?.0 * 100.0
    );

    // ---------------------------------------------------------------
    // 5. SABR with warm-start
    // ---------------------------------------------------------------

    let sabr_data: Vec<(f64, f64)> = vec![
        (90.0, 0.24),
        (95.0, 0.22),
        (100.0, 0.20),
        (105.0, 0.21),
        (110.0, 0.23),
    ];

    let sabr1 = SabrSmile::calibrate(forward, expiry, 0.5, &sabr_data)?;
    let sabr2 = SabrSmile::calibrate_with_config(
        forward,
        expiry,
        0.5,
        &sabr_data,
        &DataFilter::default(),
        &WeightingScheme::default(),
        Some(&sabr1),
    )?;

    println!("\n--- SABR warm-start ---");
    println!("Cold ATM: {:.4}%", sabr1.vol(Strike(100.0))?.0 * 100.0);
    println!("Warm ATM: {:.4}%", sabr2.vol(Strike(100.0))?.0 * 100.0);

    Ok(())
}
