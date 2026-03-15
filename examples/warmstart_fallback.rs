//! Warm-start fallback demo: bad seed recovery (#114).
//!
//! Simulates the real-world scenario where a previous day's SVI calibration
//! becomes a bad seed for the next day due to large DTE change. Before the
//! fix, this would fail; now it falls back to grid search.
//!
//! Run with: `cargo run --example warmstart_fallback`

use std::time::Instant;

use volsurf::calibration::{DataFilter, WeightingScheme};
use volsurf::smile::{SmileSection, SviSmile};
use volsurf::types::Strike;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Day N-1: 30 DTE, calibrate successfully
    let fwd_prev = 5200.0;
    let t_prev = 30.0 / 365.0;
    let market_prev: Vec<(f64, f64)> = vec![
        (5000.0, 0.22),
        (5050.0, 0.21),
        (5100.0, 0.195),
        (5150.0, 0.185),
        (5200.0, 0.18),
        (5250.0, 0.185),
        (5300.0, 0.195),
        (5350.0, 0.21),
        (5400.0, 0.23),
    ];

    let t0 = Instant::now();
    let smile_prev = SviSmile::calibrate(fwd_prev, t_prev, &market_prev)?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Day N-1 (DTE=30, cold start): {cold_ms:.1}ms");
    println!("  ATM vol: {:.4}", smile_prev.vol(Strike(5200.0))?.0);

    // Day N: 10 DTE (weekend jump, T dropped 67%)
    let fwd = 5180.0;
    let t = 10.0 / 365.0;
    let market: Vec<(f64, f64)> = vec![
        (5020.0, 0.28),
        (5060.0, 0.25),
        (5100.0, 0.22),
        (5140.0, 0.195),
        (5180.0, 0.185),
        (5220.0, 0.19),
        (5260.0, 0.205),
        (5300.0, 0.225),
        (5340.0, 0.26),
    ];

    // Warm-start with previous day's smile — seed's (m, sigma) may be stale
    let t1 = Instant::now();
    let smile = SviSmile::calibrate_with_config(
        fwd,
        t,
        &market,
        &DataFilter::default(),
        &WeightingScheme::default(),
        Some(&smile_prev),
    )?;
    let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("\nDay N (DTE=10, warm-start): {warm_ms:.1}ms");
    println!("  ATM vol: {:.4}", smile.vol(Strike(5180.0))?.0);

    // Cold-start for comparison
    let t2 = Instant::now();
    let smile_cold = SviSmile::calibrate(fwd, t, &market)?;
    let cold2_ms = t2.elapsed().as_secs_f64() * 1000.0;
    println!("\nDay N (DTE=10, cold start): {cold2_ms:.1}ms");
    println!("  ATM vol: {:.4}", smile_cold.vol(Strike(5180.0))?.0);

    let atm_diff = (smile.vol(Strike(5180.0))?.0 - smile_cold.vol(Strike(5180.0))?.0).abs();
    println!("\nATM vol diff (warm vs cold): {atm_diff:.6}");

    // Deliberately bad seed — m=500 is far outside any valid region
    println!("\n--- Extreme bad seed (m=500) ---");
    let bad_seed = SviSmile::new(5200.0, 1.0, 0.01, 0.3, -0.3, 500.0, 100.0)?;
    let t3 = Instant::now();
    let smile_bad = SviSmile::calibrate_with_config(
        fwd,
        t,
        &market,
        &DataFilter::default(),
        &WeightingScheme::default(),
        Some(&bad_seed),
    )?;
    let bad_ms = t3.elapsed().as_secs_f64() * 1000.0;
    println!("Recovered via grid search fallback: {bad_ms:.1}ms");
    println!("  ATM vol: {:.4}", smile_bad.vol(Strike(5180.0))?.0);

    Ok(())
}
