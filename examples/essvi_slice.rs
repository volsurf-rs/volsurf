//! eSSVI slice demo: construction, vol queries, arb checks, serde.
//!
//! Run with: `cargo run --example essvi_slice`

use volsurf::SmileSection;
use volsurf::surface::{EssviSlice, SsviSlice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Typical equity parameters: 1Y tenor, 20% ATM vol → θ = 0.04
    let slice = EssviSlice::new(100.0, 1.0, -0.4, 0.6, 0.5, 0.04)?;

    println!("=== eSSVI Slice ===");
    println!("Forward: {:.0}  Expiry: {:.2}Y  θ: {:.4}", slice.forward(), slice.expiry(), slice.theta());
    println!("ρ(θ): {:.2}   η: {:.1}   γ: {:.1}", slice.rho(), slice.eta(), slice.gamma());
    println!();

    // Vol smile across strikes
    println!("{:<10} {:>10} {:>12}", "Strike", "Vol (%)", "Variance");
    println!("{:-<10} {:-<10} {:-<12}", "", "", "");
    for &k in &[80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0] {
        let vol = slice.vol(k)?;
        let var = slice.variance(k)?;
        println!("{:<10.0} {:>10.2} {:>12.6}", k, vol.0 * 100.0, var.0);
    }
    println!();

    // Equivalence with SsviSlice (same params → identical output)
    let ssvi = SsviSlice::new(100.0, 1.0, -0.4, 0.6, 0.5, 0.04)?;
    println!("=== Equivalence Check (eSSVI vs SSVI, same ρ) ===");
    let mut max_diff = 0.0_f64;
    for &k in &[70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0] {
        let v1 = slice.vol(k)?.0;
        let v2 = ssvi.vol(k)?.0;
        let diff = (v1 - v2).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Max vol difference across 7 strikes: {:.1e}", max_diff);
    println!();

    // Butterfly arbitrage check
    let report = slice.is_arbitrage_free()?;
    println!("=== Arbitrage Check ===");
    println!("Butterfly arb-free: {}", report.is_free);
    if !report.butterfly_violations.is_empty() {
        println!("Violations: {}", report.butterfly_violations.len());
    }
    println!();

    // Serde round-trip
    let json = serde_json::to_string_pretty(&slice)?;
    println!("=== Serde ===");
    println!("{json}");
    let restored: EssviSlice = serde_json::from_str(&json)?;
    println!("Round-trip ATM vol: {:.6} → {:.6}",
        slice.vol(100.0)?.0, restored.vol(100.0)?.0);
    println!();

    // Short-end slice with different ρ (the eSSVI motivation)
    println!("=== Short vs Long Tenor (different ρ) ===");
    let short = EssviSlice::new(100.0, 0.05, -0.7, 0.6, 0.5, 0.002)?;
    let long = EssviSlice::new(100.0, 2.0, -0.3, 0.6, 0.5, 0.08)?;
    println!("1M  (ρ=-0.70): ATM vol = {:.2}%", short.vol(100.0)?.0 * 100.0);
    println!("2Y  (ρ=-0.30): ATM vol = {:.2}%", long.vol(100.0)?.0 * 100.0);
    let skew_short = short.vol(95.0)?.0 - short.vol(105.0)?.0;
    let skew_long = long.vol(95.0)?.0 - long.vol(105.0)?.0;
    println!("1M  skew (95-105): {:.2} vol pts", skew_short * 100.0);
    println!("2Y  skew (95-105): {:.2} vol pts", skew_long * 100.0);

    Ok(())
}
