//! Dupire local volatility from an SSVI surface.
//!
//! Local vol is the key output of a calibrated surface — it's what you
//! feed into a forward PDE or Monte Carlo to price path-dependent exotics.
//! The Dupire formula (Gatheral 2006, Eq 1.10) extracts σ_loc(T, K) from
//! the total implied variance surface using finite differences.
//!
//! Shows how to:
//!   1. Build an SSVI surface and wrap it for Dupire
//!   2. Query local vol at a (T, K) grid
//!   3. Compare local vol vs implied vol — local vol amplifies the skew
//!   4. Customize the finite-difference step size
//!
//! Run with: `cargo run --example local_vol`

use std::sync::Arc;
use volsurf::local_vol::{DupireLocalVol, LocalVol};
use volsurf::surface::{SsviSurface, VolSurface};
use volsurf::{Strike, Tenor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------
    // 1. Build an SSVI surface (same params as ssvi_surface example)
    // ---------------------------------------------------------------

    let surface = SsviSurface::new(
        -0.3, // rho: equity skew
        0.5,  // eta: smile amplitude
        0.5,  // gamma: term structure decay
        vec![0.25, 0.50, 1.0, 2.0],
        vec![100.0; 4],
        vec![0.01, 0.02, 0.04, 0.08], // ~20% ATM vol
    )?;

    // ---------------------------------------------------------------
    // 2. Wrap in Arc and create DupireLocalVol
    // ---------------------------------------------------------------

    let arc_surface: Arc<dyn VolSurface> = Arc::new(surface);
    let dupire = DupireLocalVol::new(Arc::clone(&arc_surface));

    println!("Dupire local vol from SSVI surface (default bump_size=1%)\n");

    // ---------------------------------------------------------------
    // 3. Local vol grid
    // ---------------------------------------------------------------

    let tenors = [0.25, 0.50, 1.0, 2.0];
    let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];

    println!("--- Local volatility grid ---\n");
    print!("{:>8}", "T\\K");
    for &k in &strikes {
        print!("{k:>10.0}");
    }
    println!();
    println!("{}", "-".repeat(78));

    for &t in &tenors {
        print!("{t:>8.2}");
        for &k in &strikes {
            let lv = dupire.local_vol(Tenor(t), Strike(k))?;
            print!("{:>10.2}%", lv.0 * 100.0);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // 4. Local vol vs implied vol at a single tenor
    // ---------------------------------------------------------------

    println!("\n--- Local vol vs Implied vol at T=1.0 ---\n");
    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "Strike", "Implied", "Local", "Ratio"
    );
    println!("{}", "-".repeat(42));

    let t = 1.0;
    for &k in &strikes {
        let iv = arc_surface.black_vol(Tenor(t), Strike(k))?.0;
        let lv = dupire.local_vol(Tenor(t), Strike(k))?.0;
        println!(
            "{k:>8.0} {:>9.4}% {:>9.4}% {:>9.2}",
            iv * 100.0,
            lv * 100.0,
            lv / iv
        );
    }

    println!("\nLocal vol amplifies the skew: it's higher than implied on the");
    println!("put wing and varies more sharply across strikes.");

    // ---------------------------------------------------------------
    // 5. Interpolated tenor (not in the SSVI grid)
    // ---------------------------------------------------------------

    println!("\n--- Local vol at T=0.75 (interpolated) ---\n");
    for &k in &[90.0, 100.0, 110.0] {
        let lv = dupire.local_vol(Tenor(0.75), Strike(k))?;
        let iv = arc_surface.black_vol(Tenor(0.75), Strike(k))?.0;
        println!(
            "K={k:.0}: local={:.4}%, implied={:.4}%",
            lv.0 * 100.0,
            iv * 100.0
        );
    }

    // ---------------------------------------------------------------
    // 6. Custom bump size
    // ---------------------------------------------------------------

    println!("\n--- Bump size comparison at (T=1.0, K=100) ---\n");

    let bumps = [0.02, 0.01, 0.005, 0.001];
    println!("{:>10} {:>12}", "Bump", "Local vol");
    println!("{}", "-".repeat(24));

    for &h in &bumps {
        let d = DupireLocalVol::new(Arc::clone(&arc_surface)).with_bump_size(h)?;
        let lv = d.local_vol(Tenor(1.0), Strike(100.0))?;
        println!("{h:>10.3} {:>11.6}%", lv.0 * 100.0);
    }
    println!("\nSmaller bump → more accurate finite differences (O(h^2) convergence).");

    Ok(())
}
