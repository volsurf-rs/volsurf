//! Extended SSVI surface: construction, calibration, and comparison with SSVI.
//!
//! eSSVI (Hendriks-Martini 2019) extends SSVI by making the skew parameter
//! ρ depend on maturity: ρ(θ) = ρ₀ + (ρₘ − ρ₀)·(θ/θ_max)^a. This matches
//! the empirical observation that short-dated equity smiles have steeper skew
//! than long-dated ones.
//!
//! Shows how to:
//!   1. Construct an eSSVI surface with known parameters
//!   2. See ρ(θ) vary across tenors — the key eSSVI innovation
//!   3. Query vol across a (T, K) grid
//!   4. Calibrate via the two-stage API: fit_per_tenor → from_per_tenor
//!   5. Use the one-shot calibrate() convenience wrapper
//!   6. Run structural calendar-arbitrage checks (Thm 4.1, Eq 4.10)
//!   7. Compare eSSVI vs SSVI — where maturity-dependent ρ matters
//!
//! Run with: `cargo run --example essvi_surface`

use volsurf::surface::{EssviSurface, SsviSurface, VolSurface};
use volsurf::{Strike, Tenor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------
    // 1. Construct from known parameters
    // ---------------------------------------------------------------

    let tenors = vec![0.25, 0.50, 1.0, 2.0];
    let forwards = vec![100.0; 4];
    let thetas = vec![0.01, 0.02, 0.04, 0.08]; // ~20% ATM vol

    let surface = EssviSurface::new(
        -0.7, // rho_0: steep skew at short maturities
        -0.3, // rho_m: flatter skew at long maturities
        0.5,  // a: shape exponent for rho(theta)
        0.5,  // eta: smile amplitude
        0.5,  // gamma: term structure decay
        tenors.clone(),
        forwards.clone(),
        thetas.clone(),
    )?;

    println!(
        "eSSVI surface: rho_0={:.1}, rho_m={:.1}, a={:.1}, eta={:.1}, gamma={:.1}",
        surface.rho_0(),
        surface.rho_m(),
        surface.a(),
        surface.eta(),
        surface.gamma()
    );

    // ---------------------------------------------------------------
    // 2. rho(theta) across tenors
    // ---------------------------------------------------------------

    println!("\n--- rho(theta) across tenors ---\n");
    println!("{:>8} {:>8} {:>8}", "T", "theta", "rho");
    println!("{}", "-".repeat(28));
    for (&t, &theta) in tenors.iter().zip(thetas.iter()) {
        println!("{t:>8.2} {theta:>8.4} {:>8.4}", surface.rho(theta));
    }

    // ---------------------------------------------------------------
    // 3. Vol grid
    // ---------------------------------------------------------------

    println!("\n--- Vol grid (expiry x strike) ---\n");

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
            let vol = surface.black_vol(Tenor(t), Strike(k))?;
            print!("{:>8.2}%", vol.0 * 100.0);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // 4. Two-stage calibration
    // ---------------------------------------------------------------

    println!("\n--- Two-stage calibration ---\n");

    // Generate synthetic market data from the known surface
    let cal_strikes: Vec<f64> = (70..=130).step_by(5).map(|k| k as f64).collect();

    let market_data: Vec<Vec<(f64, f64)>> = tenors
        .iter()
        .map(|&t| {
            cal_strikes
                .iter()
                .map(|&k| Ok((k, surface.black_vol(Tenor(t), Strike(k))?.0)))
                .collect::<Result<Vec<_>, volsurf::VolSurfError>>()
        })
        .collect::<Result<_, _>>()?;

    // Stage 1: fit SVI per tenor
    let fits = EssviSurface::fit_per_tenor(&market_data, &tenors, &forwards)?;

    println!("Per-tenor SVI fits:");
    for fit in &fits {
        println!(
            "  T={:.2}: theta={:.6}, rms={:.2e}",
            fit.tenor, fit.theta, fit.rms_error
        );
    }

    // Stage 2: fit global rho(theta) + eta/gamma
    let calibrated = EssviSurface::from_per_tenor(&fits)?;

    println!(
        "\nOriginal:   rho_0={:.4}, rho_m={:.4}, a={:.4}, eta={:.4}, gamma={:.4}",
        surface.rho_0(),
        surface.rho_m(),
        surface.a(),
        surface.eta(),
        surface.gamma()
    );
    println!(
        "Calibrated: rho_0={:.4}, rho_m={:.4}, a={:.4}, eta={:.4}, gamma={:.4}",
        calibrated.rho_0(),
        calibrated.rho_m(),
        calibrated.a(),
        calibrated.eta(),
        calibrated.gamma()
    );

    let mut max_err: f64 = 0.0;
    for &t in &tenors {
        for &k in &cal_strikes {
            let orig = surface.black_vol(Tenor(t), Strike(k))?.0;
            let cal = calibrated.black_vol(Tenor(t), Strike(k))?.0;
            max_err = max_err.max((orig - cal).abs());
        }
    }
    println!("Max vol error: {max_err:.2e}");

    // ---------------------------------------------------------------
    // 5. One-shot calibrate (same result, less control)
    // ---------------------------------------------------------------

    println!("\n--- One-shot calibrate ---\n");

    let oneshot = EssviSurface::calibrate(&market_data, &tenors, &forwards)?;
    println!(
        "One-shot: rho_0={:.4}, rho_m={:.4}, a={:.4}, eta={:.4}, gamma={:.4}",
        oneshot.rho_0(),
        oneshot.rho_m(),
        oneshot.a(),
        oneshot.eta(),
        oneshot.gamma()
    );

    // ---------------------------------------------------------------
    // 6. Structural calendar-arb check (Thm 4.1, Eq 4.10)
    // ---------------------------------------------------------------

    println!("\n--- Calendar arbitrage (structural) ---\n");

    let violations = surface.calendar_check_structural();
    println!("Structural violations: {}", violations.len());

    let diag = surface.diagnostics()?;
    println!(
        "Numerical calendar violations: {}",
        diag.calendar_violations.len()
    );
    println!("Surface arb-free: {}", diag.is_free);

    // ---------------------------------------------------------------
    // 7. Compare eSSVI vs SSVI (fixed rho = rho_m)
    // ---------------------------------------------------------------

    println!(
        "\n--- eSSVI vs SSVI (rho fixed at rho_m={:.1}) ---\n",
        surface.rho_m()
    );

    let ssvi = SsviSurface::new(
        surface.rho_m(), // fixed rho
        surface.eta(),
        surface.gamma(),
        tenors.clone(),
        forwards.clone(),
        thetas.clone(),
    )?;

    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10}",
        "T", "K", "eSSVI", "SSVI", "Diff(bp)"
    );
    println!("{}", "-".repeat(46));

    for &t in &[0.25, 0.50, 1.0, 2.0] {
        for &k in &[90.0, 100.0, 110.0] {
            let ev = surface.black_vol(Tenor(t), Strike(k))?.0;
            let sv = ssvi.black_vol(Tenor(t), Strike(k))?.0;
            let diff_bp = (ev - sv) * 10_000.0;
            println!(
                "{t:>6.2} {k:>6.0} {:>9.4}% {:>9.4}% {:>9.1}",
                ev * 100.0,
                sv * 100.0,
                diff_bp
            );
        }
    }

    println!("\neSSVI diverges from SSVI most at short tenors where rho(theta)");
    println!(
        "is steeper ({:.2} vs {:.2}). They converge at T=2Y where rho -> rho_m.",
        surface.rho(*thetas.first().unwrap()),
        surface.rho_m()
    );

    Ok(())
}
