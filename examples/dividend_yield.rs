//! Dividend yield and per-tenor forward override demo.
//!
//! Uses real SPX option data from 2025-06-10 to compare three approaches
//! to forward price calculation when building a vol surface:
//!
//! 1. Rate only:          F = S · exp(r · T)
//! 2. With dividend yield: F = S · exp((r − q) · T)
//! 3. Per-tenor forwards:  F from put-call parity (market-implied)
//!
//! The industry standard (Gatheral 2006, Dupire 1994) treats forwards as
//! observable market inputs via put-call parity: F = K + C − P at the
//! near-ATM strike. The entire SVI/SSVI framework parameterizes in
//! log(K/F), making forward accuracy critical even though SABR/SVI
//! calibrations can absorb moderate forward errors by adjusting their
//! parameters. Wrong forwards propagate silently into delta hedges,
//! local vol extraction, and risk sensitivities.
//!
//! Run with: cargo run --example dividend_yield

use volsurf::surface::{SmileModel, SurfaceBuilder, VolSurface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SPX close 2025-06-10 (FRED), SOFR 2025-06-10 (NY Fed)
    let spot = 6038.81;
    let rate = 0.0428;

    // Implied dividend yield backed out from the forward curve.
    // Median across T > 0.3Y tenors is ~0.95%.
    let q = 0.0095;

    // Market-implied forwards from put-call parity (F = K + C - P at near-ATM strike).
    // Source: SPX options on OPRA, 2025-06-10, consolidated BBO.
    let forwards: &[(f64, f64)] = &[
        (0.1040, 6057.90),
        (0.3532, 6111.35),
        (1.0212, 6249.70),
        (1.5222, 6348.80),
        (2.5188, 6556.75),
    ];

    // Per-tenor (strike, implied vol) from SABR fit to market data.
    // 9 strikes spanning ~5700-6400 (±5% around ATM).
    let strikes: &[f64] = &[
        5700.0, 5800.0, 5900.0, 6000.0, 6050.0, 6100.0, 6200.0, 6300.0, 6400.0,
    ];
    let tenor_vols: &[(f64, &[f64])] = &[
        (
            0.1040,
            &[
                0.1923, 0.1766, 0.1612, 0.1464, 0.1396, 0.1332, 0.1230, 0.1176, 0.1172,
            ],
        ),
        (
            0.3532,
            &[
                0.1944, 0.1845, 0.1747, 0.1652, 0.1605, 0.1559, 0.1471, 0.1390, 0.1318,
            ],
        ),
        (
            1.0212,
            &[
                0.1918, 0.1853, 0.1790, 0.1728, 0.1697, 0.1667, 0.1609, 0.1553, 0.1499,
            ],
        ),
        (
            1.5222,
            &[
                0.1909, 0.1852, 0.1796, 0.1742, 0.1716, 0.1690, 0.1639, 0.1591, 0.1546,
            ],
        ),
        (
            2.5188,
            &[
                0.1919, 0.1873, 0.1829, 0.1785, 0.1764, 0.1743, 0.1703, 0.1663, 0.1626,
            ],
        ),
    ];

    // Approach 1: rate only — F = S · exp(r · T), ignores dividends
    let mut b1 = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .model(SmileModel::Sabr { beta: 0.5 });
    for &(t, vols) in tenor_vols {
        b1 = b1.add_tenor(t, strikes, vols);
    }
    let surface_rate_only = b1.build()?;

    // Approach 2: with dividend yield — F = S · exp((r - q) · T)
    let mut b2 = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .dividend_yield(q)
        .model(SmileModel::Sabr { beta: 0.5 });
    for &(t, vols) in tenor_vols {
        b2 = b2.add_tenor(t, strikes, vols);
    }
    let surface_with_q = b2.build()?;

    // Approach 3: per-tenor market-implied forwards from put-call parity
    let mut b3 = SurfaceBuilder::new()
        .spot(spot)
        .rate(rate)
        .model(SmileModel::Sabr { beta: 0.5 });
    for (&(t, vols), &(_, fwd)) in tenor_vols.iter().zip(forwards.iter()) {
        b3 = b3.add_tenor_with_forward(t, strikes, vols, fwd);
    }
    let surface_pcp = b3.build()?;

    println!("SPX Volatility Surface — Forward Price Comparison");
    println!(
        "Spot = {spot:.2}, r = {:.2}% (SOFR), q = {:.2}%",
        rate * 100.0,
        q * 100.0
    );
    println!();
    println!(
        "{:>6}  {:>10}  {:>12}  {:>12}  {:>12}",
        "T", "F(market)", "rate only", "rate-div", "put-call"
    );
    println!("{}", "-".repeat(58));

    for &(t, fwd_mkt) in forwards {
        let f1 = surface_rate_only.smile_at(t)?.forward();
        let f2 = surface_with_q.smile_at(t)?.forward();
        let f3 = surface_pcp.smile_at(t)?.forward();
        let err1 = f1 - fwd_mkt;
        let err2 = f2 - fwd_mkt;
        let err3 = f3 - fwd_mkt;
        println!(
            "{t:>6.3}  {fwd_mkt:>10.2}  {:>+8.1} ({:>+.1}%)  {:>+8.1} ({:>+.1}%)  {:>+8.1} ({:>+.1}%)",
            err1,
            err1 / fwd_mkt * 100.0,
            err2,
            err2 / fwd_mkt * 100.0,
            err3,
            err3 / fwd_mkt * 100.0,
        );

        assert!((f3 - fwd_mkt).abs() < 1.0, "pcp forward mismatch at T={t}");
    }

    println!();
    println!("Rate-only forwards overshoot because S&P 500 dividends (~1.3%/yr) reduce");
    println!("the forward below the risk-free growth rate. dividend_yield(q) corrects for");
    println!("medium/long tenors but a single flat q can't capture the discrete dividend");
    println!("term structure. add_tenor_with_forward() uses the exact market-implied");
    println!("forward from put-call parity — the standard approach in practice.");

    Ok(())
}
