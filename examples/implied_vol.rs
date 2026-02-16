//! Extract implied volatility from option prices.
//!
//! Shows how to:
//!   - Price an option with Black-Scholes
//!   - Extract implied vol via Jäckel's algorithm
//!   - Verify round-trip accuracy
//!
//! Run with: `cargo run --example implied_vol`

use volsurf::implied::{black_price, BlackImpliedVol};
use volsurf::OptionType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let forward = 100.0;
    let strike = 105.0;
    let expiry = 0.5; // 6 months
    let vol = 0.25; // 25% implied vol

    // ---------------------------------------------------------------
    // 1. Price a call and a put
    // ---------------------------------------------------------------

    let call_price = black_price(forward, strike, vol, expiry, OptionType::Call)?;
    let put_price = black_price(forward, strike, vol, expiry, OptionType::Put)?;

    println!("Black-Scholes pricing (undiscounted)");
    println!("  Forward: {forward}");
    println!("  Strike:  {strike}");
    println!("  Expiry:  {expiry}y");
    println!("  Vol:     {:.0}%", vol * 100.0);
    println!();
    println!("  Call price: {call_price:.6}");
    println!("  Put price:  {put_price:.6}");
    println!(
        "  Put-call parity check: C - P = {:.6}, F - K = {:.6}",
        call_price - put_price,
        forward - strike
    );

    // ---------------------------------------------------------------
    // 2. Extract implied vol from the prices
    // ---------------------------------------------------------------

    let iv_call = BlackImpliedVol::compute(call_price, forward, strike, expiry, OptionType::Call)?;
    let iv_put = BlackImpliedVol::compute(put_price, forward, strike, expiry, OptionType::Put)?;

    println!("\nImplied vol extraction (Jäckel)");
    println!("  From call: {:.12}", iv_call.0);
    println!("  From put:  {:.12}", iv_put.0);
    println!("  Input vol: {:.12}", vol);

    // ---------------------------------------------------------------
    // 3. Round-trip accuracy
    // ---------------------------------------------------------------

    let call_reprice = black_price(forward, strike, iv_call.0, expiry, OptionType::Call)?;
    let round_trip_error = (call_price - call_reprice).abs();
    println!("\nRound-trip accuracy");
    println!("  Original price:  {call_price:.15}");
    println!("  Repriced:        {call_reprice:.15}");
    println!("  Error:           {round_trip_error:.2e}");

    // ---------------------------------------------------------------
    // 4. Scan across strikes
    // ---------------------------------------------------------------

    println!("\n--- IV extraction across strikes ---\n");
    println!("{:>8} {:>12} {:>12} {:>14}", "Strike", "Call Price", "IV", "Round-trip err");
    println!("{}", "-".repeat(50));

    for k in [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0] {
        let price = black_price(forward, k, vol, expiry, OptionType::Call)?;
        let iv = BlackImpliedVol::compute(price, forward, k, expiry, OptionType::Call)?;
        let reprice = black_price(forward, k, iv.0, expiry, OptionType::Call)?;
        let err = (price - reprice).abs();
        println!("{k:>8.0} {price:>12.6} {:>11.8}% {err:>14.2e}", iv.0);
    }

    Ok(())
}
