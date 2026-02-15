use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use volsurf::smile::{SmileSection, SviSmile};
use volsurf::surface::SurfaceBuilder;

/// Generate synthetic SVI market data (strike, vol) pairs for benchmarking.
fn generate_market_data(forward: f64, expiry: f64, n_strikes: usize) -> Vec<(f64, f64)> {
    let svi = SviSmile::new(forward, expiry, 0.04, 0.10, -0.30, 0.0, 0.20)
        .expect("benchmark SVI params should be valid");
    let k_min = forward * 0.8;
    let k_max = forward * 1.2;
    (0..n_strikes)
        .map(|i| {
            let k = k_min + (k_max - k_min) * (i as f64 / (n_strikes - 1) as f64);
            let v = svi.vol(k).expect("SVI vol should succeed").0;
            (k, v)
        })
        .collect()
}

/// Generate multi-tenor surface data: Vec<(expiry, strikes, vols)>.
fn generate_surface_data(
    spot: f64,
    rate: f64,
    n_tenors: usize,
    n_strikes: usize,
) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
    (1..=n_tenors)
        .map(|i| {
            let t = i as f64 * 0.25;
            let fwd = spot * (rate * t).exp();
            let data = generate_market_data(fwd, t, n_strikes);
            let (strikes, vols): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
            (t, strikes, vols)
        })
        .collect()
}

fn calibration_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration");

    // SVI calibration — Zeliade quasi-explicit method, 20 strikes
    let forward = 100.0;
    let expiry = 0.25;
    let market_data = generate_market_data(forward, expiry, 20);
    group.bench_function("svi_calibration", |b| {
        b.iter(|| {
            SviSmile::calibrate(black_box(forward), black_box(expiry), black_box(&market_data))
                .unwrap()
        });
    });

    group.finish();
}

fn construction_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    let spot = 100.0;
    let rate = 0.05;

    // 5 tenors x 20 strikes
    let data_5 = generate_surface_data(spot, rate, 5, 20);
    group.bench_function("surface_build_5_tenors", |b| {
        b.iter(|| {
            let mut builder = SurfaceBuilder::new().spot(black_box(spot)).rate(black_box(rate));
            for (t, strikes, vols) in &data_5 {
                builder = builder.add_tenor(*t, strikes, vols);
            }
            builder.build().unwrap()
        });
    });

    // 20 tenors x 30 strikes — target < 10ms
    let data_20 = generate_surface_data(spot, rate, 20, 30);
    group.bench_function("surface_build_20_tenors", |b| {
        b.iter(|| {
            let mut builder = SurfaceBuilder::new().spot(black_box(spot)).rate(black_box(rate));
            for (t, strikes, vols) in &data_20 {
                builder = builder.add_tenor(*t, strikes, vols);
            }
            builder.build().unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, calibration_benchmarks, construction_benchmarks);
criterion_main!(benches);
