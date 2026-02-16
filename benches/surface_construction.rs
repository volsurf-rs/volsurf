use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use volsurf::smile::{SabrSmile, SmileSection, SviSmile};
use volsurf::surface::{SmileModel, SsviSurface, SurfaceBuilder, VolSurface};

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

/// Generate synthetic SABR market data (strike, vol) pairs for benchmarking.
fn generate_sabr_market_data(forward: f64, expiry: f64, n_strikes: usize) -> Vec<(f64, f64)> {
    let sabr = SabrSmile::new(forward, expiry, 0.30, 0.5, -0.30, 0.40)
        .expect("benchmark SABR params should be valid");
    let k_min = forward * 0.8;
    let k_max = forward * 1.2;
    (0..n_strikes)
        .map(|i| {
            let k = k_min + (k_max - k_min) * (i as f64 / (n_strikes - 1) as f64);
            let v = sabr.vol(k).expect("SABR vol should succeed").0;
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

/// Generate multi-tenor SABR surface data.
fn generate_sabr_surface_data(
    spot: f64,
    rate: f64,
    n_tenors: usize,
    n_strikes: usize,
) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
    (1..=n_tenors)
        .map(|i| {
            let t = i as f64 * 0.25;
            let fwd = spot * (rate * t).exp();
            let data = generate_sabr_market_data(fwd, t, n_strikes);
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
    let svi_data = generate_market_data(forward, expiry, 20);
    group.bench_function("svi_calibration", |b| {
        b.iter(|| {
            SviSmile::calibrate(black_box(forward), black_box(expiry), black_box(&svi_data))
                .unwrap()
        });
    });

    // SABR calibration — analytic alpha + Nelder-Mead 2D, 10 strikes, target < 1ms
    let sabr_data = generate_sabr_market_data(forward, expiry, 10);
    group.bench_function("sabr_calibration", |b| {
        b.iter(|| {
            SabrSmile::calibrate(
                black_box(forward),
                black_box(expiry),
                black_box(0.5),
                black_box(&sabr_data),
            )
            .unwrap()
        });
    });

    // SSVI calibration — two-stage SVI + Nelder-Mead, 3 tenors x 10 strikes
    // Generate data from an actual SSVI surface for clean round-trip calibration.
    let ssvi_tenors = vec![0.25, 0.50, 1.0];
    let ssvi_forwards = vec![100.0, 100.0, 100.0];
    let ssvi_source = SsviSurface::new(-0.30, 1.0, 0.50, ssvi_tenors.clone(), ssvi_forwards.clone(), vec![0.01, 0.02, 0.04])
        .expect("benchmark SSVI params should be valid");
    let ssvi_market_data: Vec<Vec<(f64, f64)>> = ssvi_tenors
        .iter()
        .zip(ssvi_forwards.iter())
        .map(|(&t, &f)| {
            let k_min = f * 0.8;
            let k_max = f * 1.2;
            (0..10)
                .map(|i| {
                    let k = k_min + (k_max - k_min) * (i as f64 / 9.0);
                    let v = ssvi_source.black_vol(t, k).expect("SSVI vol should succeed").0;
                    (k, v)
                })
                .collect()
        })
        .collect();
    group.bench_function("ssvi_calibration", |b| {
        b.iter(|| {
            SsviSurface::calibrate(
                black_box(&ssvi_market_data),
                black_box(&ssvi_tenors),
                black_box(&ssvi_forwards),
            )
            .unwrap()
        });
    });

    group.finish();
}

fn construction_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    let spot = 100.0;
    let rate = 0.05;

    // SVI surface: 5 tenors x 20 strikes
    let data_5 = generate_surface_data(spot, rate, 5, 20);
    group.bench_function("svi_surface_5_tenors", |b| {
        b.iter(|| {
            let mut builder = SurfaceBuilder::new().spot(black_box(spot)).rate(black_box(rate));
            for (t, strikes, vols) in &data_5 {
                builder = builder.add_tenor(*t, strikes, vols);
            }
            builder.build().unwrap()
        });
    });

    // SVI surface: 20 tenors x 30 strikes — target < 10ms
    let data_20 = generate_surface_data(spot, rate, 20, 30);
    group.bench_function("svi_surface_20_tenors", |b| {
        b.iter(|| {
            let mut builder = SurfaceBuilder::new().spot(black_box(spot)).rate(black_box(rate));
            for (t, strikes, vols) in &data_20 {
                builder = builder.add_tenor(*t, strikes, vols);
            }
            builder.build().unwrap()
        });
    });

    // SABR surface: 5 tenors x 10 strikes
    let sabr_data_5 = generate_sabr_surface_data(spot, rate, 5, 10);
    group.bench_function("sabr_surface_5_tenors", |b| {
        b.iter(|| {
            let mut builder = SurfaceBuilder::new()
                .spot(black_box(spot))
                .rate(black_box(rate))
                .model(SmileModel::Sabr { beta: 0.5 });
            for (t, strikes, vols) in &sabr_data_5 {
                builder = builder.add_tenor(*t, strikes, vols);
            }
            builder.build().unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, calibration_benchmarks, construction_benchmarks);
criterion_main!(benches);
