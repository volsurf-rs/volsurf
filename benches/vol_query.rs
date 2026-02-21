use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use volsurf::local_vol::{DupireLocalVol, LocalVol};
use volsurf::smile::{SabrSmile, SmileSection, SplineSmile, SviSmile};
use volsurf::surface::{PiecewiseSurface, SsviSurface, SurfaceBuilder, VolSurface};

/// Build a realistic SviSmile for benchmarking (SPX-like 3M smile).
fn make_svi_smile() -> SviSmile {
    SviSmile::new(100.0, 0.25, 0.04, 0.10, -0.30, 0.0, 0.20)
        .expect("benchmark SVI params should be valid")
}

/// Build a realistic SabrSmile for benchmarking (equity-like 3M smile).
fn make_sabr_smile() -> SabrSmile {
    SabrSmile::new(100.0, 0.25, 0.30, 0.5, -0.30, 0.40)
        .expect("benchmark SABR params should be valid")
}

/// Build a SplineSmile with 20 knot points derived from an SVI smile.
fn make_spline_smile() -> SplineSmile {
    let svi = make_svi_smile();
    let n = 20;
    let k_min = 80.0;
    let k_max = 120.0;
    let mut strikes = Vec::with_capacity(n);
    let mut variances = Vec::with_capacity(n);
    for i in 0..n {
        let k = k_min + (k_max - k_min) * (i as f64 / (n - 1) as f64);
        let v = svi.vol(k).expect("SVI vol should succeed").0;
        strikes.push(k);
        variances.push(v * v * 0.25); // variance = vol^2 * expiry
    }
    SplineSmile::new(100.0, 0.25, strikes, variances)
        .expect("benchmark spline params should be valid")
}

/// Build a 3-tenor PiecewiseSurface via SurfaceBuilder.
fn make_surface() -> PiecewiseSurface {
    let spot = 100.0;
    let rate = 0.05;
    let tenors = [0.25, 0.50, 1.0];
    let n_strikes = 20;

    let mut builder = SurfaceBuilder::new().spot(spot).rate(rate);
    for &t in &tenors {
        let fwd = spot * (rate * t).exp();
        let svi = SviSmile::new(fwd, t, 0.04, 0.10, -0.30, 0.0, 0.20)
            .expect("SVI params should be valid");
        let k_min = fwd * 0.8;
        let k_max = fwd * 1.2;
        let strikes: Vec<f64> = (0..n_strikes)
            .map(|i| k_min + (k_max - k_min) * (i as f64 / (n_strikes - 1) as f64))
            .collect();
        let vols: Vec<f64> = strikes
            .iter()
            .map(|&k| svi.vol(k).expect("SVI vol should succeed").0)
            .collect();
        builder = builder.add_tenor(t, &strikes, &vols);
    }
    builder.build().expect("surface build should succeed")
}

/// Build a 3-tenor SsviSurface for benchmarking.
fn make_ssvi_surface() -> SsviSurface {
    let tenors = vec![0.25, 0.50, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    // Increasing ATM total variances: vol ~20% => theta = vol^2 * T
    let thetas = vec![0.04 * 0.25, 0.04 * 0.50, 0.04 * 1.0];
    SsviSurface::new(-0.30, 1.0, 0.50, tenors, forwards, thetas)
        .expect("benchmark SSVI params should be valid")
}

fn smile_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("smile");

    // SVI vol query — pure arithmetic, target < 100ns
    let svi = make_svi_smile();
    group.bench_function("svi_vol_query", |b| {
        b.iter(|| svi.vol(black_box(100.0)).unwrap());
    });

    // Spline vol query — binary search + cubic eval, target < 100ns
    let spline = make_spline_smile();
    group.bench_function("spline_vol_query", |b| {
        b.iter(|| spline.vol(black_box(100.0)).unwrap());
    });

    // SABR vol query — Hagan formula with transcendentals, target < 100ns
    let sabr = make_sabr_smile();
    group.bench_function("sabr_vol_query", |b| {
        b.iter(|| sabr.vol(black_box(100.0)).unwrap());
    });

    // SABR density — Breeden-Litzenberger via finite differences
    group.bench_function("sabr_density", |b| {
        b.iter(|| sabr.density(black_box(100.0)).unwrap());
    });

    // SVI density — Breeden-Litzenberger via finite differences
    group.bench_function("svi_density", |b| {
        b.iter(|| svi.density(black_box(100.0)).unwrap());
    });

    // Spline density — Breeden-Litzenberger via finite differences
    group.bench_function("spline_density", |b| {
        b.iter(|| spline.density(black_box(100.0)).unwrap());
    });

    // SSVI slice vol query via smile_at() — target < 100ns
    let ssvi = make_ssvi_surface();
    let slice = ssvi.smile_at(0.25).expect("SSVI smile_at should succeed");
    group.bench_function("ssvi_slice_vol_query", |b| {
        b.iter(|| slice.vol(black_box(100.0)).unwrap());
    });

    group.finish();
}

fn surface_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface");

    // PiecewiseSurface vol query — interpolated tenor, OTM strike, target < 100ns
    let surface = make_surface();
    group.bench_function("piecewise_vol_query", |b| {
        b.iter(|| {
            surface
                .black_vol(black_box(0.375), black_box(105.0))
                .unwrap()
        });
    });

    // SSVI surface vol query — target < 100ns
    let ssvi = make_ssvi_surface();
    group.bench_function("ssvi_vol_query", |b| {
        b.iter(|| ssvi.black_vol(black_box(0.375), black_box(105.0)).unwrap());
    });

    // --- smile_at construction ---
    group.bench_function("piecewise_smile_at", |b| {
        b.iter(|| surface.smile_at(black_box(0.375)).unwrap());
    });
    group.bench_function("ssvi_smile_at", |b| {
        b.iter(|| ssvi.smile_at(black_box(0.375)).unwrap());
    });

    // --- diagnostics ---
    group.bench_function("piecewise_diagnostics", |b| {
        b.iter(|| surface.diagnostics().unwrap());
    });
    group.bench_function("ssvi_diagnostics", |b| {
        b.iter(|| ssvi.diagnostics().unwrap());
    });

    // --- SSVI calendar analytical ---
    group.bench_function("ssvi_calendar_analytical", |b| {
        b.iter(|| ssvi.calendar_arb_analytical());
    });

    group.finish();
}

fn local_vol_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_vol");

    let dupire = DupireLocalVol::new(Arc::new(make_ssvi_surface()));

    // Single-point Dupire query — interpolated tenor, slightly OTM
    group.bench_function("dupire_single_query", |b| {
        b.iter(|| dupire.local_vol(black_box(0.375), black_box(105.0)).unwrap());
    });

    // 20x30 grid — 600 local vol queries across the surface
    let expiries: Vec<f64> = (1..=20).map(|i| 0.1 + 0.1 * i as f64).collect();
    let strikes: Vec<f64> = (0..30).map(|j| 82.0 + 1.3 * j as f64).collect();
    group.bench_function("dupire_grid_20x30", |b| {
        b.iter(|| {
            for &t in black_box(&expiries) {
                for &k in black_box(&strikes) {
                    let _ = dupire.local_vol(t, k).unwrap();
                }
            }
        });
    });

    // Fine bump_size (0.001 vs default 0.01) — more FD evaluations
    let dupire_fine = DupireLocalVol::new(Arc::new(make_ssvi_surface()))
        .with_bump_size(0.001)
        .expect("bump_size 0.001 should be valid");
    group.bench_function("dupire_fine_bump", |b| {
        b.iter(|| dupire_fine.local_vol(black_box(0.375), black_box(105.0)).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    smile_benchmarks,
    surface_benchmarks,
    local_vol_benchmarks
);
criterion_main!(benches);
