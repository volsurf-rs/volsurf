use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use volsurf::smile::{SmileSection, SplineSmile, SviSmile};
use volsurf::surface::{PiecewiseSurface, SurfaceBuilder, VolSurface};

/// Build a realistic SviSmile for benchmarking (SPX-like 3M smile).
fn make_svi_smile() -> SviSmile {
    SviSmile::new(100.0, 0.25, 0.04, 0.10, -0.30, 0.0, 0.20)
        .expect("benchmark SVI params should be valid")
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

    group.finish();
}

fn surface_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface");

    // Surface vol query — interpolated tenor, OTM strike, target < 100ns
    let surface = make_surface();
    group.bench_function("surface_vol_query", |b| {
        b.iter(|| surface.black_vol(black_box(0.375), black_box(105.0)).unwrap());
    });

    group.finish();
}

criterion_group!(benches, smile_benchmarks, surface_benchmarks);
criterion_main!(benches);
