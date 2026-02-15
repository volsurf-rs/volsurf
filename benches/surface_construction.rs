use criterion::{criterion_group, criterion_main};
use volsurf::surface::SurfaceBuilder;

fn surface_construction_benchmarks(_c: &mut criterion::Criterion) {
    // Verify builder API is accessible from external crates.
    let _ = SurfaceBuilder::new().spot(100.0).rate(0.05);
    // TODO: Add benchmarks once surface builder is implemented
}

criterion_group!(benches, surface_construction_benchmarks);
criterion_main!(benches);
