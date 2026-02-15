use criterion::{criterion_group, criterion_main};
use volsurf::*;

fn vol_query_benchmarks(_c: &mut criterion::Criterion) {
    // Verify public API is accessible from external crates.
    let _ = (Strike(100.0), Tenor(0.25), Vol(0.20), Variance(0.04));
    // TODO: Add benchmarks once surface types are implemented
}

criterion_group!(benches, vol_query_benchmarks);
criterion_main!(benches);
