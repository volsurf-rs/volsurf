use criterion::{criterion_group, criterion_main};

fn surface_construction_benchmarks(_c: &mut criterion::Criterion) {
    // TODO: Add benchmarks once surface builder is implemented
}

criterion_group!(benches, surface_construction_benchmarks);
criterion_main!(benches);
