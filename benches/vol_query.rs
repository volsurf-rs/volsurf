use criterion::{criterion_group, criterion_main};

fn vol_query_benchmarks(_c: &mut criterion::Criterion) {
    // TODO: Add benchmarks once surface types are implemented
}

criterion_group!(benches, vol_query_benchmarks);
criterion_main!(benches);
