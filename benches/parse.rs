use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use polypath::ObjObject;

fn read_obj(path: &str) -> ObjObject {
    ObjObject::read_from_file(path).unwrap()
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("cube", |b| {
        b.iter(|| read_obj(black_box("./meshes/cube.obj")))
    });

    c.bench_function("cheburashka", |b| {
        b.iter(|| read_obj(black_box("./meshes/cheburashka.obj")))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
