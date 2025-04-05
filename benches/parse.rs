use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use polypath::ObjObject;

fn read_obj(path: &str) -> ObjObject {
    ObjObject::read_from_file(path).expect(path)
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("cubes", |b| {
        b.iter(|| read_obj(black_box("./meshes/cubes.obj")))
    });

    c.bench_function("cheburashka", |b| {
        b.iter(|| read_obj(black_box("./meshes/cheburashka.obj")))
    });

    c.bench_function("armadillo", |b| {
        b.iter(|| read_obj(black_box("./meshes/armadillo.obj")))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
