use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use polypath::ObjObject;

fn verts(obj: &ObjObject) -> Vec<polypath::VertexTextureData> {
    obj.vertices().0
}

fn verts_indexed(obj: &ObjObject) -> (Vec<usize>, Vec<polypath::VertexTextureData>) {
    let (i, v, _) = obj.vertices_indexed();
    (i, v)
}

fn benchmarks(c: &mut Criterion) {
    let obj = ObjObject::read_from_file("./meshes/cube.obj").unwrap();

    let mut group = c.benchmark_group("cube.obj");
    group.bench_function("vertices", |b| b.iter(|| verts(black_box(&obj))));
    group.bench_function("vertices indexed", |b| {
        b.iter(|| verts_indexed(black_box(&obj)))
    });

    drop(group);

    let mut group = c.benchmark_group("cheburashka.obj");
    group.bench_function("vertices", |b| b.iter(|| verts(black_box(&obj))));
    group.bench_function("vertices indexed", |b| {
        b.iter(|| verts_indexed(black_box(&obj)))
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
