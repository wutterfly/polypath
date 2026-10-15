//! The levels of detail of a mesh, and what making them costs.
//!
//! `cargo run --release --example lod -- <file.obj | file.gltf | file.glb> [levels] [reduction]`
//!
//! For every mesh (every primitive of a glTF) the report lists each level: its triangles, how
//! many of the mesh's vertices it still uses, the error (in the units of the positions, and as a
//! part of the size of the mesh) and the meshlets it has, and how long all of it took.

use std::time::Instant;

use polypath::{
    ObjObject, Vertex,
    gltf::GltfScene,
    lod::{self, Settings},
    meshlet::Limits,
    opt,
};

fn report<V: Vertex>(name: &str, indices: &[u32], vertices: &[V], settings: &Settings) {
    if indices.len() < 3 * 16 {
        return;
    }
    // The size of the mesh: the diagonal of the box around it.
    let (mut low, mut high) = ([f32::MAX; 3], [f32::MIN; 3]);
    for vertex in vertices {
        let (x, y, z) = vertex.position();
        for (axis, value) in [x, y, z].into_iter().enumerate() {
            low[axis] = low[axis].min(value);
            high[axis] = high[axis].max(value);
        }
    }
    let size = (0..3)
        .map(|axis| (high[axis] - low[axis]).powi(2))
        .sum::<f32>()
        .sqrt();

    let started = Instant::now();
    let levels = lod::build(
        indices,
        vertices,
        settings,
        (Limits::MAX, 0.5, false),
    );
    let took = started.elapsed();
    println!(
        "{name}: {} triangles, {} vertices, size {size:.2}, {} levels in {:.0} ms",
        indices.len() / 3,
        vertices.len(),
        levels.len(),
        took.as_secs_f64() * 1000.0
    );
    for (number, level) in levels.iter().enumerate() {
        let mut used = vec![false; vertices.len()];
        for &index in &level.indices {
            used[index as usize] = true;
        }
        println!(
            "  level {number}: {:>8} triangles, {:>8} vertices used, error {:>9.5} ({:>6.3}% of the size), {:>5} meshlets",
            level.indices.len() / 3,
            used.iter().filter(|&&used| used).count(),
            level.error,
            f64::from(level.error) / f64::from(size.max(f32::MIN_POSITIVE)) * 100.0,
            level.meshlets.len()
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a file to read");
    let mut settings = Settings::default();
    if let Some(levels) = args.next() {
        settings.levels = levels.parse().expect("a number of levels");
    }
    if let Some(reduction) = args.next() {
        settings.reduction = reduction.parse().expect("a number from 0.05 to 0.95");
    }
    println!("{settings:?}");

    if std::path::Path::new(&path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("obj"))
    {
        let object = ObjObject::read_from_file(&path).expect("a readable .obj");
        let (soup, _) = object.vertices();
        let (indices, vertices) = opt::indexed_vertices(&soup);
        let indices: Vec<u32> = indices.iter().map(|&i| u32::try_from(i).unwrap()).collect();
        report(&path, &indices, &vertices, &settings);
    } else {
        let scene = GltfScene::read_from_file(&path).expect("a readable glTF");
        for (number, mesh) in scene.meshes.iter().enumerate() {
            for (part, primitive) in mesh.primitives.iter().enumerate() {
                let name = format!("mesh {number} part {part}");
                report(&name, &primitive.indices, &primitive.positions, &settings);
            }
        }
    }
}
