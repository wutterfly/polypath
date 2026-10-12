//! How good the meshlets of a mesh are.
//!
//! `cargo run --release --example meshlets -- <file.obj | file.gltf | file.glb> [cone threshold]
//! [most vertices] [most triangles] [--merge]`
//!
//! `--merge` puts small meshlets that do not touch together (`merge_apart` of
//! [`polypath::meshlet::build`]).
//!
//! An `.obj` is split twice: with the triangles in the order of the file, and after
//! [`polypath::opt::optimize_vertex_order`]. A glTF is split in the order of its indices.
//!
//! Per mesh the report says how full the meshlets are (124 triangles is full), and how big the
//! spheres around them are: a meshlet of triangles that touch has a small sphere, one of triangles
//! from all over the mesh has a big one (and can never be culled). The last number is how many
//! meshlets are flat enough for the cone to cull them (the cutoff is under 0.5: they are seen from
//! behind when the viewer is within 30 degrees of the way they face).

use polypath::{
    ObjObject, opt,
    gltf::GltfScene,
    meshlet::{self, Limits},
};

fn main() {
    let merge = std::env::args().any(|arg| arg == "--merge");
    let mut args = std::env::args().skip(1).filter(|arg| arg != "--merge");
    let path = args.next().expect("a file to read");
    let threshold: f32 = args.next().map_or(0.9, |arg| arg.parse().expect("a number"));
    let mut limit = |most: usize| args.next().map_or(most, |arg| arg.parse().expect("a number"));
    let limits = Limits {
        vertices: limit(Limits::MAX.vertices),
        triangles: limit(Limits::MAX.triangles),
    };

    println!("cone threshold {threshold}, at most {limits:?}, merge apart: {merge}");
    if std::path::Path::new(&path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("obj"))
    {
        let object = ObjObject::read_from_file(&path).expect("a readable .obj");
        let (soup, _) = object.vertices();

        let (indices, vertices) = opt::indexed_vertices(&soup);
        let indices: Vec<u32> = indices.iter().map(|&i| u32::try_from(i).unwrap()).collect();
        let meshlets = meshlet::build(&indices, &vertices, limits, threshold, merge);
        summary("in the order of the file", &meshlets.records, indices.len() / 3, limits);

        let ordered = opt::optimize_vertex_order(soup);
        let (indices, vertices) = opt::indexed_vertices(&ordered);
        let indices: Vec<u32> = indices.iter().map(|&i| u32::try_from(i).unwrap()).collect();
        let meshlets = meshlet::build(&indices, &vertices, limits, threshold, merge);
        summary("after optimize_vertex_order", &meshlets.records, indices.len() / 3, limits);
    } else {
        let scene = GltfScene::read_from_file(&path).expect("a readable glTF");
        let mut all = Vec::new();
        for mesh in &scene.meshes {
            for primitive in &mesh.primitives {
                all.push(meshlet::build(
                    &primitive.indices,
                    &primitive.positions,
                    limits,
                    threshold,
                    merge,
                ));
            }
        }
        let triangles: Vec<usize> = scene
            .meshes
            .iter()
            .flat_map(|mesh| mesh.primitives.iter().map(|p| p.indices.len() / 3))
            .collect();
        println!("{} primitives", all.len());
        let combined = all.iter().flat_map(|m| m.records.iter()).copied().collect::<Vec<_>>();
        summary("in the order of the indices", &combined, triangles.iter().sum(), limits);
    }
}

fn summary(label: &str, records: &[meshlet::MeshletRecord], triangles: usize, limits: Limits) {
    let count = records.len();
    let mut radii: Vec<f32> = records.iter().map(|r| r.bounding.radius).collect();
    radii.sort_by(f32::total_cmp);
    let mean = radii.iter().sum::<f32>() / count as f32;
    let flat = records.iter().filter(|r| r.cone_cutoff < 0.5).count();
    let vertices: u32 = records.iter().map(|r| r.vertex_count).sum();
    println!("{label}");
    println!(
        "  {triangles} triangles in {count} meshlets: {:.1} triangles each (full is {}), {:.1} vertices each",
        triangles as f32 / count as f32,
        limits.triangles,
        vertices as f32 / count as f32,
    );
    println!(
        "  sphere radius: mean {mean:.3}, median {:.3}, 90% under {:.3}, largest {:.3}",
        radii[count / 2],
        radii[count * 9 / 10],
        radii[count - 1],
    );
    // How the triangles are spread over the sizes of meshlets.
    let mut buckets = [0_usize; 5];
    for record in records {
        let at = match record.triangle_count {
            0..=7 => 0,
            8..=31 => 1,
            32..=63 => 2,
            64..=123 => 3,
            _ => 4,
        };
        buckets[at] += record.triangle_count as usize;
    }
    println!(
        "  share of the triangles in meshlets of 1-7 / 8-31 / 32-63 / 64-123 / 124: {:.0}% {:.0}% {:.0}% {:.0}% {:.0}%",
        100.0 * buckets[0] as f32 / triangles as f32,
        100.0 * buckets[1] as f32 / triangles as f32,
        100.0 * buckets[2] as f32 / triangles as f32,
        100.0 * buckets[3] as f32 / triangles as f32,
        100.0 * buckets[4] as f32 / triangles as f32,
    );
    println!(
        "  cone can cull: {flat} of {count} ({:.0}%)",
        100.0 * flat as f32 / count as f32
    );
}
