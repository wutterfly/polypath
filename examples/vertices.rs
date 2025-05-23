use polypath::{ObjObject, opt};

const MESHES: &[&str] = &[
    "./meshes/armadillo.obj",   // 0
    "./meshes/cheburashka.obj", // 1
    "./meshes/cubes.obj",       // 2
];

fn main() {
    for mesh in MESHES {
        let start = std::time::Instant::now();

        // read .obj file
        // automatic triangulates faces (from max 4 vertices per face)
        let obj = ObjObject::read_from_file(mesh).unwrap();
        println!(
            "[{mesh}] took [{}ms] with [{} vertices]",
            start.elapsed().as_millis(),
            obj.vert_count()
        );

        // extract all the vertices (position, ?color, ?normal, ?texture coord, ?material index)
        // and all materials that are used (accessed by material index)
        // as for each face the possible same vertex is included, this can be rather inefficient
        // every 3 vertices are 1 face
        let (verts, _) = obj.vertices();
        println!("verts: {}", verts.len());

        // extract all the vertices (position, ?color, ?normal, ?texture coord, ?material index)
        // and all materials that are used (accessed by material index)
        let (vertices, _) = obj.vertices();

        // optimize vertex ordering
        let vertices = opt::optimize_vertex_order(vertices);

        // constructs an index buffer, deduplicating the raw vertices
        let (indicies, verts) = opt::indexed_vertices(&vertices);
        println!("indicies: {}  --  verts: {}", indicies.len(), verts.len());
    }
}
