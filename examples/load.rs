use polypath::ObjObject;

const MESHES: &[&str] = &[
    "./meshes/armadillo.obj",   // 0
    "./meshes/cheburashka.obj", // 1
    "./meshes/cubes.obj",       // 2
];

fn main() {
    for mesh in &MESHES[..] {
        let start = std::time::Instant::now();

        // read .obj file
        // automatic triangulates faces (from max 4 verticies per face)
        let obj = ObjObject::read_from_file(mesh).unwrap();
        println!(
            "[{mesh}] took [{}ms] with [{} verticies]",
            start.elapsed().as_millis(),
            obj.vert_count()
        );

        // extract all the verticies (position, ?color, ?normal, ?texture coord, ?material index)
        // and all materials that are used (accessed by material index)
        // as for each face the possible same vertex is included, this can be rather inefficient
        // every 3 vertices are 1 face
        let (verts, _) = obj.verticies();
        println!("verts: {}", verts.len());

        // extract all the verticies (position, ?color, ?normal, ?texture coord, ?material index)
        // and all materials that are used (accessed by material index)
        // constructs an index buffer, deduplicating the raw verticies
        let (indicies, verts, _) = obj.verticies_indexed();
        println!("indicies: {}  --  verts: {}", indicies.len(), verts.len());
    }
}
