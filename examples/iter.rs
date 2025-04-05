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

        // an .obj file can contain multiple objects
        // this library interprets .obj with the following hirachy:
        // .obj file -> [objects] -> [groups] -> [faces] -> [verticies]
        // iterate over all contained objects
        for o in obj.objects_iter() {
            println!("Object name: {}", o.name());
            println!("Object material: {:?}", o.mtllib());

            // an .obj file can contain multiple groups
            for g in o.group_iter() {
                println!("Group name: {}", g.name());
                println!("Group material: {:?}", g.mtluse());

                // each group can contain multiple faces
                for f in g.faces_iter() {
                    // extract set of 3 verticies from each face
                    let [v1, v2, v3] = f.verticies();

                    println!("Positions: ");
                    println!("{:?}", v1.position);
                    println!("{:?}", v2.position);
                    println!("{:?}", v3.position);
                }
            }
        }
    }
}
