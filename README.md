![example workflow](https://github.com/wutterfly/polypath/actions/workflows/rust.yml/badge.svg)

# Polypath

"A very basic file parser for .obj and .mtl files."


Allows to read in *.obj* files, extraxt vertices or iterate over contained *objects*, *groups*, *faces* and vertices.



# Example

Manually iterating over each *object*, *group* and *face*.

```rust
use polypath::ObjObject;

fn main() {
    let start = std::time::Instant::now();

    // read .obj file
    // automatic triangulates faces (from max 4 vertices per face)
    let mesh = "./meshes/cheburashka.obj";
    let obj = ObjObject::read_from_file(mesh).unwrap();
    println!(
        "[{mesh}] took [{}ms] with [{} vertices]",
        start.elapsed().as_millis(),
        obj.vert_count()
    );

    // an .obj file can contain multiple objects
    // this library interprets .obj with the following hirachy:
    // .obj file -> [objects] -> [groups] -> [faces] -> [vertices]
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
                // extract set of 3 vertices from each face
                let [v1, v2, v3] = f.vertices();

                println!("Positions: ");
                println!("{:?}", v1.position);
                println!("{:?}", v2.position);
                println!("{:?}", v3.position);
            }
        }
    }
}

```


Directly get vertices, but loosing any grouping done via *objects* or *groups*.

```rust
use polypath::ObjObject;

fn main() {
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
    // constructs an index buffer, deduplicating the raw vertices
    let (indicies, verts, _) = obj.vertices_indexed();
    println!("indicies: {}  --  verts: {}", indicies.len(), verts.len());
}
```

# Missing features:

- smooth shading
  - "s off", "s 1"
- reading .mtl files
- vertex normal calculation


# Supported .obj Features
- vertices ("v )
  + colors
- vertex normals ("vn ")
- vertex texture coords ("vt ")
- objects ("o ")
- groups ("g ")
- faces ("f ")
  - max 4 vertices per face
- comments ("# ")
  - get ignored
- material library ("mtllib ")
- material use ("mtluse ")





# Test Model Sources:
- https://github.com/alecjacobson/common-3d-test-models/tree/master