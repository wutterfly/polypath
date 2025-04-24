use std::{fmt::Write as _, io::Write as _};

use polypath::{ObjObject, VertexTextureData, opt};

#[test]
fn test_write_back() {
    let obj = ObjObject::read_from_file("./meshes/cheburashka.obj").unwrap();

    _object_to_file(&obj);

    let (verts, _) = obj.vertices();
    println!("verts: {}", verts.len());

    _vertex_to_file(verts);

    let (vertices, _) = obj.vertices();
    let (indicies, verts) = opt::indexed_vertices(&vertices);

    println!("indicies: {}  --  verts: {}", indicies.len(), verts.len());

    write_indexed_to_file(verts, indicies);
}

fn write_indexed_to_file(verts: Vec<VertexTextureData>, indicies: Vec<usize>) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("./meshes/test_output.obj")
        .unwrap();

    let mut normal = false;
    let mut str = String::with_capacity(1024 * 512 * 512);
    for vert in verts.iter().map(|v| v.vertex) {
        _ = writeln!(
            &mut str,
            "v {} {} {}",
            vert.position.0, vert.position.1, vert.position.2
        );

        if let Some((n1, n2, n3)) = vert.normal {
            _ = writeln!(&mut str, "vn {n1} {n2} {n3}");
            normal = true;
        }
    }

    file.write(str.as_bytes()).unwrap();
    str.clear();

    for chunk in indicies.chunks_exact(3) {
        let pos1 = chunk[0] + 1;
        let pos2 = chunk[1] + 1;
        let pos3 = chunk[2] + 1;

        if normal {
            _ = writeln!(&mut str, "f {pos1}//{pos1} {pos2}//{pos2} {pos3}//{pos3}");
        } else {
            _ = writeln!(&mut str, "f {pos1} {pos2} {pos3}");
        }
    }

    file.write(str.as_bytes()).unwrap();

    str.clear();
}

fn _vertex_to_file(verts: Vec<VertexTextureData>) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("./meshes/test_output.obj")
        .unwrap();

    let mut i = 1;
    let mut str = String::with_capacity(512);

    let mut normals = false;

    for vert in verts.chunks_exact(3) {
        let p1 = vert[0].vertex.position;
        let p2 = vert[1].vertex.position;
        let p3 = vert[2].vertex.position;

        _ = writeln!(&mut str, "v {} {} {}", p1.0, p1.1, p1.2);
        _ = writeln!(&mut str, "v {} {} {}", p2.0, p2.1, p2.2);
        _ = writeln!(&mut str, "v {} {} {}\n", p3.0, p3.1, p3.2);

        if let (Some(n1), Some(n2), Some(n3)) = (
            vert[0].vertex.normal,
            vert[1].vertex.normal,
            vert[2].vertex.normal,
        ) {
            _ = writeln!(&mut str, "vn {} {} {}", n1.0, n1.1, n1.2);
            _ = writeln!(&mut str, "vn {} {} {}", n2.0, n2.1, n2.2);
            _ = writeln!(&mut str, "vn {} {} {}\n", n3.0, n3.1, n3.2);

            normals = true;
        }

        let j = i + 1;
        let k = i + 2;

        if normals {
            _ = writeln!(&mut str, "f {i}//{i} {j}//{j} {k}//{k}\n");
        } else {
            _ = writeln!(&mut str, "f {i} {j} {k}\n");
        }

        i += 3;

        file.write(str.as_bytes()).unwrap();

        str.clear();
    }
}

fn _object_to_file(obj: &ObjObject) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("./meshes/test_output.obj")
        .unwrap();

    let mut str = String::with_capacity(512);
    let mut i = 1;

    for o in obj.objects_iter() {
        _ = writeln!(&mut str, "o {}\n", o.name());

        for g in o.group_iter() {
            _ = writeln!(&mut str, "g {}\n", g.name());
            for f in g.faces_iter() {
                let mut normals = false;

                for v in f.vertices() {
                    let (p1, p2, p3) = v.position;
                    _ = writeln!(&mut str, "v {p1} {p2} {p3}");

                    if let Some((n1, n2, n3)) = v.normal {
                        _ = writeln!(&mut str, "vn {n1} {n2} {n3}\n");
                        normals = true;
                    }
                }

                let j = i + 1;
                let k = i + 2;

                if normals {
                    _ = writeln!(&mut str, "f {i}//{i} {j}//{j} {k}//{k}\n");
                } else {
                    _ = writeln!(&mut str, "f {i} {j} {k}\n");
                }

                i += 3;

                file.write(str.as_bytes()).unwrap();
                str.clear();
            }
        }
    }
}
