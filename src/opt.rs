use std::collections::{HashMap, HashSet, hash_map::Entry};

use rustc_hash::FxBuildHasher;

use crate::VertexTextureData;

#[must_use]
/// Optimizes the ordering of vertices.
///
/// Takes a list of verticies, where every set of 3 vertices is assumed 1 triangle. Reorders the vertices for optimal cache reuse.
pub fn optimize_vertex_order(mut vertices: Vec<VertexTextureData>) -> Vec<VertexTextureData> {
    if vertices.is_empty() {
        return Vec::new();
    }

    assert_eq!(vertices.len() % 3, 0, "Every 3 vertices are 1 triangle");

    let mut new_vertices = Vec::with_capacity(vertices.len());
    let vc = vertices.len();

    // keep track of which faces were already added
    let mut current_faces = HashSet::with_capacity_and_hasher(vertices.len() / 3, FxBuildHasher);

    // build adjacency map
    let mut adjacency: HashMap<_, HashSet<_, _>, _> =
        HashMap::with_capacity_and_hasher(vertices.len(), FxBuildHasher);

    let face_iter = vertices
        .chunks(3)
        .map(|face| <[VertexTextureData; 3]>::try_from(face).unwrap());
    for face in face_iter {
        for vertex in face {
            match adjacency.entry(vertex) {
                Entry::Occupied(mut occupied_entry) => {
                    _ = occupied_entry.get_mut().insert([face[0], face[1], face[2]]);
                }
                Entry::Vacant(vacant_entry) => {
                    _ = vacant_entry
                        .insert(HashSet::with_hasher(FxBuildHasher))
                        .insert([face[0], face[1], face[2]]);
                }
            }
        }
    }

    // keep track of recently added vertices
    let mut stack = Vec::new();
    let mut current = vertices.pop().expect("Checked at the start");

    loop {
        // check if there are un-added faces for current vertex
        if let Some(faces) = adjacency.get_mut(&current) {
            // add all faces that use current vertex
            for [v0, v1, v2] in faces.drain() {
                if current_faces.insert([v0, v1, v2]) {
                    new_vertices.push(v0);
                    new_vertices.push(v1);
                    new_vertices.push(v2);

                    stack.push(v0);
                    stack.push(v1);
                    stack.push(v2);
                }
            }
        }

        // every face of current vertex is added
        if let Some(rm) = adjacency.remove(&current) {
            debug_assert!(rm.is_empty());
        }

        // first try to use a recently used vertex
        if let Some(next) = stack.pop() {
            current = next;
        }
        // then take the next vertex
        else if let Some(next) = vertices.pop() {
            current = next;
        }
        // no more vertices
        else {
            break;
        }
    }

    debug_assert!(stack.is_empty());
    debug_assert!(vertices.is_empty());
    debug_assert!(adjacency.is_empty());

    // make sure that if we removed a dublicate face, the vertex count is still correct
    debug_assert_eq!((vc - new_vertices.len()) % 3, 0);

    new_vertices
}

#[must_use]
/// Returns:
/// - a [Vec][std::vec::Vec] containing each unqiue vertex.
/// - a [Vec][std::vec::Vec] containing indicies into the vertex buffer. Every 3 indicies build a face.
pub fn indexed_vertices(vertices: &[VertexTextureData]) -> (Vec<usize>, Vec<VertexTextureData>) {
    let mut indicies = Vec::with_capacity(vertices.len());
    let mut vertices_new = Vec::with_capacity(vertices.len() / 3);

    let mut index_map = HashMap::<VertexTextureData, usize, _>::with_capacity_and_hasher(
        vertices.len(),
        FxBuildHasher,
    );

    let mut index_c = 0;

    let face_iter = vertices.chunks(3).map(|face| {
        let face: [VertexTextureData; 3] = TryFrom::try_from(face).unwrap();
        face
    });
    for face in face_iter {
        //
        for vertex in face {
            match index_map.entry(vertex) {
                Entry::Occupied(occupied_entry) => {
                    let index = *occupied_entry.get();
                    indicies.push(index);
                }
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(index_c);
                    vertices_new.push(vertex);
                    indicies.push(index_c);
                    index_c += 1;
                }
            }
        }
    }

    (indicies, vertices_new)
}
