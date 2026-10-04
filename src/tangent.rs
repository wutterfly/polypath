//! Tangents: the direction on a surface that a normal map's red channel is measured along.
//!
//! A normal map stores the direction of the surface at every texel, but relative to the surface:
//! red is how far it leans along the *tangent* (the direction in which the texture coordinate `u`
//! grows), green how far along the *bitangent* (the direction in which `v` grows) and blue how far
//! along the normal. To use it, a renderer needs that frame at every point, and the cheap way is to
//! work it out once for every vertex, as a tangent plus a sign (the bitangent is
//! `cross(normal, tangent) * sign`: the sign is what lets a model with mirrored texture
//! coordinates keep a map that is right on both halves).
//!
//! [`generate_tangents`] works that out for an indexed mesh. The frames follow the texture
//! coordinates as they are given: `v` growing towards the bitangent, which is the convention of
//! OpenGL normal maps (green up) and of the `.obj` format. A renderer that flips `v` afterwards
//! has to flip the green channel with it (or the sign).

use crate::VertexTextureData;

/// How small a number has to be to count as zero: for the area of a triangle in texture space and
/// for the length of a tangent.
const EPSILON: f32 = 1e-12;

/// A tangent: `x`, `y` and `z` of a unit vector at a right angle to the vertex normal, and `w`.
///
/// `w` is 1 or -1 and gives the side of the bitangent (`cross(normal, tangent) * w`). All zero
/// when the vertex has no frame: no normal, no texture coordinate, or only triangles whose texture
/// coordinates have no area.
pub type Tangent = [f32; 4];

/// Works out the tangent of every vertex of an indexed mesh: `indices` are three for each triangle,
/// into `vertices`. Returns one [`Tangent`] for each vertex, in the same order.
///
/// The tangent of a triangle follows its texture coordinates. A vertex that several triangles share
/// gets the average of theirs, each weighted by the angle the triangle has at the vertex (so that a
/// fan of many thin triangles does not outweigh one wide one), made to stand at a right angle to
/// the vertex normal.
///
/// A vertex needs a normal and a texture coordinate to have a frame. Where a mirrored texture
/// meets itself, the vertices on the seam have to be separate for each side (as they are in an
/// `.obj` file whose texture coordinates differ there), or the average cancels out.
///
/// # Panics
/// If an index is not below the number of vertices.
#[must_use]
pub fn generate_tangents(vertices: &[VertexTextureData], indices: &[usize]) -> Vec<Tangent> {
    // What every triangle adds to its corners: the direction of `u` (tangent) and of `v`
    // (bitangent), as they lie on the surface.
    let mut tangents = vec![[0.0_f32; 3]; vertices.len()];
    let mut bitangents = vec![[0.0_f32; 3]; vertices.len()];

    for triangle in indices.as_chunks::<3>().0 {
        let corners = *triangle;
        let positions = corners.map(|index| vertices[index].vertex.position);
        let uvs = corners.map(|index| vertices[index].vertex.texture_coord);
        let [Some(uv0), Some(uv1), Some(uv2)] = uvs else {
            continue;
        };

        let edge1 = sub(to_array(positions[1]), to_array(positions[0]));
        let edge2 = sub(to_array(positions[2]), to_array(positions[0]));
        let (du1, dv1) = (uv1.0 - uv0.0, uv1.1 - uv0.1);
        let (du2, dv2) = (uv2.0 - uv0.0, uv2.1 - uv0.1);

        // The area of the triangle in texture space, twice, with a sign: it is negative when the
        // texture is mirrored there. Without area there is no direction to find.
        let determinant = du1.mul_add(dv2, -(du2 * dv1));
        if determinant.abs() < EPSILON {
            continue;
        }
        let inverse = 1.0 / determinant;
        let tangent = scale(sub(scale(edge1, dv2), scale(edge2, dv1)), inverse);
        let bitangent = scale(sub(scale(edge2, du1), scale(edge1, du2)), inverse);

        for (corner, &index) in corners.iter().enumerate() {
            let weight = corner_angle(
                to_array(positions[corner]),
                to_array(positions[(corner + 1) % 3]),
                to_array(positions[(corner + 2) % 3]),
            );
            add_scaled(&mut tangents[index], tangent, weight);
            add_scaled(&mut bitangents[index], bitangent, weight);
        }
    }

    vertices
        .iter()
        .zip(tangents.iter().zip(&bitangents))
        .map(|(vertex, (&tangent, &bitangent))| {
            let Some(normal) = vertex.vertex.normal else {
                return [0.0; 4];
            };
            frame(to_array(normal), tangent, bitangent)
        })
        .collect()
}

/// The tangent for a vertex with `normal` out of what its triangles added up to.
fn frame(normal: [f32; 3], tangent: [f32; 3], bitangent: [f32; 3]) -> Tangent {
    let length = dot(normal, normal).sqrt();
    if length < EPSILON {
        return [0.0; 4];
    }
    let normal = scale(normal, 1.0 / length);

    // The part of the tangent that is at a right angle to the normal (the normal may lean away
    // from the surface the triangles lie in, and it is the one the light uses).
    let tangent = sub(tangent, scale(normal, dot(normal, tangent)));
    let tangent_length = dot(tangent, tangent).sqrt();
    if tangent_length < EPSILON {
        return [0.0; 4];
    }
    let tangent = scale(tangent, 1.0 / tangent_length);

    // The bitangent is on the side that `v` grows towards; it is the other one if the texture is
    // mirrored.
    let handedness = if dot(cross(normal, tangent), bitangent) < 0.0 {
        -1.0
    } else {
        1.0
    };
    [tangent[0], tangent[1], tangent[2], handedness]
}

/// The angle at `corner` of the triangle made with the two other points, in radians.
fn corner_angle(corner: [f32; 3], next: [f32; 3], previous: [f32; 3]) -> f32 {
    let a = sub(next, corner);
    let b = sub(previous, corner);
    let lengths = (dot(a, a) * dot(b, b)).sqrt();
    if lengths < EPSILON {
        return 0.0;
    }
    (dot(a, b) / lengths).clamp(-1.0, 1.0).acos()
}

fn to_array(point: (f32, f32, f32)) -> [f32; 3] {
    point.into()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], factor: f32) -> [f32; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

const fn add_scaled(target: &mut [f32; 3], a: [f32; 3], factor: f32) {
    target[0] = a[0].mul_add(factor, target[0]);
    target[1] = a[1].mul_add(factor, target[1]);
    target[2] = a[2].mul_add(factor, target[2]);
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VertexData;

    fn vertex(
        position: (f32, f32, f32),
        normal: Option<(f32, f32, f32)>,
        uv: Option<(f32, f32)>,
    ) -> VertexTextureData {
        VertexTextureData {
            material_index: 0,
            vertex: VertexData {
                position,
                color: None,
                normal,
                texture_coord: uv,
            },
        }
    }

    const UP: Option<(f32, f32, f32)> = Some((0.0, 0.0, 1.0));

    /// A square in the XY plane facing +Z, as two triangles, with the texture coordinate of each
    /// corner given by `uv` (called with x and y).
    fn quad(uv: impl Fn(f32, f32) -> (f32, f32)) -> (Vec<VertexTextureData>, Vec<usize>) {
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let vertices = corners
            .iter()
            .map(|&(x, y)| vertex((x, y, 0.0), UP, Some(uv(x, y))))
            .collect();
        (vertices, vec![0, 1, 2, 0, 2, 3])
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn assert_tangent(tangent: Tangent, expected: [f32; 4]) {
        assert!(
            tangent.iter().zip(expected).all(|(&a, b)| close(a, b)),
            "{tangent:?} is not {expected:?}"
        );
    }

    #[test]
    fn a_square_whose_u_grows_along_x_has_the_tangent_along_x() {
        let (vertices, indices) = quad(|x, y| (x, y));

        let tangents = generate_tangents(&vertices, &indices);

        assert_eq!(tangents.len(), 4);
        for tangent in tangents {
            // `v` grows along +Y, which is `cross(normal, tangent)`: the usual side.
            assert_tangent(tangent, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn a_mirrored_texture_flips_the_side_of_the_bitangent_and_not_the_tangent() {
        // `v` grows towards -Y.
        let (vertices, indices) = quad(|x, y| (x, -y));

        for tangent in generate_tangents(&vertices, &indices) {
            assert_tangent(tangent, [1.0, 0.0, 0.0, -1.0]);
        }
    }

    #[test]
    fn the_tangent_follows_the_direction_u_grows_in() {
        // `u` grows along +Y, and `v` towards -X: the texture is turned a quarter.
        let (vertices, indices) = quad(|x, y| (y, -x));

        for tangent in generate_tangents(&vertices, &indices) {
            // `cross(normal, tangent)` is -X here: the side `v` grows towards.
            assert_tangent(tangent, [0.0, 1.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn the_size_of_the_texture_on_the_surface_does_not_change_the_direction() {
        let (vertices, indices) = quad(|x, y| (x * 8.0, y * 0.25));

        for tangent in generate_tangents(&vertices, &indices) {
            assert_tangent(tangent, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn a_tangent_is_at_a_right_angle_to_a_normal_that_leans_and_is_a_unit_vector() {
        // The surface is the XY plane, but the normals lean towards +X.
        let lean = Some((0.6, 0.0, 0.8));
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let vertices: Vec<_> = corners
            .iter()
            .map(|&(x, y)| vertex((x, y, 0.0), lean, Some((x, y))))
            .collect();

        for tangent in generate_tangents(&vertices, &[0, 1, 2, 0, 2, 3]) {
            let length = dot(
                [tangent[0], tangent[1], tangent[2]],
                [tangent[0], tangent[1], tangent[2]],
            );
            let across = dot([tangent[0], tangent[1], tangent[2]], [0.6, 0.0, 0.8]);
            assert!(close(length, 1.0), "{tangent:?}");
            assert!(close(across, 0.0), "{tangent:?}");
            assert!(close(tangent[3].abs(), 1.0));
        }
    }

    #[test]
    fn a_vertex_that_triangles_share_gets_the_same_tangent_from_each() {
        // Vertices 0 and 2 are in both triangles of the square.
        let (vertices, indices) = quad(|x, y| (x, y));

        let tangents = generate_tangents(&vertices, &indices);

        assert_tangent(tangents[0], tangents[2]);
        assert_tangent(tangents[1], tangents[3]);
    }

    #[test]
    fn a_vertex_where_the_texture_turns_gets_a_tangent_between_those_of_its_triangles() {
        // Vertex 0 is shared by two triangles, at the same angle (a quarter turn) each. In the first
        // `u` grows along +X, in the second along +Y: the average points in between.
        let vertices = vec![
            vertex((0.0, 0.0, 0.0), UP, Some((0.0, 0.0))),
            vertex((1.0, 0.0, 0.0), UP, Some((1.0, 0.0))),
            vertex((0.0, 1.0, 0.0), UP, Some((0.0, 1.0))),
            vertex((-1.0, 0.0, 0.0), UP, Some((0.0, 1.0))),
            vertex((0.0, 1.0, 0.0), UP, Some((1.0, 0.0))),
        ];

        let tangents = generate_tangents(&vertices, &[0, 1, 2, 0, 3, 4]);

        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert_tangent(tangents[0], [half, half, 0.0, 1.0]);
        // The vertices that belong to one triangle only keep its own direction.
        assert_tangent(tangents[1], [1.0, 0.0, 0.0, 1.0]);
        assert_tangent(tangents[4], [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn no_vertex_has_a_frame_without_a_normal_or_a_texture_coordinate() {
        let vertices = vec![
            vertex((0.0, 0.0, 0.0), None, Some((0.0, 0.0))),
            vertex((1.0, 0.0, 0.0), UP, None),
            vertex((0.0, 1.0, 0.0), UP, Some((0.0, 1.0))),
        ];

        for tangent in generate_tangents(&vertices, &[0, 1, 2]) {
            assert_eq!(tangent, [0.0; 4]);
        }
    }

    #[test]
    fn a_triangle_whose_texture_coordinates_have_no_area_gives_no_frame() {
        let vertices = vec![
            vertex((0.0, 0.0, 0.0), UP, Some((0.5, 0.5))),
            vertex((1.0, 0.0, 0.0), UP, Some((0.5, 0.5))),
            vertex((0.0, 1.0, 0.0), UP, Some((0.5, 0.5))),
        ];

        for tangent in generate_tangents(&vertices, &[0, 1, 2]) {
            assert_eq!(tangent, [0.0; 4]);
        }
    }

    #[test]
    fn a_triangle_with_no_area_in_space_does_not_poison_its_neighbours() {
        // The last triangle has its three corners on a line.
        let (mut vertices, mut indices) = quad(|x, y| (x, y));
        vertices.push(vertex((2.0, 0.0, 0.0), UP, Some((2.0, 0.0))));
        vertices.push(vertex((3.0, 0.0, 0.0), UP, Some((3.0, 0.0))));
        indices.extend([1, 4, 5]);

        for tangent in generate_tangents(&vertices, &indices).into_iter().take(4) {
            assert!(tangent.iter().all(|value| value.is_finite()), "{tangent:?}");
            assert_tangent(tangent, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn no_triangles_and_no_vertices_are_fine() {
        assert_eq!(generate_tangents(&[], &[]), Vec::<Tangent>::new());
        let vertices = vec![vertex((0.0, 0.0, 0.0), UP, Some((0.0, 0.0)))];
        assert_eq!(generate_tangents(&vertices, &[]), [[0.0; 4]]);
        // Indices that are not a whole triangle are left out.
        assert_eq!(generate_tangents(&vertices, &[0, 0]), [[0.0; 4]]);
    }
}
