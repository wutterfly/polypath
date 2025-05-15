use crate::bounding::{Sphere, build_bounding_sphere};

use super::vec3::Vec3;

use super::Vertex;

/// Represent a cluster of triangles.
///
/// A cluster of triangle indices into a vertex buffer.
///
/// The cone component represents the average Meshlet normal (x,y,z) and an angle (w).
///
/// The bounding sphere contains all vertices for this meshlet.
#[derive(Debug)]
pub struct Meshlet<const VERTEX_COUNT: usize, const TRIANGLE_COUNT: usize> {
    pub cone: (f32, f32, f32, f32),
    pub bounding: Sphere,
    pub vertices: [u32; VERTEX_COUNT],
    pub triangles: [[u8; 3]; TRIANGLE_COUNT],
    pub vertex_count: u8,
    pub triangle_count: u8,
}

impl<const VERTEX_COUNT: usize, const TRIANGLE_COUNT: usize> Default
    for Meshlet<VERTEX_COUNT, TRIANGLE_COUNT>
{
    #[inline]
    fn default() -> Self {
        Self {
            cone: (0.0, 0.0, 0.0, 0.0),
            bounding: Sphere {
                center: (0.0, 0.0, 0.0),
                radius: 0.0,
            },
            vertices: [0; VERTEX_COUNT],
            triangles: [[0; 3]; TRIANGLE_COUNT],
            vertex_count: 0,
            triangle_count: 0,
        }
    }
}

/// Generates Meshlets from index and vertex data. Takes an additional cone threshold, that controls how wide the normal cone can be.
///
/// The cone threshold can be between \[0.1, 0.9\]. A larger cone threshold means more meshlets (meshlets don't get filled), but a more uniform triangle normal direction.
pub fn build_meshlets<const VERTEX_COUNT: usize, const TRIANGLE_COUNT: usize, V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    mut cone_threshold: f32,
) -> Vec<Meshlet<VERTEX_COUNT, TRIANGLE_COUNT>> {
    cone_threshold = f32::clamp(cone_threshold, 0.1, 0.9);

    let mut meshlets = Vec::new();

    // state of the current meshlet
    let mut meshlet: Meshlet<VERTEX_COUNT, TRIANGLE_COUNT> = Meshlet::default();
    let mut contained: Vec<i32> = vec![-1i32; vertices.len()];
    let mut current_vertices: Vec<(f32, f32, f32)> = Vec::with_capacity(VERTEX_COUNT);
    let mut current_normals: Vec<Vec3> = Vec::with_capacity(TRIANGLE_COUNT);

    // iterate of faces (set of 3 indices)
    let faces = indices
        .chunks_exact(3)
        .map(|f| <[u32; 3]>::try_from(f).unwrap());

    for [i0, i1, i2] in faces {
        //
        let normal = triangle_normal(
            Vec3::from(vertices[i0 as usize].position()),
            Vec3::from(vertices[i1 as usize].position()),
            Vec3::from(vertices[i2 as usize].position()),
        );

        // get indices into vertex buffer for current face/indices
        let va = contained[i0 as usize];
        let vb = contained[i1 as usize];
        let vc = contained[i2 as usize];

        // get number of new vertices in this face
        let additional_vertices = u8::from(va == -1) + u8::from(vb == -1) + u8::from(vc == -1);

        // check if the meshlet is full or face normals are too far apart
        let indices_full = meshlet.triangle_count as usize == meshlet.triangles.len();
        let verts_full =
            (meshlet.vertex_count + additional_vertices) as usize > meshlet.vertices.len();
        let too_wide = !check_cone_next(&current_normals, normal, cone_threshold);

        // flush meshlet
        if indices_full || verts_full || too_wide {
            debug_assert!(check_cone(&current_normals, cone_threshold));
            meshlet.cone = calc_cone(&current_normals);
            current_normals.clear();

            meshlet.bounding = build_bounding_sphere(current_vertices.iter().copied());
            current_vertices.clear();

            contained.fill(-1);
            meshlets.push(std::mem::take(&mut meshlet));
        }

        // reborrow here - implicit drop of av, bv, cv
        let [va, vb, vc] = contained
            .get_disjoint_mut([i0 as usize, i1 as usize, i2 as usize])
            .unwrap();

        // if vertex is not already in meshlet
        if *va == -1 {
            // push vertex
            *va = i32::from(meshlet.vertex_count);
            // set vertex index
            meshlet.vertices[meshlet.vertex_count as usize] = i0;
            meshlet.vertex_count += 1;
        }

        if *vb == -1 {
            // push vertex
            *vb = i32::from(meshlet.vertex_count);
            // set vertex index
            meshlet.vertices[meshlet.vertex_count as usize] = i1;
            meshlet.vertex_count += 1;
        }

        if *vc == -1 {
            // push vertex
            *vc = i32::from(meshlet.vertex_count);
            // set vertex index
            meshlet.vertices[meshlet.vertex_count as usize] = i2;
            meshlet.vertex_count += 1;
        }

        // set meshlet vertex indices
        meshlet.triangles[meshlet.triangle_count as usize] = [
            u8::try_from(*va).unwrap(),
            u8::try_from(*vb).unwrap(),
            u8::try_from(*vc).unwrap(),
        ];
        meshlet.triangle_count += 1;

        // add positions & normal for this face
        current_normals.push(normal);
        current_vertices.extend_from_slice(&[
            vertices[i0 as usize].position(),
            vertices[i1 as usize].position(),
            vertices[i2 as usize].position(),
        ]);
    }

    // check if there is already data written to meshlet
    if meshlet.triangle_count != 0 {
        // if there is index data, there has to be vertex data
        debug_assert!(meshlet.vertex_count != 0 && meshlet.triangle_count != 0);

        debug_assert!(check_cone(&current_normals, cone_threshold));
        meshlet.cone = calc_cone(&current_normals);
        meshlet.bounding = build_bounding_sphere(current_vertices.iter().copied());

        meshlets.push(meshlet);
    }

    meshlets
}

fn triangle_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    let p10 = p0 - p1;
    let p20 = p2 - p1;

    let n = Vec3::cross(&p10, &p20);

    if n == Vec3::zero() { n } else { n.normalized() }
}

fn check_cone(normals: &[Vec3], th: f32) -> bool {
    let mut avg = Vec3::zero();

    for n in normals {
        avg += *n;
    }

    if avg != Vec3::zero() {
        avg = avg.normalized();
    }

    let mut mdot = 1.0;

    for n in normals {
        let dot = Vec3::dot(&avg, n);

        mdot = f32::min(mdot, dot);

        if mdot < th {
            return false;
        }
    }

    true
}

fn check_cone_next(normals: &[Vec3], next: Vec3, th: f32) -> bool {
    let mut avg = Vec3::zero();

    for n in normals.iter().chain(std::iter::once(&next)) {
        avg += *n;
    }

    if avg != Vec3::zero() {
        avg = avg.normalized();
    }

    let mut mdot = 1.0;

    for n in normals.iter().chain(std::iter::once(&next)) {
        let dot = Vec3::dot(&avg, n);

        mdot = f32::min(mdot, dot);

        if mdot < th {
            return false;
        }
    }

    true
}

fn calc_cone(normals: &[Vec3]) -> (f32, f32, f32, f32) {
    let mut avg = Vec3::zero();

    for n in normals {
        avg += *n;
    }

    if avg != Vec3::zero() {
        avg = avg.normalized();
    }

    let mut mdot = 1.0;

    for n in normals {
        let dot = Vec3::dot(&avg, n);

        mdot = f32::min(mdot, dot);
    }

    let conew = if mdot <= 0.0 {
        1.0
    } else {
        f32::sqrt(mdot.mul_add(-mdot, 1.0))
    };

    (avg.x, avg.y, avg.z, conew)
}
