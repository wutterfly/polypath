use crate::{VertexTextureData, vec3::Vec3};

/// A bounding sphere around a cluster of points.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: (f32, f32, f32),
    pub radius: f32,
}

/// Builds a bounding sphere around the given points.
#[must_use]
pub fn build_bounding_sphere(vertices: impl Iterator<Item = (f32, f32, f32)> + Clone) -> Sphere {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;

    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    // find min/max for every axis (x,y,z): every point is below the start of the minimum and
    // above the start of the maximum, so the first point replaces both
    for p in vertices.clone().map(Vec3::from) {
        // x
        min_x = f32::min(min_x, p.x);
        max_x = f32::max(max_x, p.x);

        // y
        min_y = f32::min(min_y, p.y);
        max_y = f32::max(max_y, p.y);

        // z
        min_z = f32::min(min_z, p.z);
        max_z = f32::max(max_z, p.z);
    }

    // find axis with greatest diameter
    let center = Vec3::new(
        f32::midpoint(min_x, max_x),
        f32::midpoint(min_y, max_y),
        f32::midpoint(min_z, max_z),
    );

    // got bounding box with corners (Vec3<min_x, min_y, min_z> , Vec3<max_x, max_y, max_z>)
    // now find a sphere and make sure, each point is contained in it

    let mut radius = 0.0;
    for p in vertices.clone().map(Vec3::from) {
        let distance = Vec3::distance(p, center);
        radius = f32::max(radius, distance);
    }

    Sphere {
        center: (center.x, center.y, center.z),
        radius,
    }
}

/// Moves and scales `vertices` so that their bounding sphere is the unit sphere: centred on the
/// origin, with a radius of 1. Whatever size a model was made in, and wherever it was placed,
/// it then fits in the same space. Only the positions change: normals, colours and texture
/// coordinates stay as they are, which is right because the scale is the same on every axis.
///
/// Returns the sphere the vertices had before, to undo this (the centre is where the model
/// was, the radius its size).
///
/// Vertices that all lie at one point are only moved to the origin: there is no size to scale.
/// Nothing happens to an empty slice.
pub fn normalize_to_unit_sphere(vertices: &mut [VertexTextureData]) -> Sphere {
    let sphere = build_bounding_sphere(vertices.iter().map(|vertex| vertex.vertex.position));
    let (cx, cy, cz) = sphere.center;
    let scale = if sphere.radius > 0.0 {
        1.0 / sphere.radius
    } else {
        1.0
    };

    for vertex in vertices {
        let (x, y, z) = vertex.vertex.position;
        vertex.vertex.position = ((x - cx) * scale, (y - cy) * scale, (z - cz) * scale);
    }

    sphere
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VertexData;

    fn vertex_at(position: (f32, f32, f32)) -> VertexTextureData {
        VertexTextureData {
            material_index: 0,
            vertex: VertexData {
                position,
                normal: Some((0.0, 1.0, 0.0)),
                color: Some((0.5, 0.5, 0.5)),
                texture_coord: Some((0.25, 0.75)),
            },
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn a_sphere_around_points_far_from_the_origin_is_centered_on_them() {
        // A box from (10, 20, 30) to (12, 24, 30): its centre is (11, 22, 30), and the corners
        // are sqrt(1 + 4) away from it.
        let points = [(10.0, 20.0, 30.0), (12.0, 24.0, 30.0)];

        let sphere = build_bounding_sphere(points.iter().copied());

        assert!(close(sphere.center.0, 11.0));
        assert!(close(sphere.center.1, 22.0));
        assert!(close(sphere.center.2, 30.0));
        assert!(close(sphere.radius, 5.0_f32.sqrt()));
    }

    #[test]
    fn a_sphere_around_points_with_negative_coordinates() {
        let points = [(-3.0, -1.0, -2.0), (-1.0, -1.0, -2.0)];

        let sphere = build_bounding_sphere(points.iter().copied());

        assert!(close(sphere.center.0, -2.0));
        assert!(close(sphere.center.1, -1.0));
        assert!(close(sphere.center.2, -2.0));
        assert!(close(sphere.radius, 1.0));
    }

    #[test]
    fn every_point_is_inside_the_sphere() {
        let points = [
            (0.5, 7.0, -3.0),
            (4.0, 1.0, 2.0),
            (-2.0, 3.0, 9.0),
            (6.0, -5.0, 0.0),
        ];

        let sphere = build_bounding_sphere(points.iter().copied());

        for (x, y, z) in points {
            let (cx, cy, cz) = sphere.center;
            let distance = ((x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2)).sqrt();
            assert!(distance <= sphere.radius + 1e-4);
        }
    }

    #[test]
    fn normalizing_makes_the_unit_sphere_and_returns_the_old_one() {
        let mut vertices = [
            vertex_at((10.0, 20.0, 30.0)),
            vertex_at((14.0, 20.0, 30.0)),
            vertex_at((12.0, 23.0, 34.0)),
        ];

        let before = normalize_to_unit_sphere(&mut vertices);
        let after = build_bounding_sphere(vertices.iter().map(|vertex| vertex.vertex.position));

        assert!(close(after.center.0, 0.0));
        assert!(close(after.center.1, 0.0));
        assert!(close(after.center.2, 0.0));
        assert!(close(after.radius, 1.0));
        // The returned sphere is what it was: the centre of the box (12, 21.5, 32).
        assert!(close(before.center.0, 12.0));
        assert!(close(before.center.1, 21.5));
        assert!(close(before.center.2, 32.0));
        assert!(before.radius > 1.0);
    }

    #[test]
    fn normalizing_changes_nothing_but_positions() {
        let mut vertices = [vertex_at((1.0, 2.0, 3.0)), vertex_at((5.0, 6.0, 7.0))];
        let original = vertices[0].vertex;

        _ = normalize_to_unit_sphere(&mut vertices);

        assert_eq!(vertices[0].vertex.normal, original.normal);
        assert_eq!(vertices[0].vertex.color, original.color);
        assert_eq!(vertices[0].vertex.texture_coord, original.texture_coord);
        assert_ne!(vertices[0].vertex.position, original.position);
    }

    #[test]
    fn normalizing_keeps_the_proportions() {
        // Twice as wide as high: still twice as wide after.
        let mut vertices = [vertex_at((0.0, 0.0, 0.0)), vertex_at((4.0, 2.0, 0.0))];

        _ = normalize_to_unit_sphere(&mut vertices);

        let (ax, ay, _) = vertices[0].vertex.position;
        let (bx, by, _) = vertices[1].vertex.position;
        assert!(close((bx - ax) / (by - ay), 2.0));
    }

    #[test]
    fn normalizing_one_point_moves_it_to_the_origin() {
        let mut vertices = [vertex_at((3.0, 4.0, 5.0)), vertex_at((3.0, 4.0, 5.0))];

        let before = normalize_to_unit_sphere(&mut vertices);

        assert!(close(before.radius, 0.0));
        assert_eq!(vertices[0].vertex.position, (0.0, 0.0, 0.0));
    }

    #[test]
    fn normalizing_nothing_does_nothing() {
        let mut vertices: [VertexTextureData; 0] = [];

        _ = normalize_to_unit_sphere(&mut vertices);
    }
}
