use crate::vec3::Vec3;

/// A bounding sphere around a cluster of points.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: (f32, f32, f32),
    pub radius: f32,
}

/// Builds a bounding sphere around the given points.
pub fn build_bounding_sphere(vertices: impl Iterator<Item = (f32, f32, f32)> + Clone) -> Sphere {
    let mut min_x = f32::MIN;
    let mut max_x = f32::MAX;

    let mut min_y = f32::MIN;
    let mut max_y = f32::MAX;

    let mut min_z = f32::MIN;
    let mut max_z = f32::MAX;

    // find min/max for every axis (x,y,z)
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
