//! Meshlets: a mesh split into small clusters of triangles that touch.
//!
//! A meshlet has at most [`MAX_VERTICES`] vertices and [`MAX_TRIANGLES`] triangles (or fewer, as
//! [`Limits`] says), a sphere around
//! it and a cone of the directions its triangles face, so that a renderer can leave out what is
//! outside of the view or turned away from it a cluster at a time, and so that mesh shaders can
//! draw a cluster per workgroup. [`build`] makes them, in the flat arrays ([`Meshlets`]) that go
//! to the GPU as they are.

use crate::{
    Vertex,
    bounding::{Sphere, build_bounding_sphere},
    vec3::Vec3,
};

/// The direction that is the average of unit `normals`, of length 1; all zero if there are none
/// (or they cancel out).
fn average_direction<'a>(normals: impl Iterator<Item = &'a Vec3>) -> Vec3 {
    let mut sum = Vec3::zero();
    for normal in normals {
        sum += *normal;
    }
    if sum == Vec3::zero() {
        sum
    } else {
        sum.normalized()
    }
}

/// The smallest dot product any of `normals` has with `axis`: 1 if they all face the same way, 0
/// if one is at a right angle (a cone as wide as that can not be used for culling).
fn smallest_dot<'a>(axis: &Vec3, normals: impl Iterator<Item = &'a Vec3>) -> f32 {
    normals
        .map(|normal| Vec3::dot(axis, normal))
        .fold(1.0, f32::min)
}

/// The sine of the angle of a cone whose widest direction has the dot product `smallest_dot` with
/// its axis (1 for a cone that is not narrower than a half space).
fn cone_cutoff(smallest_dot: f32) -> f32 {
    if smallest_dot <= 0.0 {
        1.0
    } else {
        smallest_dot.mul_add(-smallest_dot, 1.0).max(0.0).sqrt()
    }
}

/// Whether the directions `normals` and `next` all are within `threshold` (a dot product) of their
/// average.
fn fits_cone(normals: &[Vec3], next: Vec3, threshold: f32) -> bool {
    let all = normals.iter().chain(std::iter::once(&next));
    let axis = average_direction(all.clone());
    smallest_dot(&axis, all) >= threshold
}

/// The most vertices a meshlet of [`build`] has.
///
/// 64 (with [`MAX_TRIANGLES`]) is what mesh shaders are happiest with: the limits of the hardware
/// are higher, but this fills the lanes of a workgroup well, and a local index is a byte.
pub const MAX_VERTICES: usize = 64;

/// The most triangles a meshlet of [`build`] has: 124, so that the bytes of the local indices (3
/// for each triangle, 372) plus a count fit a whole number of 4-byte words.
pub const MAX_TRIANGLES: usize = 124;

/// How full [`build`] makes a meshlet: at most this many vertices and triangles.
///
/// Meshlets are never fuller than the format has room for ([`MAX_VERTICES`] and [`MAX_TRIANGLES`],
/// which is what [`Limits::MAX`] is). Smaller ones have tighter spheres and cones, so more can be
/// left out, but there are more of them to test and to draw, and in a mesh shader fewer of the
/// vertices are shared. Which is better is for the renderer's own measurements to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most vertices a meshlet has: 3 (a triangle has to fit) to [`MAX_VERTICES`].
    pub vertices: usize,
    /// The most triangles a meshlet has: 1 to [`MAX_TRIANGLES`].
    pub triangles: usize,
}

impl Limits {
    /// All the format has room for.
    pub const MAX: Self = Self {
        vertices: MAX_VERTICES,
        triangles: MAX_TRIANGLES,
    };
}

impl Default for Limits {
    fn default() -> Self {
        Self::MAX
    }
}

/// A local index that no vertex has: the vertex is not in the meshlet being made.
const NOT_IN: u8 = u8::MAX;

/// One meshlet of a [`Meshlets`]: a small cluster of triangles, what is needed to tell whether it
/// can be seen, and where its parts are.
///
/// The ranges are in the arrays of the [`Meshlets`] they belong to:
///
/// - `vertices[vertex_start .. vertex_start + vertex_count]` are the vertices the meshlet uses, as
///   indices into the vertices of the mesh.
/// - `triangles[3 * triangle_start .. 3 * (triangle_start + triangle_count)]` are its triangles,
///   as indices into *that list* (so that they fit a byte).
/// - `indices[3 * triangle_start .. 3 * (triangle_start + triangle_count)]` are the same
///   triangles, as indices into the vertices of the mesh: what an ordinary indexed draw wants, so
///   a meshlet is a range of one index buffer.
#[derive(Debug, Clone, Copy)]
pub struct MeshletRecord {
    /// A sphere around the vertices of the meshlet, in the space of the mesh.
    pub bounding: Sphere,
    /// The average direction the triangles face (length 1; or all zero, for a meshlet of
    /// triangles with no area).
    pub cone_axis: (f32, f32, f32),
    /// How wide the directions the triangles face are around `cone_axis`: the sine of the angle
    /// of the widest one. 0 is a flat meshlet; the more it is the more the meshlet bends.
    ///
    /// The meshlet is seen from behind, so that none of its triangles can be seen, when
    /// `dot(centre - eye, axis) >= cutoff * length(centre - eye) + radius * (1 + cutoff)`, where
    /// `centre` and `radius` are the sphere's (the last part is the sphere's own size: every point
    /// of the meshlet is in it).
    pub cone_cutoff: f32,
    /// Where the vertices of the meshlet start in `vertices`.
    pub vertex_start: u32,
    /// How many vertices the meshlet has: 1 to [`MAX_VERTICES`].
    pub vertex_count: u32,
    /// Where the triangles of the meshlet start, in triangles, in `triangles` and in `indices`
    /// (3 numbers each).
    pub triangle_start: u32,
    /// How many triangles the meshlet has: 1 to [`MAX_TRIANGLES`].
    pub triangle_count: u32,
}

/// The meshlets of one mesh, in flat arrays that go to the GPU as they are.
///
/// The triangles of the mesh are each in exactly one meshlet, and `indices` is the index list of
/// the mesh in the order of the meshlets (so the same triangles, in another order). Drawing it
/// whole looks the same as drawing the original; drawing a range of it is drawing a meshlet.
#[derive(Debug, Clone, Default)]
pub struct Meshlets {
    /// The meshlets, in the order of their triangles.
    pub records: Vec<MeshletRecord>,
    /// For every meshlet, the vertices it uses (see [`MeshletRecord`]): indices into the vertices
    /// of the mesh, without a vertex twice in one meshlet.
    pub vertices: Vec<u32>,
    /// For every meshlet, its triangles as indices into its own part of `vertices`: 3 for each
    /// triangle, each less than the meshlet's `vertex_count`.
    pub triangles: Vec<u8>,
    /// The triangles of every meshlet as indices into the vertices of the mesh: 3 for each
    /// triangle (so `3 * triangle_start` is where a meshlet starts, for a draw).
    pub indices: Vec<u32>,
}

impl Meshlets {
    /// How many meshlets there are.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether there are none (the mesh has no triangles).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Splits a mesh into meshlets of at most `limits.vertices` vertices and `limits.triangles`
/// triangles.
///
/// `indices` are the triangles (3 numbers each) of a mesh with `vertices`. A meshlet is **grown**:
/// it starts at a triangle (next to where the last one ended, so that the meshlets follow the
/// surface) and takes, one by one, the triangle that touches it and is the best fit: no new
/// vertices if there is one, otherwise the nearest, and the one that faces the way the meshlet
/// already does. It is closed when nothing that touches it fits, so that the order of `indices`
/// does not matter, and neither does it matter that vertices are doubled where the texture has a
/// seam (triangles that touch at the same place are neighbours).
///
/// `cone_threshold` is the smallest dot product any triangle of a meshlet may have with the average
/// direction they face (between 0.1 and 0.9: a larger one gives more, flatter meshlets, which the
/// cone culls better; 0.1 is about 84 degrees either side).
///
/// The triangles are taken as counter-clockwise from the side that is the front, for which the
/// normal is `(b - a) x (c - a)`. A triangle with no area is kept in whichever meshlet it comes in,
/// but does not count for the cone.
///
/// # Panics
/// If `indices.len()` is not a multiple of 3, if an index is not a vertex of `vertices`, if
/// there are more than `u32::MAX` triangles or vertices, or if `limits` are not within
/// [`Limits::MAX`] (3 vertices and 1 triangle at least).
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "one algorithm that reads top to bottom: pick a start, grow, close; its state is shared by all three parts"
)]
pub fn build<V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    limits: Limits,
    cone_threshold: f32,
) -> Meshlets {
    assert_eq!(
        indices.len() % 3,
        0,
        "the indices are triangles: 3 numbers each"
    );
    assert!(
        u32::try_from(indices.len()).is_ok() && u32::try_from(vertices.len()).is_ok(),
        "the numbers of a mesh fit 32 bits"
    );
    assert!(
        (3..=MAX_VERTICES).contains(&limits.vertices)
            && (1..=MAX_TRIANGLES).contains(&limits.triangles),
        "the limits of a meshlet are within what the format has room for"
    );

    let mut builder = Builder {
        vertices,
        limits,
        threshold: if cone_threshold.is_nan() {
            // A threshold that is not a number gets the middle of the range.
            0.5
        } else {
            cone_threshold.clamp(0.1, 0.9)
        },
        local: vec![NOT_IN; vertices.len()],
        members: Vec::with_capacity(limits.vertices),
        triangles: Vec::with_capacity(limits.triangles * 3),
        normals: Vec::with_capacity(limits.triangles),
        out: Meshlets {
            indices: Vec::with_capacity(indices.len()),
            ..Meshlets::default()
        },
    };

    let faces = indices.as_chunks::<3>().0;
    let neighbours = Neighbours::new(faces, vertices);
    let normals: Vec<Option<Vec3>> = faces
        .iter()
        .map(|&face| builder.face_normal(face))
        .collect();
    let centroids: Vec<Vec3> = faces
        .iter()
        .map(|&[a, b, c]| {
            let (a, b, c) = (
                builder.position(a),
                builder.position(b),
                builder.position(c),
            );
            Vec3::new(
                (a.x + b.x + c.x) / 3.0,
                (a.y + b.y + c.y) / 3.0,
                (a.z + b.z + c.z) / 3.0,
            )
        })
        .collect();

    // Per triangle: whether it is in a meshlet already, and the number of the meshlet it was a
    // candidate of last (so that it is only once among the candidates of one).
    let mut taken = vec![false; faces.len()];
    let mut candidate_of = vec![u32::MAX; faces.len()];
    let mut rejected_by = vec![u32::MAX; faces.len()];
    // The triangles that touch the meshlet being made and are not in one.
    let mut frontier: Vec<u32> = Vec::new();
    // Where to look for the next triangle that is in no meshlet, when there is none next to the
    // last one.
    let mut cursor = 0;
    let mut meshlet_number = 0_u32;

    loop {
        // The start: next to the meshlet that was just closed if there is a place (the triangle
        // with the fewest neighbours left: a corner, which is what would be left alone
        // otherwise), or the next one that is left.
        frontier.retain(|&triangle| !taken[triangle as usize]);
        let seed = frontier
            .iter()
            .copied()
            .min_by_key(|&triangle| {
                (
                    neighbours.live(faces[triangle as usize], &taken, triangle),
                    triangle,
                )
            })
            .or_else(|| {
                while cursor < faces.len() && taken[cursor] {
                    cursor += 1;
                }
                (cursor < faces.len()).then(|| u32::try_from(cursor).expect("checked above"))
            });
        let Some(seed) = seed else {
            break;
        };
        frontier.clear();
        meshlet_number += 1;

        // The sum of what is known of the meshlet being made: where it is, and which way it faces.
        let mut centroid_sum = Vec3::zero();
        let mut normal_sum = Vec3::zero();
        let mut count = 0.0_f32;
        let mut next = Some(seed);

        while let Some(triangle) = next.take() {
            let at = triangle as usize;
            taken[at] = true;
            builder.push(faces[at], normals[at]);
            centroid_sum += centroids[at];
            if let Some(normal) = normals[at] {
                normal_sum += normal;
            }
            count += 1.0;
            for neighbour in neighbours.of(faces[at]) {
                let n = neighbour as usize;
                if !taken[n] && candidate_of[n] != meshlet_number {
                    candidate_of[n] = meshlet_number;
                    frontier.push(neighbour);
                }
            }

            if builder.triangles.len() / 3 == limits.triangles {
                break;
            }

            // The best of the candidates that fit; one that does not is left out for this meshlet.
            let centre = Vec3::new(
                centroid_sum.x / count,
                centroid_sum.y / count,
                centroid_sum.z / count,
            );
            let axis_length = normal_sum.length();
            loop {
                let mut best: Option<(f32, u32)> = None;
                for &candidate in &frontier {
                    let c = candidate as usize;
                    if taken[c] || rejected_by[c] == meshlet_number {
                        continue;
                    }
                    let new = builder.new_vertices(faces[c]);
                    if builder.members.len() + new > limits.vertices {
                        // Members only grow: it does not fit this meshlet any more.
                        rejected_by[c] = meshlet_number;
                        continue;
                    }
                    // How well it faces the way the meshlet does (1: the same).
                    let facing = match normals[c] {
                        Some(normal) if axis_length > 0.0 => {
                            Vec3::dot(&normal_sum, &normal) / axis_length
                        }
                        _ => 1.0,
                    };
                    if facing < builder.threshold {
                        // The average moves as triangles are added, so this is not forever, but
                        // it is for the exact check below to say.
                        continue;
                    }
                    let distance = {
                        let d = centroids[c] - centre;
                        Vec3::dot(&d, &d)
                    };
                    // A triangle that brings nothing is free. Otherwise: the nearest, with a
                    // penalty for new vertices (a strip grows long and thin, which is a big
                    // sphere) and for facing another way.
                    let score = if new == 0 {
                        -1.0
                    } else {
                        #[allow(clippy::cast_precision_loss)] // 0 to 3
                        let new = new as f32;
                        distance * 0.5f32.mul_add(new, 1.0) * (2.0 - facing)
                    };
                    if best.is_none_or(|(best_score, _)| score < best_score) {
                        best = Some((score, candidate));
                    }
                }
                let Some((_, candidate)) = best else {
                    break;
                };
                let c = candidate as usize;
                if builder.fits(faces[c], normals[c]) {
                    next = Some(candidate);
                    break;
                }
                rejected_by[c] = meshlet_number;
            }
        }
        builder.close();
    }

    builder.out
}

/// Which triangles touch which: triangles that have a vertex in the same place are neighbours,
/// whatever their indices say (a vertex that the texture has a seam at is two vertices).
struct Neighbours {
    /// For each place a vertex is at (the same place is the same number), where its triangles
    /// start in `triangles` (and the start of the next place is where they end).
    starts: Vec<u32>,
    /// The triangles of every place.
    triangles: Vec<u32>,
    /// For each vertex of the mesh, its place.
    place: Vec<u32>,
}

impl Neighbours {
    fn new<V: Vertex>(faces: &[[u32; 3]], vertices: &[V]) -> Self {
        // Vertices at the same position are at the same place. (-0.0 is 0.0: they are added to
        // 0.0 to look the same.)
        let mut places = std::collections::HashMap::with_capacity_and_hasher(
            vertices.len(),
            rustc_hash::FxBuildHasher,
        );
        let place: Vec<u32> = vertices
            .iter()
            .map(|vertex| {
                let (x, y, z) = vertex.position();
                let key = [
                    (x + 0.0).to_bits(),
                    (y + 0.0).to_bits(),
                    (z + 0.0).to_bits(),
                ];
                let next = u32::try_from(places.len()).expect("the numbers of a mesh fit 32 bits");
                *places.entry(key).or_insert(next)
            })
            .collect();

        // How many triangles each place has (once for a triangle that has it twice), then the
        // starts, then the triangles.
        let mut starts = vec![0_u32; places.len() + 1];
        for face in faces {
            for corner in Self::distinct(*face, &place) {
                starts[corner as usize + 1] += 1;
            }
        }
        for at in 1..starts.len() {
            starts[at] += starts[at - 1];
        }
        let mut fill = starts.clone();
        let mut triangles = vec![0_u32; starts[places.len()] as usize];
        for (triangle, face) in faces.iter().enumerate() {
            let triangle = u32::try_from(triangle).expect("the numbers of a mesh fit 32 bits");
            for corner in Self::distinct(*face, &place) {
                triangles[fill[corner as usize] as usize] = triangle;
                fill[corner as usize] += 1;
            }
        }

        Self {
            starts,
            triangles,
            place,
        }
    }

    /// The places of the corners of a triangle, one time each.
    fn distinct(face: [u32; 3], place: &[u32]) -> impl Iterator<Item = u32> {
        let places = face.map(|corner| place[corner as usize]);
        (0..3)
            .filter(move |&at| !places[..at].contains(&places[at]))
            .map(move |at| places[at])
    }

    /// The triangles that touch a triangle (it is one of them, and a triangle that touches at
    /// more than one place is more than once).
    fn of(&self, face: [u32; 3]) -> impl Iterator<Item = u32> {
        Self::distinct(face, &self.place).flat_map(move |place| {
            let range =
                self.starts[place as usize] as usize..self.starts[place as usize + 1] as usize;
            self.triangles[range].iter().copied()
        })
    }

    /// How many triangles that are not taken touch the triangle `triangle`.
    fn live(&self, face: [u32; 3], taken: &[bool], triangle: u32) -> usize {
        self.of(face)
            .filter(|&other| other != triangle && !taken[other as usize])
            .count()
    }
}

/// The meshlet being made, and what is needed to make it.
struct Builder<'a, V> {
    vertices: &'a [V],
    limits: Limits,
    threshold: f32,
    /// For every vertex of the mesh, its number in the meshlet being made, or `NOT_IN`. Only the
    /// vertices in `members` are anything else: they are put back when the meshlet is closed (so
    /// that a mesh of a million vertices does not mean a million writes for every meshlet).
    local: Vec<u8>,
    /// The vertices of the meshlet being made, as indices into the mesh.
    members: Vec<u32>,
    /// Its triangles, as indices into `members`: 3 for each.
    triangles: Vec<u8>,
    /// The directions its triangles face (not the ones with no area).
    normals: Vec<Vec3>,
    out: Meshlets,
}

impl<V: Vertex> Builder<'_, V> {
    fn position(&self, index: u32) -> Vec3 {
        Vec3::from(self.vertices[index as usize].position())
    }

    /// The direction the triangle faces (length 1), or none if it has no area.
    fn face_normal(&self, [a, b, c]: [u32; 3]) -> Option<Vec3> {
        let a = self.position(a);
        let normal = Vec3::cross(&(self.position(b) - a), &(self.position(c) - a));
        let length = normal.length();
        (length > 0.0 && length.is_finite()).then(|| normal.normalized())
    }

    /// How many vertices the triangle brings that the meshlet does not have (one that is twice in
    /// the triangle is counted once).
    fn new_vertices(&self, corners: [u32; 3]) -> usize {
        let mut new = 0;
        for (at, &corner) in corners.iter().enumerate() {
            if self.local[corner as usize] == NOT_IN && !corners[..at].contains(&corner) {
                new += 1;
            }
        }
        new
    }

    /// Whether the triangle can be added: there is room for it, and the directions the meshlet
    /// faces do not get too wide.
    fn fits(&self, corners: [u32; 3], normal: Option<Vec3>) -> bool {
        self.triangles.len() / 3 < self.limits.triangles
            && self.members.len() + self.new_vertices(corners) <= self.limits.vertices
            && normal.is_none_or(|normal| fits_cone(&self.normals, normal, self.threshold))
    }

    /// Adds a triangle to the meshlet being made: [`fits`](Self::fits) says whether it can.
    fn push(&mut self, corners: [u32; 3], normal: Option<Vec3>) {
        for corner in corners {
            let mut local = self.local[corner as usize];
            if local == NOT_IN {
                // At most the limit of the format (64), so it fits.
                local = u8::try_from(self.members.len()).expect("at most 64 vertices");
                self.local[corner as usize] = local;
                self.members.push(corner);
            }
            self.triangles.push(local);
        }
        if let Some(normal) = normal {
            self.normals.push(normal);
        }
    }

    /// Finishes the meshlet being made (if there is one), and starts an empty one.
    fn close(&mut self) {
        if self.triangles.is_empty() {
            return;
        }

        let axis = average_direction(self.normals.iter());
        let cutoff = cone_cutoff(smallest_dot(&axis, self.normals.iter()));
        let bounding = build_bounding_sphere(
            self.members
                .iter()
                .map(|&member| self.vertices[member as usize].position()),
        );

        let to_u32 =
            |value: usize| u32::try_from(value).expect("the numbers of a mesh fit 32 bits");
        self.out.records.push(MeshletRecord {
            bounding,
            cone_axis: (axis.x, axis.y, axis.z),
            cone_cutoff: cutoff,
            vertex_start: to_u32(self.out.vertices.len()),
            vertex_count: to_u32(self.members.len()),
            triangle_start: to_u32(self.out.indices.len() / 3),
            triangle_count: to_u32(self.triangles.len() / 3),
        });

        for &local in &self.triangles {
            self.out.indices.push(self.members[usize::from(local)]);
        }
        self.out.vertices.extend_from_slice(&self.members);
        self.out.triangles.extend_from_slice(&self.triangles);

        for &member in &self.members {
            self.local[member as usize] = NOT_IN;
        }
        self.members.clear();
        self.triangles.clear();
        self.normals.clear();
    }
}

#[cfg(test)]
#[allow(
    clippy::suboptimal_flops,
    clippy::chunks_exact_to_as_chunks,
    clippy::assert_is_empty,
    clippy::tuple_array_conversions,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "plain test code: clearer than the fused forms"
)]
mod flat_tests {
    use super::*;

    /// `n` by `n` squares in the xz plane, two triangles each, counter-clockwise seen from above
    /// (their normals are +y).
    fn grid(n: u32) -> (Vec<u32>, Vec<[f32; 3]>) {
        let mut positions = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                positions.push([x as f32, 0.0, z as f32]);
            }
        }
        let at = |x: u32, z: u32| z * (n + 1) + x;
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let (a, b, c, d) = (at(x, z), at(x, z + 1), at(x + 1, z), at(x + 1, z + 1));
                indices.extend_from_slice(&[a, b, c, c, b, d]);
            }
        }
        (indices, positions)
    }

    /// A sphere of radius 1 made of rings, counter-clockwise seen from outside.
    fn sphere(rings: u32, segments: u32) -> (Vec<u32>, Vec<[f32; 3]>) {
        let mut positions = Vec::new();
        for ring in 0..=rings {
            let polar = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..segments {
                let azimuth = std::f32::consts::TAU * segment as f32 / segments as f32;
                positions.push([
                    polar.sin() * azimuth.cos(),
                    polar.cos(),
                    polar.sin() * azimuth.sin(),
                ]);
            }
        }
        let at = |ring: u32, segment: u32| ring * segments + segment % segments;
        let mut indices = Vec::new();
        for ring in 0..rings {
            for segment in 0..segments {
                let (a, b, c, d) = (
                    at(ring, segment),
                    at(ring + 1, segment),
                    at(ring, segment + 1),
                    at(ring + 1, segment + 1),
                );
                // Seen from outside, a is above b, and c is the next one round (towards +z at the
                // front): a, b, c is counter-clockwise from where the normal points to.
                indices.extend_from_slice(&[a, c, b, c, d, b]);
            }
        }
        (indices, positions)
    }

    fn triangle_set(indices: &[u32]) -> Vec<[u32; 3]> {
        let mut triangles: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| {
                // The same triangle from any of its corners.
                let first = (0..3).min_by_key(|&k| t[k]).unwrap();
                [t[first], t[(first + 1) % 3], t[(first + 2) % 3]]
            })
            .collect();
        triangles.sort_unstable();
        triangles
    }

    /// Everything the format promises, for any mesh.
    fn check(indices: &[u32], positions: &[[f32; 3]], meshlets: &Meshlets) {
        // The same triangles, each once (the order and the corner they start at do not matter).
        assert_eq!(meshlets.indices.len(), indices.len());
        assert_eq!(triangle_set(&meshlets.indices), triangle_set(indices));

        let mut vertices = 0;
        let mut triangles = 0;
        for record in &meshlets.records {
            // The ranges are one after the other, and inside the limits.
            assert_eq!(record.vertex_start as usize, vertices);
            assert_eq!(record.triangle_start as usize, triangles);
            let (vertex_count, triangle_count) =
                (record.vertex_count as usize, record.triangle_count as usize);
            assert!((1..=MAX_VERTICES).contains(&vertex_count));
            assert!((1..=MAX_TRIANGLES).contains(&triangle_count));
            vertices += vertex_count;
            triangles += triangle_count;

            let members = &meshlets.vertices[record.vertex_start as usize..][..vertex_count];
            let local =
                &meshlets.triangles[3 * record.triangle_start as usize..][..3 * triangle_count];
            let expanded =
                &meshlets.indices[3 * record.triangle_start as usize..][..3 * triangle_count];

            // No vertex twice, and the local indices say the same as the expanded ones.
            let mut sorted = members.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), members.len());
            for (&local, &index) in local.iter().zip(expanded) {
                assert!(usize::from(local) < vertex_count);
                assert_eq!(members[usize::from(local)], index);
            }

            // The sphere has every vertex (a little slack for rounding).
            for &member in members {
                let p = positions[member as usize];
                let c = record.bounding.center;
                let distance =
                    ((p[0] - c.0).powi(2) + (p[1] - c.1).powi(2) + (p[2] - c.2).powi(2)).sqrt();
                assert!(distance <= record.bounding.radius * 1.0001 + 1e-5);
            }
        }
        assert_eq!(vertices, meshlets.vertices.len());
        assert_eq!(triangles * 3, meshlets.triangles.len());
    }

    #[test]
    fn smaller_limits_make_smaller_meshlets() {
        let (indices, positions) = grid(20);
        let limits = Limits {
            vertices: 16,
            triangles: 12,
        };
        let meshlets = build(&indices, &positions, limits, 0.5);
        check(&indices, &positions, &meshlets);

        for record in &meshlets.records {
            assert!(record.vertex_count as usize <= limits.vertices);
            assert!(record.triangle_count as usize <= limits.triangles);
        }
        // 800 triangles, 12 at most each; and more of them than with the most there is room for.
        assert!(meshlets.len() >= 67);
        assert!(meshlets.len() > build(&indices, &positions, Limits::MAX, 0.5).len());
    }

    #[test]
    fn limits_outside_the_format_are_refused() {
        let (indices, positions) = grid(2);
        for limits in [
            Limits {
                vertices: 2,
                triangles: 8,
            },
            Limits {
                vertices: 16,
                triangles: 0,
            },
            Limits {
                vertices: MAX_VERTICES + 1,
                triangles: 8,
            },
            Limits {
                vertices: 16,
                triangles: MAX_TRIANGLES + 1,
            },
        ] {
            let result = std::panic::catch_unwind(|| build(&indices, &positions, limits, 0.5));
            assert!(result.is_err(), "{limits:?}");
        }
    }

    #[test]
    fn nothing_makes_nothing() {
        let meshlets = build::<[f32; 3]>(&[], &[], Limits::MAX, 0.5);
        assert!(meshlets.is_empty());
        assert!(meshlets.indices.is_empty());
    }

    #[test]
    fn a_flat_grid_is_flat_meshlets_that_face_up() {
        let (indices, positions) = grid(20);
        let meshlets = build(&indices, &positions, Limits::MAX, 0.5);
        check(&indices, &positions, &meshlets);

        // 800 triangles, 124 at most each.
        assert!(meshlets.len() >= 7);
        for record in &meshlets.records {
            // The normal of (b - a) x (c - a) of the first triangle is +y: the convention.
            assert!((record.cone_axis.1 - 1.0).abs() < 1e-5, "{record:?}");
            assert!(record.cone_cutoff.abs() < 1e-3);
        }
    }

    #[test]
    fn a_sphere_is_cut_where_it_bends_too_much() {
        let (indices, positions) = sphere(24, 32);
        let threshold = 0.9;
        let meshlets = build(&indices, &positions, Limits::MAX, threshold);
        check(&indices, &positions, &meshlets);

        for record in &meshlets.records {
            let axis = Vec3::from(record.cone_axis);
            let triangles = &meshlets.indices[3 * record.triangle_start as usize..]
                [..3 * record.triangle_count as usize];
            for face in triangles.chunks_exact(3) {
                let p = |i: u32| {
                    Vec3::from((
                        positions[i as usize][0],
                        positions[i as usize][1],
                        positions[i as usize][2],
                    ))
                };
                let (a, b, c) = (p(face[0]), p(face[1]), p(face[2]));
                let normal = Vec3::cross(&(b - a), &(c - a));
                // (The ring at the south pole is a point only to within rounding: no area to speak of.)
                if normal.length() < 1e-4 {
                    continue;
                }
                let dot = Vec3::dot(&axis, &normal.normalized());
                // Every triangle is inside the cone the record says, and the cone is what the
                // threshold asked for.
                assert!(dot >= threshold - 1e-4, "{dot}");
                assert!(
                    (1.0 - dot * dot).max(0.0).sqrt() <= record.cone_cutoff + 1e-4,
                    "{dot} {}",
                    record.cone_cutoff
                );
                // A sphere faces outward: away from the middle of it, which is the origin.
                let centre = Vec3::new(
                    (a.x + b.x + c.x) / 3.0,
                    (a.y + b.y + c.y) / 3.0,
                    (a.z + b.z + c.z) / 3.0,
                );
                assert!(Vec3::dot(&centre, &normal) > 0.0, "inside out");
            }
        }
    }

    /// The triangles in an order that has nothing to do with where they are (the same every time).
    fn shuffled(indices: &[u32]) -> Vec<u32> {
        let mut triangles: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        let mut state = 0x2545_f491_u32;
        for at in (1..triangles.len()).rev() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            triangles.swap(at, (state >> 8) as usize % (at + 1));
        }
        triangles.concat()
    }

    #[test]
    fn meshlets_are_patches_not_strips_whatever_the_order() {
        // 3200 triangles: a patch of about 100 of them has a radius of about 5; the rows of the
        // grid that come one after another would be 40 units long, a radius of 20.
        let (indices, positions) = grid(40);
        for order in [indices.clone(), shuffled(&indices)] {
            let meshlets = build(&order, &positions, Limits::MAX, 0.5);
            check(&order, &positions, &meshlets);
            for record in &meshlets.records {
                assert!(record.bounding.radius < 10.0, "{}", record.bounding.radius);
            }
            // 3200 / 124 is 26 if they all were full: they are not, but not many times that.
            assert!(meshlets.len() < 80, "{}", meshlets.len());
        }
    }

    #[test]
    fn vertices_that_a_seam_doubles_are_still_neighbours() {
        // Two flat patches that meet along a line, with the vertices of the line twice (as where a
        // texture has a seam): the patches touch, so they are one meshlet (32 triangles, 30
        // vertices); if the seam kept them apart they would be two.
        let (mut indices, mut positions) = grid(4);
        let mut copy_of: Vec<Option<u32>> = vec![None; positions.len()];
        for triangle in indices.chunks_exact_mut(3) {
            // The triangles on the far side of the line use copies of the vertices.
            if triangle.iter().all(|&i| positions[i as usize][0] >= 2.0) {
                for index in triangle {
                    let original = *index as usize;
                    *index = *copy_of[original].get_or_insert_with(|| {
                        positions.push(positions[original]);
                        u32::try_from(positions.len() - 1).unwrap()
                    });
                }
            }
        }
        assert!(positions.len() > 25, "the vertices were doubled");
        let meshlets = build(&indices, &positions, Limits::MAX, 0.5);
        check(&indices, &positions, &meshlets);
        assert_eq!(meshlets.len(), 1);
    }

    #[test]
    fn a_triangle_with_no_area_does_not_split_a_meshlet() {
        let (mut indices, positions) = grid(4);
        let without = build(&indices, &positions, Limits::MAX, 0.5);
        // Three corners that are one vertex, in the middle of the list.
        let middle = indices.len() / 2 / 3 * 3;
        indices.splice(middle..middle, [5, 5, 5]);
        let with = build(&indices, &positions, Limits::MAX, 0.5);
        check(&indices, &positions, &with);
        assert_eq!(with.len(), without.len());
    }

    #[test]
    fn a_triangle_with_a_vertex_twice_counts_it_once() {
        // A strip where a corner is repeated: [0, 0, 1] has two vertices, not three.
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let indices = [0, 0, 1, 0, 1, 2];
        let meshlets = build(&indices, &positions, Limits::MAX, 0.5);
        check(&indices, &positions, &meshlets);
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets.records[0].vertex_count, 3);
    }

    #[test]
    fn the_cone_threshold_is_kept_in_its_range() {
        let (indices, positions) = sphere(8, 8);
        // Out of range values are made to fit, not refused.
        for threshold in [-1.0, 0.0, 5.0, f32::NAN] {
            let meshlets = build(&indices, &positions, Limits::MAX, threshold);
            check(&indices, &positions, &meshlets);
        }
    }
}
