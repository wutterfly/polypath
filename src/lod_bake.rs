//! Baking levels of detail: making a chain once, and reading it again instead of making it.
//!
//! Making the levels of a mesh takes a while (see [`lod::build`]: a mesh of 100,000 triangles takes
//! a second or two), and what comes out depends on nothing but the mesh and the settings. So a
//! chain can be made ahead of time and kept: [`content_key`] says which chain is the one for a mesh
//! (a number made of its positions, its triangles and everything the chain is made with),
//! [`levels_to_bytes`] puts it in a file, and [`levels_from_bytes`] reads it back, and gives
//! `None` for a file that is not the chain asked for (it is of another mesh, or other settings,
//! or an older version of the algorithm, or it is cut short or damaged).
//!
//! What is kept of a level is its error and its meshlets: the meshlets have the triangles of the
//! level (as `indices`, in the order of the meshlets), and the index list of the level that comes
//! back is those.

use std::hash::Hasher;

use rustc_hash::FxHasher;

use crate::{
    Vertex,
    bounding::Sphere,
    lod::{Level, Settings},
    meshlet::{Limits, MeshletRecord, Meshlets},
};

/// The version of the algorithm that makes the levels and of the format of the file: a chain that
/// was baked with another is not read.
pub const VERSION: u32 = 1;

/// The first four bytes of a file of levels.
const MAGIC: [u8; 4] = *b"PLOD";

/// What is made with a chain beside the mesh: the settings, and what the meshlets of every level
/// are made with.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub settings: Settings,
    pub limits: Limits,
    pub cone_threshold: f32,
    pub merge_apart: bool,
}

/// The number that says which chain is the one for this mesh and recipe.
///
/// It is made of the indices, the positions of the vertices, the recipe and [`VERSION`]. Two chains
/// have the same one only if they were made from the same mesh the same way.
#[must_use]
pub fn content_key<V: Vertex>(indices: &[u32], vertices: &[V], recipe: &Recipe) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write_u32(VERSION);
    let settings = &recipe.settings;
    hasher.write_u32(settings.reduction.to_bits());
    hasher.write_usize(settings.levels);
    hasher.write_usize(settings.min_triangles);
    hasher.write_u32(settings.border_weight.to_bits());
    hasher.write_u32(settings.seam_weight.to_bits());
    hasher.write_usize(recipe.limits.vertices);
    hasher.write_usize(recipe.limits.triangles);
    hasher.write_u32(recipe.cone_threshold.to_bits());
    hasher.write_u8(u8::from(recipe.merge_apart));
    hasher.write_usize(indices.len());
    for &index in indices {
        hasher.write_u32(index);
    }
    hasher.write_usize(vertices.len());
    for vertex in vertices {
        let (x, y, z) = vertex.position();
        hasher.write_u32(x.to_bits());
        hasher.write_u32(y.to_bits());
        hasher.write_u32(z.to_bits());
    }
    hasher.finish()
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_len(bytes: &mut Vec<u8>, length: usize) {
    put_u32(bytes, u32::try_from(length).expect("a list of a mesh fits 32 bits"));
}

/// The levels after the first (the mesh itself is not kept) as a file, for [`content_key`] `key`.
///
/// # Panics
/// If a list is longer than `u32::MAX`.
#[must_use]
pub fn levels_to_bytes(key: u64, levels: &[Level]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    put_u32(&mut bytes, VERSION);
    bytes.extend_from_slice(&key.to_le_bytes());
    put_len(&mut bytes, levels.len());
    for level in levels {
        put_f32(&mut bytes, level.error);
        let meshlets = &level.meshlets;
        put_len(&mut bytes, meshlets.records.len());
        for record in &meshlets.records {
            let (x, y, z) = record.bounding.center;
            for value in [x, y, z, record.bounding.radius] {
                put_f32(&mut bytes, value);
            }
            let (x, y, z) = record.cone_axis;
            for value in [x, y, z, record.cone_cutoff] {
                put_f32(&mut bytes, value);
            }
            for value in [
                record.vertex_start,
                record.vertex_count,
                record.triangle_start,
                record.triangle_count,
            ] {
                put_u32(&mut bytes, value);
            }
        }
        put_len(&mut bytes, meshlets.vertices.len());
        for &vertex in &meshlets.vertices {
            put_u32(&mut bytes, vertex);
        }
        put_len(&mut bytes, meshlets.triangles.len());
        bytes.extend_from_slice(&meshlets.triangles);
        put_len(&mut bytes, meshlets.indices.len());
        for &index in &meshlets.indices {
            put_u32(&mut bytes, index);
        }
    }
    bytes
}

/// A reader of the bytes of a file: every read says whether there was enough.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.bytes.split_at_checked(count)?;
        self.bytes = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&mut self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }

    /// A length that the rest of the file can have room for items of `size` bytes of (so a damaged
    /// number does not ask for memory that is not there).
    fn length(&mut self, size: usize) -> Option<usize> {
        let length = usize::try_from(self.u32()?).ok()?;
        (length.checked_mul(size)? <= self.bytes.len()).then_some(length)
    }
}

/// The levels in `bytes` if it is a file of levels for `key` (see [`levels_to_bytes`]), else `None`.
#[must_use]
pub fn levels_from_bytes(bytes: &[u8], key: u64) -> Option<Vec<Level>> {
    let mut reader = Reader { bytes };
    if reader.take(4)? != MAGIC || reader.u32()? != VERSION {
        return None;
    }
    let found = reader.take(8)?;
    if u64::from_le_bytes(found.try_into().ok()?) != key {
        return None;
    }
    let count = reader.length(4)?;
    let mut levels = Vec::with_capacity(count);
    for _ in 0..count {
        let error = reader.f32()?;
        let record_count = reader.length(48)?;
        let mut records = Vec::with_capacity(record_count);
        for _ in 0..record_count {
            let center = (reader.f32()?, reader.f32()?, reader.f32()?);
            let radius = reader.f32()?;
            let cone_axis = (reader.f32()?, reader.f32()?, reader.f32()?);
            let cone_cutoff = reader.f32()?;
            records.push(MeshletRecord {
                bounding: Sphere { center, radius },
                cone_axis,
                cone_cutoff,
                vertex_start: reader.u32()?,
                vertex_count: reader.u32()?,
                triangle_start: reader.u32()?,
                triangle_count: reader.u32()?,
            });
        }
        let vertex_count = reader.length(4)?;
        let vertices = (0..vertex_count)
            .map(|_| reader.u32())
            .collect::<Option<Vec<u32>>>()?;
        let triangle_bytes = reader.length(1)?;
        let triangles = reader.take(triangle_bytes)?.to_vec();
        let index_count = reader.length(4)?;
        let indices = (0..index_count)
            .map(|_| reader.u32())
            .collect::<Option<Vec<u32>>>()?;
        let meshlets = Meshlets {
            records,
            vertices,
            triangles,
            indices: indices.clone(),
        };
        levels.push(Level {
            indices,
            error,
            meshlets,
        });
    }
    // Nothing is left over: the file is what it says.
    reader.bytes.is_empty().then_some(levels)
}

/// The levels after the first of a mesh (the mesh itself is not one of them).
///
/// They are read from `file` if it is the chain for this mesh and recipe, and made otherwise (and
/// `file` is then the bytes of them, for the caller to keep). Returns the levels, and whether they
/// were made now (`true`) or read (`false`).
#[must_use]
pub fn baked_levels<V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    recipe: &Recipe,
    file: &mut Option<Vec<u8>>,
) -> (Vec<Level>, bool) {
    let key = content_key(indices, vertices, recipe);
    if let Some(bytes) = file.as_deref()
        && let Some(levels) = levels_from_bytes(bytes, key)
    {
        return (levels, false);
    }
    let levels: Vec<Level> = crate::lod::build(
        indices,
        vertices,
        &recipe.settings,
        (recipe.limits, recipe.cone_threshold, recipe.merge_apart),
    )
    .into_iter()
    .skip(1)
    .collect();
    *file = Some(levels_to_bytes(key, &levels));
    (levels, true)
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "a test makes a sphere from a few hundred numbers"
)]
mod tests {
    use super::*;

    fn sphere() -> (Vec<[f32; 3]>, Vec<u32>) {
        let (rings, segments) = (16_u32, 24_u32);
        let mut positions = vec![[0.0, 1.0, 0.0]];
        for ring in 1..rings {
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
        positions.push([0.0, -1.0, 0.0]);
        let south = u32::try_from(positions.len() - 1).unwrap();
        let start = |ring: u32| 1 + (ring - 1) * segments;
        let mut indices = Vec::new();
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            indices.extend([0, start(1) + next, start(1) + segment]);
        }
        for ring in 1..rings - 1 {
            for segment in 0..segments {
                let next = (segment + 1) % segments;
                let (a, b) = (start(ring) + segment, start(ring) + next);
                let (c, d) = (start(ring + 1) + segment, start(ring + 1) + next);
                indices.extend([a, b, c, b, d, c]);
            }
        }
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            indices.extend([south, start(rings - 1) + segment, start(rings - 1) + next]);
        }
        (positions, indices)
    }

    fn recipe() -> Recipe {
        Recipe {
            settings: Settings {
                levels: 3,
                min_triangles: 8,
                ..Settings::default()
            },
            limits: Limits::MAX,
            cone_threshold: 0.5,
            merge_apart: false,
        }
    }

    #[test]
    fn baked_levels_come_back_as_they_went() {
        let (positions, indices) = sphere();
        let key = content_key(&indices, &positions, &recipe());
        let made: Vec<Level> = crate::lod::build(
            &indices,
            &positions,
            &recipe().settings,
            (Limits::MAX, 0.5, false),
        )
        .into_iter()
        .skip(1)
        .collect();
        assert!(made.len() >= 2);

        let bytes = levels_to_bytes(key, &made);
        let read = levels_from_bytes(&bytes, key).expect("the file is the chain");

        assert_eq!(read.len(), made.len());
        for (read, made) in read.iter().zip(&made) {
            assert_eq!(read.error, made.error);
            assert_eq!(read.meshlets.vertices, made.meshlets.vertices);
            assert_eq!(read.meshlets.triangles, made.meshlets.triangles);
            assert_eq!(read.meshlets.indices, made.meshlets.indices);
            assert_eq!(read.meshlets.records.len(), made.meshlets.records.len());
            for (a, b) in read.meshlets.records.iter().zip(&made.meshlets.records) {
                assert_eq!(a.bounding.center, b.bounding.center);
                assert_eq!(a.cone_axis, b.cone_axis);
                assert_eq!(a.triangle_count, b.triangle_count);
            }
            // The index list is the triangles of the meshlets, in their order.
            assert_eq!(read.indices, made.meshlets.indices);
        }
    }

    #[test]
    fn a_file_of_another_mesh_or_recipe_or_a_damaged_one_is_not_read() {
        let (positions, indices) = sphere();
        let key = content_key(&indices, &positions, &recipe());
        let bytes = levels_to_bytes(key, &[]);
        assert_eq!(levels_from_bytes(&bytes, key).map(|levels| levels.len()), Some(0));

        // Another mesh: a vertex moved, an index changed.
        let mut moved = positions.clone();
        moved[3][0] += 0.001;
        assert_ne!(content_key(&indices, &moved, &recipe()), key);
        let mut turned = indices.clone();
        turned.swap(0, 1);
        assert_ne!(content_key(&turned, &positions, &recipe()), key);
        // Another recipe.
        let mut other = recipe();
        other.settings.reduction = 0.4;
        assert_ne!(content_key(&indices, &positions, &other), key);
        other = recipe();
        other.settings.seam_weight = 5.0;
        assert_ne!(content_key(&indices, &positions, &other), key);
        assert!(levels_from_bytes(&bytes, key ^ 1).is_none());

        // Damaged: cut short at every length, the magic changed, a byte added.
        let (positions, indices) = sphere();
        let made: Vec<Level> = crate::lod::build(
            &indices,
            &positions,
            &recipe().settings,
            (Limits::MAX, 0.5, false),
        )
        .into_iter()
        .skip(1)
        .collect();
        let bytes = levels_to_bytes(key, &made);
        for cut in (0..bytes.len()).step_by(97) {
            assert!(levels_from_bytes(&bytes[..cut], key).is_none(), "cut at {cut}");
        }
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert!(levels_from_bytes(&wrong, key).is_none());
        let mut longer = bytes;
        longer.push(0);
        assert!(levels_from_bytes(&longer, key).is_none());
    }

    #[test]
    fn a_chain_is_made_once_and_read_after() {
        let (positions, indices) = sphere();
        let mut file = None;

        let (first, made) = baked_levels(&indices, &positions, &recipe(), &mut file);
        assert!(made);
        assert!(file.is_some());
        let (second, made_again) = baked_levels(&indices, &positions, &recipe(), &mut file);

        assert!(!made_again);
        assert_eq!(first.len(), second.len());
        // A file of another mesh is made again.
        let mut other = recipe();
        other.cone_threshold = 0.6;
        let (_, made_other) = baked_levels(&indices, &positions, &other, &mut file);
        assert!(made_other);
    }
}
