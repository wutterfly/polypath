//! Levels of detail: the same mesh with fewer triangles, made from the mesh itself.
//!
//! A level reuses the vertices of the mesh as they are and has an index list of its own (and its
//! own [`Meshlets`]): triangles are merged by **collapsing an edge**, which moves one end of it
//! onto the other, so that every vertex of a level is a vertex of the mesh, with its own normal and
//! texture coordinates, and nothing is written but indices. [`build`] makes a whole chain of
//! levels, each with about half the triangles of the one before, in one pass of collapses.
//!
//! Which edge goes next is decided by the **quadric error metric** (Garland and Heckbert, 1997):
//! every vertex remembers the planes of the triangles that touch it, and moving it onto a
//! neighbour costs the sum of the squared distances of the new place to those planes. The cheapest
//! collapse is always done next, so what is flat goes first and what is the shape of the thing stays.
//!
//! # Seams
//!
//! A mesh that has a texture seam or a hard edge has that vertex more than once, one for each
//! side, all at the same place. The collapses are decided for **places** (all the vertices with
//! the same position are one), and a place is only collapsed onto another if each of its vertices
//! has an edge to a vertex of the other: the sides that meet there are the sides that meet at the
//! other end, so what is a seam stays a seam and no crack opens. The triangles of each side are
//! then rewritten to use their own vertex of the place they go to. A corner of a cube (three
//! sides) is not collapsed onto a place that has two of them: the cube stays a cube.
//!
//! # Shading
//!
//! A level has the vertices of the mesh, so a triangle of it that is bigger than the ones it
//! replaced interpolates the normals and texture coordinates of its corners over a longer way.
//! Where those change fast (a rounded body of many thin triangles, a bevel, a pattern) that is a
//! different picture even if the surface is where it was. When the vertices have normals and
//! texture coordinates ([`Vertex::normal`], [`Vertex::texture_coord`]), they are part of the
//! error: every vertex remembers, for each of them, how it varied over the triangles that were
//! merged into it (the way the planes are remembered: a linear function of the position for each
//! triangle, summed), and a collapse costs how far the vertex it goes to is from those.
//! [`Settings::normal_weight`] and [`Settings::uv_weight`] say how much that is in the units of
//! the positions, as a part of the size of the mesh. The error of a level ([`Level::error`])
//! includes it, so a renderer that shows a level when its error is under a pixel shows it when the
//! shading is off by under about that too.
//!
//! # What is kept
//!
//! - The open border of a mesh (edges with one triangle): the planes that hold it are weighted
//!   ([`Settings::border_weight`]), and a place on a border is only collapsed along it. With
//!   [`Settings::lock_borders`] those places cannot move, so separately simplified parts keep
//!   their original shared outline.
//! - The orientation: a collapse that turns a triangle over (its normal moves more than
//!   [`MAX_TURN`] from where it was) is not done, and neither is one that makes the surface touch
//!   itself at a point or an edge.
//!
//! Triangles with no area (two corners in one place) and triangles with a position that is not a
//! number are not in any level above the first.

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "the metric is a sum of squares of distances that is only compared, and counts of               triangles are far below 2^52: the last bit of either does not matter"
)]

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use rustc_hash::FxBuildHasher;

use crate::{
    Vertex,
    meshlet::{self, Limits, Meshlets},
};

/// How far a triangle may turn in a collapse: the dot product of the normal before and after, as
/// at least this (0.2 is about 78 degrees).
pub const MAX_TURN: f64 = 0.2;

/// How a chain of levels is made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// What part of the triangles of a level the next one keeps: 0.5 is half. From 0.05 to 0.95.
    pub reduction: f32,
    /// How many levels there are after the first one (the mesh as it is) at most.
    pub levels: usize,
    /// No level has fewer triangles than this: the chain ends there.
    pub min_triangles: usize,
    /// How much it matters that the open border of the mesh stays where it is, against any one
    /// plane of a triangle (which is 1). A hundred strongly resists movement; only
    /// `lock_borders` guarantees the border stays fixed.
    pub border_weight: f32,
    /// Keep every vertex on an open border at its original position. Use this when adjacent
    /// parts are simplified separately: a weighted border alone can move and expose cracks.
    /// Interior vertices may still collapse onto a locked border. False by default, so a
    /// standalone open surface can simplify its outline with `border_weight` instead.
    pub lock_borders: bool,
    /// How much a change of the normal counts in the error, as a part of the size of the mesh (the
    /// diagonal of its box) for a normal that is turned by 1 (about 60 degrees): 0.1 means that
    /// 10 degrees is as much as 1.7 percent of the size. 0 leaves the normals out. Used when the
    /// vertices have normals.
    pub normal_weight: f32,
    /// The same for the texture coordinates: a change of 1 (the whole texture) counts as this
    /// part of the size of the mesh. 0 leaves them out. Used when the vertices have them.
    pub uv_weight: f32,
    /// What a side of a seam costs that has no edge to the place it is collapsed onto, in squared
    /// distances of the collapse (a collapse of length `d` with one such side costs
    /// `seam_weight * d * d` more). The side then takes whichever vertex of that place is the
    /// nearest to it, and its triangles have the texture coordinates and normal of another side
    /// at that corner: the shape is right and the shading is a little off. 0 does not allow it
    /// (the strict rule: a mesh that is split into many small islands of texture then goes down
    /// little), and a few is the price of things that are not worth it unless the rest has gone.
    pub seam_weight: f32,
}

impl Default for Settings {
    /// Up to six levels of half the triangles each, none under 64 triangles, a border that stays.
    fn default() -> Self {
        Self {
            reduction: 0.5,
            levels: 6,
            min_triangles: 64,
            border_weight: 100.0,
            lock_borders: false,
            normal_weight: 0.1,
            uv_weight: 1.0,
            seam_weight: 1.0,
        }
    }
}

/// A vertex with a normal and texture coordinates besides its position (see "Shading" above).
///
/// Any [`Vertex`] that has them does as well; this is one for a caller that has the lists and not
/// a type of its own.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Corner {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texture_coord: [f32; 2],
}

impl Vertex for Corner {
    #[inline]
    fn position(&self) -> (f32, f32, f32) {
        <(f32, f32, f32)>::from(self.position)
    }

    #[inline]
    fn normal(&self) -> Option<(f32, f32, f32)> {
        Some(<(f32, f32, f32)>::from(self.normal))
    }

    #[inline]
    fn texture_coord(&self) -> Option<(f32, f32)> {
        Some(<(f32, f32)>::from(self.texture_coord))
    }
}

/// One level of detail: its indices (into the vertices of the mesh, which are all of the mesh),
/// how far it can be from the mesh, and its meshlets.
#[derive(Debug, Clone)]
pub struct Level {
    /// The triangles, 3 indices each, into the vertices of the mesh. The first level is the
    /// indices the mesh was given as they were; the others are some of its triangles, with the
    /// corners moved by the collapses, in the order of the triangles they come from.
    pub indices: Vec<u32>,
    /// How far this level is from the mesh at its worst, in the units of the positions: the root
    /// of the mean squared distance of a place that was moved to the planes of the triangles that
    /// it stood for, at the collapse that did the most of it so far (the planes are weighted by
    /// their area). It follows the distance, and is not a bound on it. 0 for the first level, and
    /// never less than the level before. A renderer shows a level when this, seen from where
    /// the camera is, is under a pixel or so. It includes what the normals and texture coordinates
    /// are off by (see "Shading" above), so it is what a view that shows the shading uses.
    pub error: f32,
    /// The same without the shading: how far the surface itself is. What a view that only needs the
    /// shape uses (a shadow). Never more than `error`.
    pub geometric_error: f32,
    /// The meshlets of `indices`: they index the same vertices.
    pub meshlets: Meshlets,
}

/// Makes the levels of detail of a mesh.
///
/// The first is the mesh, the others have fewer triangles (see [`Settings`]), up to as many as
/// the settings ask for and the mesh allows (a mesh that can not be made smaller without folding
/// itself, such as a cube, has the one level).
///
/// `limits`, `cone_threshold` and `merge_apart` are what [`meshlet::build`] is given for every
/// level.
///
/// # Panics
/// If `indices.len()` is not a multiple of 3, or an index is not a vertex of `vertices`, or what
/// [`meshlet::build`] panics for.
#[must_use]
pub fn build<V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    settings: &Settings,
    (limits, cone_threshold, merge_apart): (Limits, f32, bool),
) -> Vec<Level> {
    let chain = simplify_with_errors(indices, vertices, settings);
    chain
        .into_iter()
        .map(|(indices, error, geometric_error)| Level {
            meshlets: meshlet::build(&indices, vertices, limits, cone_threshold, merge_apart),
            indices,
            error,
            geometric_error,
        })
        .collect()
}

/// The index lists of the levels, and the error of each (see [`build`]), without the meshlets.
///
/// # Panics
/// If `indices.len()` is not a multiple of 3, or an index is not a vertex of `vertices`.
#[must_use]
pub fn simplify<V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    settings: &Settings,
) -> Vec<(Vec<u32>, f32)> {
    simplify_with_errors(indices, vertices, settings)
        .into_iter()
        .map(|(indices, error, _)| (indices, error))
        .collect()
}

/// The same with the error of each level without the shading as well (see
/// [`Level::geometric_error`]): the indices, the error, and the geometric error.
///
/// # Panics
/// As [`simplify`].
#[must_use]
pub fn simplify_with_errors<V: Vertex>(
    indices: &[u32],
    vertices: &[V],
    settings: &Settings,
) -> Vec<(Vec<u32>, f32, f32)> {
    assert_eq!(
        indices.len() % 3,
        0,
        "the indices are triangles: 3 numbers each"
    );
    assert!(
        indices.iter().all(|&index| (index as usize) < vertices.len()),
        "every index is a vertex"
    );
    let mut levels = vec![(indices.to_vec(), 0.0, 0.0)];

    let mut state = State::new(indices, vertices, settings);
    let reduction = f64::from(settings.reduction.clamp(0.05, 0.95));
    let first = state.alive_triangles;
    let mut wanted = first as f64;
    let mut before = first;
    for _ in 0..settings.levels {
        wanted *= reduction;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // a count, above 0
        let target = (wanted.ceil() as usize).max(settings.min_triangles);
        // A level that is not under what the one before has by a fifth is not worth a draw.
        if target >= before || (target as f64) > before as f64 * 0.95 {
            break;
        }
        state.collapse_until(target);
        if state.alive_triangles >= before || state.alive_triangles == 0 {
            break;
        }
        before = state.alive_triangles;
        levels.push((state.indices(), state.error(), state.geometric_error()));
        if before <= settings.min_triangles {
            break;
        }
    }
    levels
}

/// A symmetric 4 by 4 matrix, the upper triangle: what a place knows of the planes around it, and
/// the weight of them (the area they stand for). A point `p` is at the mean squared distance
/// `v^T Q v / weight` from them (with `v = (p, 1)`), which is a distance whatever how many planes
/// there are.
#[derive(Debug, Clone, Copy, Default)]
struct Quadric {
    terms: [f64; 10],
    weight: f64,
}

impl Quadric {
    /// The quadric of the plane `normal . p + d = 0` (the normal is a unit), with weight `weight`.
    fn plane(normal: [f64; 3], d: f64, weight: f64) -> Self {
        let [a, b, c] = normal;
        Self {
            terms: [
                weight * a * a,
                weight * a * b,
                weight * a * c,
                weight * a * d,
                weight * b * b,
                weight * b * c,
                weight * b * d,
                weight * c * c,
                weight * c * d,
                weight * d * d,
            ],
            weight,
        }
    }

    fn add(&mut self, other: &Self) {
        for (mine, theirs) in self.terms.iter_mut().zip(other.terms) {
            *mine += theirs;
        }
        self.weight += other.weight;
    }

    /// The mean squared distance of `point` to the planes (0 if there are none).
    fn at(&self, [x, y, z]: [f64; 3]) -> f64 {
        if self.weight <= 0.0 {
            return 0.0;
        }
        let q = &self.terms;
        let value = q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9];
        // A sum of squares: a hair under 0 is rounding.
        value.max(0.0) / self.weight
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// A candidate collapse in the queue: move place `from` onto place `to`.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    cost: f64,
    from: u32,
    to: u32,
    /// The versions of the two places when it was made: a place that changed since makes it stale.
    versions: (u32, u32),
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// The cheapest first (the heap is a max heap), and a tie by the places, so that the same
    /// mesh always gives the same levels.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.from.cmp(&self.from))
            .then_with(|| other.to.cmp(&self.to))
    }
}

/// What a collapse does, found out before it is done.
struct Plan {
    /// The triangles that go (they have an edge between the two places).
    removed: Vec<u32>,
    /// The triangles that stay, with the corner that moves and the vertex it is moved to.
    moved: Vec<(u32, usize, u32)>,
    /// How far the surface goes from what it was: the mean squared distance of the new place to
    /// the planes of the triangles (not the borders' planes), and what its normals and texture
    /// coordinates are off by.
    error: f64,
    /// The same without what the normals and texture coordinates are off by.
    geometric: f64,
    /// The vertices that are merged: each vertex of `from` and the vertex of `to` that takes it.
    merged: Vec<(u32, u32)>,
}

/// How many numbers a vertex has that are kept: the normal (3) and the texture coordinates (2).
const CHANNELS: usize = 5;

/// What one number of a vertex (a channel) was over the triangles that were merged into a vertex:
/// each triangle has it as a linear function of the position, `g . p + d`, and what is kept is the
/// sum of the squares of how far that is from a value `a` at the point `p`, as a quadric in `p`
/// and `a` (weighted by the area, like the planes): `p^T G p + 2 h . p + c - 2 a (s . p + t) +
/// a^2 w`.
#[derive(Debug, Clone, Copy, Default)]
struct ChannelTerms {
    /// `G`: the upper triangle of the sum of `g g^T`.
    gg: [f64; 6],
    /// `h`: the sum of `d g`.
    gd: [f64; 3],
    /// `c`: the sum of `d^2`.
    dd: f64,
    /// `s`: the sum of `g`.
    g1: [f64; 3],
    /// `t`: the sum of `d`.
    d1: f64,
}

/// The channels of one vertex, and the area they stand for.
#[derive(Debug, Clone, Copy, Default)]
struct VertexTerms {
    channels: [ChannelTerms; CHANNELS],
    weight: f64,
}

impl VertexTerms {
    fn add(&mut self, other: &Self) {
        self.weight += other.weight;
        for (mine, theirs) in self.channels.iter_mut().zip(&other.channels) {
            for (a, b) in mine.gg.iter_mut().zip(theirs.gg) {
                *a += b;
            }
            for (a, b) in mine.gd.iter_mut().zip(theirs.gd) {
                *a += b;
            }
            for (a, b) in mine.g1.iter_mut().zip(theirs.g1) {
                *a += b;
            }
            mine.dd += theirs.dd;
            mine.d1 += theirs.d1;
        }
    }
}

/// The normals and the texture coordinates of the vertices, and what is remembered of them.
struct Attributes {
    /// The numbers of each vertex of the mesh (the normal, then the texture coordinates).
    values: Vec<[f64; CHANNELS]>,
    /// What each vertex stands for.
    terms: Vec<VertexTerms>,
    /// Which channels are kept (the ones that are asked for, and that are not the same everywhere).
    active: [bool; CHANNELS],
    /// The square of what a unit of each channel is in the units of the positions.
    scale_squared: [f64; CHANNELS],
    /// What the positions are taken from, so that they are small numbers.
    origin: [f64; 3],
}

impl Attributes {
    /// `None` if the vertices have nothing to keep (or it is not asked for).
    fn new<V: Vertex>(vertices: &[V], places: &[[f64; 3]], settings: &Settings) -> Option<Self> {
        let normal_weight = f64::from(settings.normal_weight.max(0.0));
        let uv_weight = f64::from(settings.uv_weight.max(0.0));
        if normal_weight <= 0.0 && uv_weight <= 0.0 {
            return None;
        }
        let mut values = vec![[0.0; CHANNELS]; vertices.len()];
        let mut seen = [false; CHANNELS];
        let mut first = [0.0; CHANNELS];
        let mut varies = [false; CHANNELS];
        for (vertex, value) in vertices.iter().zip(&mut values) {
            let normal = vertex.normal().map(|n| <[f32; 3]>::from(n).map(f64::from));
            let uv = vertex.texture_coord().map(|t| <[f32; 2]>::from(t).map(f64::from));
            if let Some(normal) = normal {
                value[..3].copy_from_slice(&normal);
            }
            if let Some(uv) = uv {
                value[3] = uv[0];
                value[4] = uv[1];
            }
            for channel in 0..CHANNELS {
                let has = if channel < 3 { normal.is_some() } else { uv.is_some() };
                if !has || !value[channel].is_finite() {
                    continue;
                }
                if seen[channel] {
                    varies[channel] |= (value[channel] - first[channel]).abs() > 1.0e-9;
                } else {
                    seen[channel] = true;
                    first[channel] = value[channel];
                }
            }
        }
        let weights = [normal_weight, normal_weight, normal_weight, uv_weight, uv_weight];
        let active: [bool; CHANNELS] = std::array::from_fn(|channel| {
            varies[channel] && weights[channel] > 0.0
        });
        if !active.iter().any(|&on| on) {
            return None;
        }
        // The size of the mesh: the diagonal of the box around its places.
        let (mut low, mut high) = ([f64::MAX; 3], [f64::MIN; 3]);
        for place in places.iter().filter(|place| place.iter().all(|v| v.is_finite())) {
            for axis in 0..3 {
                low[axis] = low[axis].min(place[axis]);
                high[axis] = high[axis].max(place[axis]);
            }
        }
        if low[0] > high[0] {
            return None;
        }
        let extent = length(sub(high, low));
        let origin = std::array::from_fn(|axis| f64::midpoint(low[axis], high[axis]));
        let scale_squared =
            std::array::from_fn(|channel| (weights[channel] * extent).powi(2));
        Some(Self {
            values,
            terms: vec![VertexTerms::default(); vertices.len()],
            active,
            scale_squared,
            origin,
        })
    }

    /// Adds a triangle (its corners are vertices of the mesh) to what its three vertices remember.
    fn add_triangle(&mut self, corners: [u32; 3], positions: [[f64; 3]; 3], area: f64) {
        let [p0, p1, p2] = positions.map(|p| sub(p, self.origin));
        let (e1, e2) = (sub(p1, p0), sub(p2, p0));
        let normal = cross(e1, e2);
        let size = dot(normal, normal);
        if size <= 0.0 {
            return;
        }
        // The gradient of a number that is `a0`, `a1` and `a2` at the corners: `g . e1 = a1 - a0`,
        // `g . e2 = a2 - a0` and no change across the triangle.
        let (across1, across2) = (cross(e2, normal), cross(normal, e1));
        let mut triangle = VertexTerms {
            weight: area,
            ..VertexTerms::default()
        };
        for channel in 0..CHANNELS {
            if !self.active[channel] {
                continue;
            }
            let a = corners.map(|corner| self.values[corner as usize][channel]);
            let (da1, da2) = (a[1] - a[0], a[2] - a[0]);
            let g = [
                (da1 * across1[0] + da2 * across2[0]) / size,
                (da1 * across1[1] + da2 * across2[1]) / size,
                (da1 * across1[2] + da2 * across2[2]) / size,
            ];
            let d = a[0] - dot(g, p0);
            let terms = &mut triangle.channels[channel];
            terms.gg = [
                area * g[0] * g[0],
                area * g[0] * g[1],
                area * g[0] * g[2],
                area * g[1] * g[1],
                area * g[1] * g[2],
                area * g[2] * g[2],
            ];
            terms.gd = [area * d * g[0], area * d * g[1], area * d * g[2]];
            terms.dd = area * d * d;
            terms.g1 = [area * g[0], area * g[1], area * g[2]];
            terms.d1 = area * d;
        }
        for corner in corners {
            self.terms[corner as usize].add(&triangle);
        }
    }

    /// How far the vertices that `merged` goes to are, at `target`, from what the vertices that
    /// are merged into them were: the mean squared difference of their normals and texture
    /// coordinates in the units of the positions squared.
    fn error(&self, merged: &[(u32, u32)], target: [f64; 3]) -> f64 {
        let p = sub(target, self.origin);
        let (mut sum, mut weight) = (0.0, 0.0);
        for &(from, to) in merged {
            let mut terms = self.terms[from as usize];
            terms.add(&self.terms[to as usize]);
            weight += terms.weight;
            for channel in 0..CHANNELS {
                if !self.active[channel] {
                    continue;
                }
                let t = &terms.channels[channel];
                let a = self.values[to as usize][channel];
                let value = t.gg[0] * p[0] * p[0]
                    + 2.0 * t.gg[1] * p[0] * p[1]
                    + 2.0 * t.gg[2] * p[0] * p[2]
                    + t.gg[3] * p[1] * p[1]
                    + 2.0 * t.gg[4] * p[1] * p[2]
                    + t.gg[5] * p[2] * p[2]
                    + 2.0 * dot(t.gd, p)
                    + t.dd
                    - 2.0 * a * (dot(t.g1, p) + t.d1)
                    + a * a * terms.weight;
                // A sum of squares: a hair under 0 is rounding.
                sum += value.max(0.0) * self.scale_squared[channel];
            }
        }
        if weight > 0.0 { sum / weight } else { 0.0 }
    }

    /// The vertices that are merged: what each remembers goes to the one it goes to.
    fn merge(&mut self, merged: &[(u32, u32)]) {
        for &(from, to) in merged {
            let moved = self.terms[from as usize];
            self.terms[to as usize].add(&moved);
        }
    }
}

/// The mesh while it is made smaller.
struct State {
    /// The triangles as indices of the mesh's vertices, changed by the collapses.
    triangles: Vec<[u32; 3]>,
    alive: Vec<bool>,
    alive_triangles: usize,
    /// For each vertex of the mesh: the place (the number of its position) it is at.
    place_of: Vec<u32>,
    positions: Vec<[f64; 3]>,
    /// For each place: the triangles that have a corner in it (some are dead: that is looked at).
    faces: Vec<Vec<u32>>,
    /// What decides the order of the collapses: the planes of the triangles and the ones that hold
    /// the borders.
    quadrics: Vec<Quadric>,
    /// The planes of the triangles alone: what the error that is told is measured with.
    surfaces: Vec<Quadric>,
    version: Vec<u32>,
    gone: Vec<bool>,
    heap: BinaryHeap<Candidate>,
    /// The most a collapse has cost so far.
    worst: f64,
    /// The same of the geometric part alone.
    worst_geometric: f64,
    /// See [`Settings::seam_weight`].
    seam_weight: f64,
    /// Original open-border places that must not move, even when their neighbours change.
    locked: Vec<bool>,
    /// What the normals and texture coordinates were, if they are kept.
    attributes: Option<Attributes>,
}

impl State {
    fn new<V: Vertex>(indices: &[u32], vertices: &[V], settings: &Settings) -> Self {
        // The places: every position once (and `-0.0` is `0.0`).
        let mut ids: HashMap<[u32; 3], u32, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(vertices.len(), FxBuildHasher);
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let place_of: Vec<u32> = vertices
            .iter()
            .map(|vertex| {
                let (x, y, z) = vertex.position();
                let key = [(x + 0.0).to_bits(), (y + 0.0).to_bits(), (z + 0.0).to_bits()];
                *ids.entry(key).or_insert_with(|| {
                    positions.push([f64::from(x), f64::from(y), f64::from(z)]);
                    u32::try_from(positions.len() - 1).expect("the places fit 32 bits")
                })
            })
            .collect();

        let mut triangles: Vec<[u32; 3]> = indices.as_chunks::<3>().0.to_vec();
        let mut alive: Vec<bool> = triangles
            .iter()
            .map(|corners| {
                let places = corners.map(|index| place_of[index as usize] as usize);
                let finite = places
                    .iter()
                    .all(|&place| positions[place].iter().all(|value| value.is_finite()));
                finite
                    && places[0] != places[1]
                    && places[1] != places[2]
                    && places[0] != places[2]
            })
            .collect();
        // A triangle with no area at all (three places on a line) is as good as none.
        for (corners, alive) in triangles.iter().zip(&mut alive) {
            if *alive {
                let [a, b, c] = corners.map(|index| positions[place_of[index as usize] as usize]);
                if length(cross(sub(b, a), sub(c, a))) <= 0.0 {
                    *alive = false;
                }
            }
        }
        let alive_triangles = alive.iter().filter(|&&alive| alive).count();
        // The corners of a triangle that is not in the levels are never read again.
        for (corners, alive) in triangles.iter_mut().zip(&alive) {
            if !alive {
                *corners = [0; 3];
            }
        }

        let place_count = positions.len();
        let mut state = Self {
            triangles,
            alive,
            alive_triangles,
            place_of,
            positions,
            faces: vec![Vec::new(); place_count],
            quadrics: vec![Quadric::default(); place_count],
            surfaces: vec![Quadric::default(); place_count],
            version: vec![0; place_count],
            gone: vec![false; place_count],
            heap: BinaryHeap::new(),
            worst: 0.0,
            worst_geometric: 0.0,
            seam_weight: 0.0,
            locked: vec![false; place_count],
            attributes: None,
        };
        state.attributes = Attributes::new(vertices, &state.positions, settings);
        state.seam_weight = f64::from(settings.seam_weight.max(0.0));
        state.make_quadrics(f64::from(settings.border_weight.max(0.0)), settings.lock_borders);
        state.push_all();
        state
    }

    fn corner_place(&self, triangle: u32, corner: usize) -> u32 {
        self.place_of[self.triangles[triangle as usize][corner] as usize]
    }

    fn places(&self, triangle: u32) -> [u32; 3] {
        [
            self.corner_place(triangle, 0),
            self.corner_place(triangle, 1),
            self.corner_place(triangle, 2),
        ]
    }

    /// The plane of a triangle: a unit normal and the offset, or `None` with no area.
    fn plane_of(&self, [a, b, c]: [u32; 3]) -> Option<([f64; 3], f64)> {
        let (a, b, c) = (
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        );
        let normal = cross(sub(b, a), sub(c, a));
        let size = length(normal);
        if size <= 0.0 {
            return None;
        }
        let unit = [normal[0] / size, normal[1] / size, normal[2] / size];
        Some((unit, -dot(unit, a)))
    }

    /// The quadrics of the places: the plane of each triangle at each of its three, and the planes
    /// that hold an open border (an edge with one triangle) in place.
    fn make_quadrics(&mut self, border_weight: f64, lock_borders: bool) {
        let mut edge_triangles: HashMap<(u32, u32), (u32, u32), FxBuildHasher> =
            HashMap::with_hasher(FxBuildHasher);
        for triangle in 0..self.triangles.len() {
            if !self.alive[triangle] {
                continue;
            }
            let number = u32::try_from(triangle).expect("the triangles fit 32 bits");
            let places = self.places(number);
            for place in places {
                self.faces[place as usize].push(number);
            }
            if let Some((normal, d)) = self.plane_of(places) {
                // Weighted by the area, so that a plane is as much as the surface it stands for.
                let [a, b, c] = places.map(|place| self.positions[place as usize]);
                let area = length(cross(sub(b, a), sub(c, a))) / 2.0;
                let quadric = Quadric::plane(normal, d, area);
                for place in places {
                    self.quadrics[place as usize].add(&quadric);
                    self.surfaces[place as usize].add(&quadric);
                }
                if let Some(attributes) = &mut self.attributes {
                    attributes.add_triangle(self.triangles[triangle], [a, b, c], area);
                }
            }
            for edge in 0..3 {
                let (a, b) = (places[edge], places[(edge + 1) % 3]);
                let key = (a.min(b), a.max(b));
                let entry = edge_triangles.entry(key).or_insert((0, number));
                entry.0 += 1;
            }
        }
        // A border: the plane through the edge that is across the triangle holds it.
        for (&(a, b), &(count, triangle)) in &edge_triangles {
            if count != 1 {
                continue;
            }
            if lock_borders {
                self.locked[a as usize] = true;
                self.locked[b as usize] = true;
            }
            let Some((normal, _)) = self.plane_of(self.places(triangle)) else {
                continue;
            };
            let (pa, pb) = (self.positions[a as usize], self.positions[b as usize]);
            let across = cross(sub(pb, pa), normal);
            let size = length(across);
            if size <= 0.0 {
                continue;
            }
            let unit = [across[0] / size, across[1] / size, across[2] / size];
            // As much as a triangle that stands on the edge (the length squared) times the weight.
            let weight = border_weight * size * size;
            let quadric = Quadric::plane(unit, -dot(unit, pa), weight);
            self.quadrics[a as usize].add(&quadric);
            self.quadrics[b as usize].add(&quadric);
        }
    }

    /// The places that share a triangle with `place`, each once.
    fn neighbours(&self, place: u32) -> Vec<u32> {
        let mut found: Vec<u32> = Vec::new();
        for &triangle in &self.faces[place as usize] {
            if !self.alive[triangle as usize] {
                continue;
            }
            for other in self.places(triangle) {
                if other != place && !found.contains(&other) {
                    found.push(other);
                }
            }
        }
        found
    }

    /// The live triangles that have a corner in both places.
    fn shared(&self, a: u32, b: u32) -> Vec<u32> {
        self.faces[a as usize]
            .iter()
            .copied()
            .filter(|&triangle| {
                self.alive[triangle as usize] && self.places(triangle).contains(&b)
            })
            .collect()
    }

    /// Every edge, both ways, in the queue.
    fn push_all(&mut self) {
        self.heap.clear();
        for place in 0..self.positions.len() {
            if self.gone[place] {
                continue;
            }
            let place = u32::try_from(place).expect("the places fit 32 bits");
            for other in self.neighbours(place) {
                // Both ways: the edge is seen from each end.
                self.push(place, other);
            }
        }
    }

    /// The collapse of `from` onto `to`, in the queue, if it can be done.
    fn push(&mut self, from: u32, to: u32) {
        if let Some((_, cost)) = self.plan(from, to) {
            self.heap.push(Candidate {
                cost,
                from,
                to,
                versions: (self.version[from as usize], self.version[to as usize]),
            });
        }
    }

    /// For the side of `from` that `vertex` is (and that no triangle of the edge is on), the
    /// vertex of `to` that is the one to take: the one on a triangle that has a place in common
    /// with a triangle of the side (they are the nearest across the surface), or the first.
    fn nearest_side(&self, vertex: u32, from: u32, to: u32) -> Option<u32> {
        let mut around: Vec<u32> = Vec::new();
        for &triangle in &self.faces[from as usize] {
            if self.alive[triangle as usize] && self.triangles[triangle as usize].contains(&vertex) {
                for place in self.places(triangle) {
                    if place != from && place != to && !around.contains(&place) {
                        around.push(place);
                    }
                }
            }
        }
        let mut first: Option<u32> = None;
        for &triangle in &self.faces[to as usize] {
            if !self.alive[triangle as usize] {
                continue;
            }
            let corners = self.triangles[triangle as usize];
            let places = self.places(triangle);
            let Some(at) = places.iter().position(|&place| place == to) else {
                continue;
            };
            let candidate = corners[at];
            if places.iter().any(|place| around.contains(place)) {
                return Some(candidate);
            }
            first = Some(first.map_or(candidate, |best| best.min(candidate)));
        }
        first
    }

    /// Whether moving a corner of each of `rewrites` (triangle and corner) to `target` turns the
    /// triangle over by more than [`MAX_TURN`], or makes it into nothing.
    fn turns_over(&self, rewrites: &[(u32, usize, u32)], target: [f64; 3]) -> bool {
        rewrites.iter().any(|&(triangle, corner, _)| {
            let places = self.places(triangle);
            let before = places.map(|place| self.positions[place as usize]);
            let mut after = before;
            after[corner] = target;
            let normal_before = cross(sub(before[1], before[0]), sub(before[2], before[0]));
            let normal_after = cross(sub(after[1], after[0]), sub(after[2], after[0]));
            let (size_before, size_after) = (length(normal_before), length(normal_after));
            size_after <= size_before * 1.0e-6
                || dot(normal_before, normal_after) < MAX_TURN * size_before * size_after
        })
    }

    /// What collapsing `from` onto `to` does and what it costs, or `None` if it can not be done:
    /// the places are not joined, a vertex of `from` has no edge to a vertex of `to`, it would turn
    /// a triangle over, or it would pinch the surface.
    #[allow(
        clippy::too_many_lines,
        reason = "the checks of one collapse, in the order they are made, then what it costs"
    )]
    fn plan(&self, from: u32, to: u32) -> Option<(Plan, f64)> {
        if self.gone[from as usize] || self.gone[to as usize] || self.locked[from as usize] {
            return None;
        }
        let removed = self.shared(from, to);
        if removed.is_empty() {
            return None;
        }

        // A place on an open border only moves along it: an edge of a border has one triangle.
        let on_border = |place: u32| {
            self.neighbours(place)
                .into_iter()
                .any(|other| self.shared(place, other).len() == 1)
        };
        if on_border(from) && removed.len() != 1 {
            return None;
        }

        // The sides of `from` that are met: each vertex of it that a live triangle has, and the
        // vertex of `to` that it has an edge to (the one that is on the triangle of the edge).
        let mut vertex_map: Vec<(u32, u32)> = Vec::new();
        let mut moved: Vec<(u32, usize, u32)> = Vec::new();
        for &triangle in &self.faces[from as usize] {
            if !self.alive[triangle as usize] {
                continue;
            }
            let corners = self.triangles[triangle as usize];
            let places = self.places(triangle);
            let Some(corner) = places.iter().position(|&place| place == from) else {
                continue;
            };
            let vertex = corners[corner];
            if removed.contains(&triangle) {
                // The triangle goes. The vertex of `to` on it is where this side of `from` goes.
                let target = places
                    .iter()
                    .position(|&place| place == to)
                    .map(|at| corners[at])?;
                match vertex_map.iter().find(|(from_vertex, _)| *from_vertex == vertex) {
                    Some(&(_, mapped)) if mapped != target => return None,
                    Some(_) => {}
                    None => vertex_map.push((vertex, target)),
                }
            } else {
                moved.push((triangle, corner, vertex));
            }
        }
        // Every triangle that stays moves its corner to the vertex of its own side. A side that no
        // triangle of the edge is on has none of its own: it takes the vertex of the place that is
        // the nearest to it (or the collapse is not done, with no seam weight).
        let mut rewrites: Vec<(u32, usize, u32)> = Vec::with_capacity(moved.len());
        let mut unmatched = 0_u32;
        let mut borrowed: Vec<(u32, u32)> = Vec::new();
        for (triangle, corner, vertex) in moved {
            let target = if let Some(&(_, target)) = vertex_map
                .iter()
                .find(|(from_vertex, _)| *from_vertex == vertex)
            {
                target
            } else if let Some(&(_, target)) = borrowed
                .iter()
                .find(|(from_vertex, _)| *from_vertex == vertex)
            {
                target
            } else {
                if self.seam_weight <= 0.0 {
                    return None;
                }
                unmatched += 1;
                let target = self.nearest_side(vertex, from, to)?;
                borrowed.push((vertex, target));
                target
            };
            rewrites.push((triangle, corner, target));
        }

        // The surface must not be pinched: the places both touch are the third corners of the
        // triangles of the edge, and no others.
        let near_from = self.neighbours(from);
        let near_to = self.neighbours(to);
        let apexes: Vec<u32> = removed
            .iter()
            .flat_map(|&triangle| self.places(triangle))
            .filter(|&place| place != from && place != to)
            .collect();
        for place in &near_from {
            if *place != to && near_to.contains(place) && !apexes.contains(place) {
                return None;
            }
        }

        // No triangle that stays is turned over (or made into nothing).
        let target_position = self.positions[to as usize];
        if self.turns_over(&rewrites, target_position) {
            return None;
        }

        // Every vertex of `from` that goes, and the vertex of `to` it goes to.
        let mut merged = vertex_map;
        for (vertex, target) in borrowed {
            if !merged.iter().any(|&(from_vertex, _)| from_vertex == vertex) {
                merged.push((vertex, target));
            }
        }
        let shading = self.attributes.as_ref().map_or(0.0, |attributes| {
            attributes.error(&merged, target_position)
        });

        let mut quadric = self.quadrics[from as usize];
        quadric.add(&self.quadrics[to as usize]);
        let moved_by = sub(self.positions[from as usize], target_position);
        let cost = quadric.at(target_position)
            + shading
            + self.seam_weight * f64::from(unmatched) * dot(moved_by, moved_by);
        let mut surface = self.surfaces[from as usize];
        surface.add(&self.surfaces[to as usize]);
        let geometric = surface.at(target_position);
        Some((
            Plan {
                removed,
                moved: rewrites,
                error: geometric + shading,
                geometric,
                merged,
            },
            cost,
        ))
    }

    /// Does the collapses, cheapest first, until `target` triangles are left or none can be done.
    fn collapse_until(&mut self, target: usize) {
        loop {
            while self.alive_triangles > target {
                let Some(candidate) = self.heap.pop() else {
                    break;
                };
                let (from, to) = (candidate.from as usize, candidate.to as usize);
                if self.gone[from]
                    || self.gone[to]
                    || candidate.versions != (self.version[from], self.version[to])
                {
                    continue;
                }
                // What the neighbours did since is checked again: the plan is made now.
                let Some((plan, _)) = self.plan(candidate.from, candidate.to) else {
                    continue;
                };
                self.apply(candidate.from, candidate.to, &plan);
            }
            if self.alive_triangles <= target {
                return;
            }
            // The queue is empty and there are more triangles than asked for: what could not be
            // done before (a neighbour was in the way) may be possible now. Once more over every
            // edge, and if nothing is done in that, it is as small as it gets.
            let before = self.alive_triangles;
            self.push_all();
            if self.heap.is_empty() {
                return;
            }
            self.collapse_pass(target);
            if self.alive_triangles >= before {
                return;
            }
        }
    }

    /// One go over the queue (see [`collapse_until`](Self::collapse_until)).
    fn collapse_pass(&mut self, target: usize) {
        while self.alive_triangles > target {
            let Some(candidate) = self.heap.pop() else {
                return;
            };
            let (from, to) = (candidate.from as usize, candidate.to as usize);
            if self.gone[from]
                || self.gone[to]
                || candidate.versions != (self.version[from], self.version[to])
            {
                continue;
            }
            let Some((plan, _)) = self.plan(candidate.from, candidate.to) else {
                continue;
            };
            self.apply(candidate.from, candidate.to, &plan);
        }
    }

    /// Collapses `from` onto `to` as `plan` says, and puts what changed in the queue.
    fn apply(&mut self, from: u32, to: u32, plan: &Plan) {
        for &triangle in &plan.removed {
            self.alive[triangle as usize] = false;
        }
        self.alive_triangles -= plan.removed.len();
        for &(triangle, corner, vertex) in &plan.moved {
            self.triangles[triangle as usize][corner] = vertex;
            self.faces[to as usize].push(triangle);
        }
        self.faces[from as usize].clear();
        let moved = self.quadrics[from as usize];
        self.quadrics[to as usize].add(&moved);
        let moved = self.surfaces[from as usize];
        self.surfaces[to as usize].add(&moved);
        if let Some(attributes) = &mut self.attributes {
            attributes.merge(&plan.merged);
        }
        self.worst = self.worst.max(plan.error);
        self.worst_geometric = self.worst_geometric.max(plan.geometric);
        self.gone[from as usize] = true;
        self.version[to as usize] += 1;

        // The edges of the place that took the other's: both ways, with new costs.
        for other in self.neighbours(to) {
            self.push(to, other);
            self.push(other, to);
        }
    }

    /// The triangles that are left, in the order of the mesh's.
    fn indices(&self) -> Vec<u32> {
        self.triangles
            .iter()
            .zip(&self.alive)
            .filter(|(_, alive)| **alive)
            .flat_map(|(corners, _)| *corners)
            .collect()
    }

    /// How far the mesh is now from the first level (see [`Level::error`]).
    #[allow(clippy::cast_possible_truncation)] // a distance as a float of the mesh's own
    fn error(&self) -> f32 {
        self.worst.sqrt() as f32
    }

    /// How far the surface is, without the shading (see [`Level::geometric_error`]).
    #[allow(clippy::cast_possible_truncation)] // a distance as a float of the mesh's own
    fn geometric_error(&self) -> f32 {
        self.worst_geometric.sqrt() as f32
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::tuple_array_conversions,
    reason = "tests of geometry on small meshes, with exact numbers where they are exact"
)]
mod tests {
    use super::*;

    /// A mesh in the form the tests make: positions and triangles.
    struct Mesh {
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
    }

    /// A sphere of `rings` rings of `segments` quads (poles are fans), with one vertex for each
    /// place: a closed surface.
    fn sphere(rings: u32, segments: u32) -> Mesh {
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
        let ring_start = |ring: u32| 1 + (ring - 1) * segments;
        let mut indices = Vec::new();
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            indices.extend([0, ring_start(1) + next, ring_start(1) + segment]);
        }
        for ring in 1..rings - 1 {
            for segment in 0..segments {
                let next = (segment + 1) % segments;
                let (a, b) = (ring_start(ring) + segment, ring_start(ring) + next);
                let (c, d) = (ring_start(ring + 1) + segment, ring_start(ring + 1) + next);
                indices.extend([a, b, c, b, d, c]);
            }
        }
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            indices.extend([
                south,
                ring_start(rings - 1) + segment,
                ring_start(rings - 1) + next,
            ]);
        }
        Mesh { positions, indices }
    }

    /// A flat grid of `n` by `n` quads in the plane y = 0, with one vertex for each place.
    fn grid(n: u32) -> Mesh {
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
                indices.extend([at(x, z), at(x, z + 1), at(x + 1, z)]);
                indices.extend([at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)]);
            }
        }
        Mesh { positions, indices }
    }

    /// A cube of 6 sides with each side's own 4 vertices (24 vertices, 8 places), 12 triangles.
    fn cube() -> Mesh {
        let corner = |bits: u32| {
            [
                if bits & 1 == 0 { -1.0 } else { 1.0 },
                if bits & 2 == 0 { -1.0 } else { 1.0 },
                if bits & 4 == 0 { -1.0 } else { 1.0 },
            ]
        };
        // Each side: its four corners in order around it, seen from outside.
        let sides: [[u32; 4]; 6] = [
            [0, 2, 3, 1], // z = -1
            [4, 5, 7, 6], // z = +1
            [0, 1, 5, 4], // y = -1
            [2, 6, 7, 3], // y = +1
            [0, 4, 6, 2], // x = -1
            [1, 3, 7, 5], // x = +1
        ];
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for side in sides {
            let first = u32::try_from(positions.len()).unwrap();
            for bits in side {
                positions.push(corner(bits));
            }
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
        }
        Mesh { positions, indices }
    }

    /// The settings of the strict rule (no side of a seam is given a vertex it has no edge to).
    fn settings(levels: usize, min_triangles: usize) -> Settings {
        Settings {
            levels,
            min_triangles,
            seam_weight: 0.0,
            ..Settings::default()
        }
    }

    /// Every edge of the triangles, by place, with how many triangles have it, for a closed
    /// surface to be checked (all of them 2).
    fn edge_counts(mesh: &Mesh, indices: &[u32]) -> HashMap<(u32, u32), u32> {
        let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
        let places: Vec<u32> = mesh
            .positions
            .iter()
            .map(|position| {
                let key = position.map(|value| (value + 0.0).to_bits());
                let next = u32::try_from(ids.len()).unwrap();
                *ids.entry(key).or_insert(next)
            })
            .collect();
        let mut counts = HashMap::new();
        for triangle in indices.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(|index| places[index as usize]);
            for (from, to) in [(a, b), (b, c), (c, a)] {
                *counts.entry((from.min(to), from.max(to))).or_insert(0) += 1;
            }
        }
        counts
    }

    #[test]
    fn the_first_level_is_the_mesh_as_it_was_given() {
        let mesh = sphere(12, 16);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(3, 8));

        assert_eq!(levels[0].0, mesh.indices);
        assert_eq!(levels[0].1, 0.0);
    }

    #[test]
    fn every_level_has_fewer_triangles_and_no_less_error() {
        let mesh = sphere(24, 32);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(5, 16));

        assert!(levels.len() >= 4, "{} levels", levels.len());
        for pair in levels.windows(2) {
            assert!(pair[1].0.len() < pair[0].0.len());
            assert!(pair[1].1 >= pair[0].1);
        }
        // About half each: the second is not far above half the first.
        let ratio = levels[1].0.len() as f64 / levels[0].0.len() as f64;
        assert!((0.4..=0.55).contains(&ratio), "{ratio}");
    }

    #[test]
    fn a_level_only_uses_the_vertices_of_the_mesh_and_is_made_of_real_triangles() {
        let mesh = sphere(20, 24);
        for (indices, _) in simplify(&mesh.indices, &mesh.positions, &settings(5, 16)) {
            assert_eq!(indices.len() % 3, 0);
            let mut seen = std::collections::HashSet::new();
            for triangle in indices.as_chunks::<3>().0 {
                assert!(triangle.iter().all(|&index| (index as usize) < mesh.positions.len()));
                // Three different vertices, and no triangle twice.
                assert!(triangle[0] != triangle[1] && triangle[1] != triangle[2] && triangle[0] != triangle[2]);
                let mut sorted = *triangle;
                sorted.sort_unstable();
                assert!(seen.insert(sorted), "a triangle is there twice");
            }
        }
    }

    #[test]
    fn a_closed_surface_stays_closed_and_a_sphere_keeps_its_euler_number() {
        let mesh = sphere(24, 32);
        for (indices, _) in simplify(&mesh.indices, &mesh.positions, &settings(5, 16)) {
            let counts = edge_counts(&mesh, &indices);
            assert!(counts.values().all(|&count| count == 2), "an open or a pinched edge");
            let places: std::collections::HashSet<u32> = counts
                .keys()
                .flat_map(|&(a, b)| [a, b])
                .collect();
            // V - E + F = 2 for a sphere.
            let euler = places.len() as i64 - counts.len() as i64 + (indices.len() / 3) as i64;
            assert_eq!(euler, 2);
        }
    }

    #[test]
    fn a_flat_surface_goes_down_to_a_few_triangles_for_no_error() {
        let mesh = grid(16);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(8, 2));
        let last = levels.last().unwrap();

        assert!(last.0.len() / 3 <= 16, "{} triangles", last.0.len() / 3);
        assert!(last.1 < 1.0e-6, "error {}", last.1);
    }

    #[test]
    fn the_border_of_a_flat_surface_stays_where_it_was() {
        let mesh = grid(12);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(6, 2));
        let last = levels.last().unwrap();

        // The four corners are still used, and no vertex is outside of the square.
        for corner in [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 0.0, 12.0], [12.0, 0.0, 12.0]] {
            let used = last.0.iter().any(|&index| mesh.positions[index as usize] == corner);
            assert!(used, "the corner {corner:?} is gone");
        }
        // The area of what is left is the area of the square (the border did not move in).
        let mut area = 0.0_f64;
        for triangle in last.0.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(|index| mesh.positions[index as usize].map(f64::from));
            area += length(cross(sub(b, a), sub(c, a))) / 2.0;
        }
        assert!((area - 144.0).abs() < 1.0e-6, "area {area}");
    }

    #[test]
    fn locked_borders_keep_every_original_edge_while_the_interior_simplifies() {
        let mut mesh = grid(12);
        // A curved outline, as on independently loaded material parts. A finite border weight
        // can shorten it, unlike the square outline in the test above.
        for p in &mut mesh.positions {
            p[1] = (p[0] * 0.3).sin() + (p[2] * 0.2).cos();
        }
        let original = edge_counts(&mesh, &mesh.indices);
        let boundary: Vec<_> = original
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(edge, _)| *edge)
            .collect();
        let settings = Settings {
            lock_borders: true,
            ..settings(6, 2)
        };
        let levels = simplify(&mesh.indices, &mesh.positions, &settings);
        assert!(levels.len() >= 2, "interior simplification must remain possible");
        assert!(levels.last().unwrap().0.len() < mesh.indices.len() / 2);
        for (indices, _) in levels {
            let current = edge_counts(&mesh, &indices);
            for edge in &boundary {
                assert_eq!(current.get(edge), Some(&1), "a shared border changed");
            }
            assert_eq!(current.values().filter(|&&count| count == 1).count(), boundary.len());
        }
    }

    #[test]
    fn a_cube_with_a_side_of_its_own_for_each_face_stays_a_cube() {
        let mesh = cube();
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(3, 2));

        // Collapsing a corner (three sides meet) onto a place with two is not possible, and that
        // is every edge: no level below the cube.
        assert_eq!(levels.len(), 1, "{:?}", levels.iter().map(|l| l.0.len() / 3).collect::<Vec<_>>());
    }

    /// A tube of `rings` rings, `segments` around, with the seam of its texture at segment 0: the
    /// vertices there are twice (a different texture coordinate each side), at the same place.
    fn seamed_tube(rings: u32, segments: u32) -> Mesh {
        let mut positions = Vec::new();
        // One more vertex at the seam than there are segments.
        for ring in 0..rings {
            for segment in 0..=segments {
                let azimuth = std::f32::consts::TAU * (segment % segments) as f32 / segments as f32;
                positions.push([azimuth.cos(), ring as f32, azimuth.sin()]);
            }
        }
        let at = |ring: u32, segment: u32| ring * (segments + 1) + segment;
        let mut indices = Vec::new();
        for ring in 0..rings - 1 {
            for segment in 0..segments {
                let (a, b) = (at(ring, segment), at(ring, segment + 1));
                let (c, d) = (at(ring + 1, segment), at(ring + 1, segment + 1));
                indices.extend([a, c, b, b, c, d]);
            }
        }
        Mesh { positions, indices }
    }

    #[test]
    fn a_seam_does_not_open_up_when_the_two_sides_are_the_same_place() {
        let mesh = seamed_tube(12, 12);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(5, 4));
        assert!(levels.len() > 1);
        for (indices, _) in &levels {
            // By place the tube has two borders (its ends) and no other open edge: a crack at the
            // seam would be an edge with one triangle in the middle of it.
            let counts = edge_counts(&mesh, indices);
            for (&(a, b), &count) in &counts {
                assert!(count <= 2, "an edge ({a}, {b}) of {count} triangles");
            }
            let open = counts.values().filter(|&&count| count == 1).count();
            // Only the two ends: the number of their edges is not more than the segments of both.
            assert!(open <= 24, "{open} open edges");
        }
    }

    /// The triangles of `mesh` with a vertex of their own for each corner: every place is as many
    /// vertices as triangles touch it.
    fn soup(mesh: &Mesh) -> Mesh {
        let mut positions = Vec::new();
        for &index in &mesh.indices {
            positions.push(mesh.positions[index as usize]);
        }
        let indices = (0..u32::try_from(positions.len()).unwrap()).collect();
        Mesh { positions, indices }
    }

    #[test]
    fn a_mesh_of_separate_triangles_does_not_go_down_by_the_strict_rule_and_does_with_a_seam_weight() {
        let mesh = soup(&sphere(16, 24));

        let strict = simplify(&mesh.indices, &mesh.positions, &settings(4, 16));
        assert_eq!(strict.len(), 1);

        let relaxed = Settings {
            seam_weight: 1.0,
            ..settings(4, 16)
        };
        let levels = simplify(&mesh.indices, &mesh.positions, &relaxed);
        assert!(levels.len() >= 3, "{} levels", levels.len());
        for (indices, _) in &levels {
            // By place it is the same closed sphere: every side moved with its place, so nothing
            // is open and nothing is pinched.
            let counts = edge_counts(&mesh, indices);
            assert!(counts.values().all(|&count| count == 2));
        }
    }

    fn corners(mesh: &Mesh, normal: impl Fn(usize) -> [f32; 3]) -> Vec<Corner> {
        mesh.positions
            .iter()
            .enumerate()
            .map(|(index, &position)| Corner {
                position,
                normal: normal(index),
                texture_coord: [position[0] / 12.0, position[2] / 12.0],
            })
            .collect()
    }

    #[test]
    fn a_flat_surface_with_smooth_shading_still_goes_down_for_no_error() {
        let mesh = grid(12);
        // The normal is the same everywhere and the texture coordinates are a linear function of
        // the position: a bigger triangle has all of it right.
        let vertices = corners(&mesh, |_| [0.0, 1.0, 0.0]);

        let levels = simplify(&mesh.indices, &vertices, &settings(6, 2));

        let last = levels.last().unwrap();
        assert!(last.0.len() < mesh.indices.len() / 4, "{} triangles", last.0.len() / 3);
        assert!(last.1 < 1.0e-3, "error {}", last.1);
    }

    #[test]
    fn shading_that_changes_on_a_flat_surface_is_part_of_the_error() {
        let mesh = grid(12);
        // Flat, with the normal of every other vertex turned aside: the positions say any
        // triangle will do, the shading does not.
        let vertices = corners(&mesh, |index| {
            if index % 2 == 0 { [0.0, 1.0, 0.0] } else { [0.8, 0.6, 0.0] }
        });
        let positions_only = simplify(&mesh.indices, &mesh.positions, &settings(6, 2));
        let shaded = simplify(&mesh.indices, &vertices, &settings(6, 2));

        assert!(positions_only.last().unwrap().1 < 1.0e-3);
        assert!(shaded.len() >= 2);
        assert!(shaded.last().unwrap().1 > 0.1, "error {}", shaded.last().unwrap().1);
        // The surface itself is flat: the geometric error, which a shadow uses, has no shading.
        let errors = simplify_with_errors(&mesh.indices, &vertices, &settings(6, 2));
        let (_, error, geometric) = errors.last().unwrap();
        assert!(*geometric < 1.0e-3 && geometric <= error, "{geometric} of {error}");
        // And what is left out is left out: with the weights at 0 it is the positions again.
        let off = Settings {
            normal_weight: 0.0,
            uv_weight: 0.0,
            ..settings(6, 2)
        };
        let ignored = simplify(&mesh.indices, &vertices, &off);
        assert_eq!(ignored, positions_only);
    }

    #[test]
    fn the_cheaper_shading_goes_first() {
        // The left half of a flat surface has one normal and the right half another, with a sharp
        // change between: the collapses in the halves cost nothing and are done before any that
        // spans the change.
        let mesh = grid(12);
        let vertices = corners(&mesh, |index| {
            if mesh.positions[index][0] < 6.0 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] }
        });
        let levels = simplify(&mesh.indices, &vertices, &settings(1, 2));

        assert_eq!(levels.len(), 2);
        // Half the triangles are gone, and what was done cost nothing: it was in the halves.
        assert!(levels[1].1 < 1.0e-3, "error {}", levels[1].1);
    }

    #[test]
    fn the_same_mesh_makes_the_same_levels() {
        let mesh = sphere(16, 20);
        let first = simplify(&mesh.indices, &mesh.positions, &settings(4, 8));
        let second = simplify(&mesh.indices, &mesh.positions, &settings(4, 8));

        assert_eq!(first, second);
    }

    #[test]
    fn a_mesh_with_nothing_in_it_has_the_one_level() {
        let positions: Vec<[f32; 3]> = vec![[0.0; 3], [1.0, 0.0, 0.0]];
        assert_eq!(simplify(&[], &positions, &settings(3, 1)).len(), 1);
        // A triangle with two corners in one place has no area, and is not in a level of its own.
        let levels = simplify(&[0, 0, 1], &positions, &settings(3, 1));
        assert_eq!(levels.len(), 1);
    }

    #[test]
    fn a_level_of_a_sphere_is_near_the_sphere() {
        let mesh = sphere(32, 48);
        let levels = simplify(&mesh.indices, &mesh.positions, &settings(4, 16));
        for (indices, error) in &levels {
            // Every vertex of the level is a vertex of the mesh, on the unit sphere: the surface
            // of the level can only be inside of it, by about what the error says.
            for &index in indices {
                let [x, y, z] = mesh.positions[index as usize];
                let radius = (x * x + y * y + z * z).sqrt();
                assert!((radius - 1.0).abs() < 1.0e-4);
            }
            assert!(*error < 0.5, "error {error}");
        }
        // The smaller the level the more it is off.
        assert!(levels.last().unwrap().1 > levels[1].1 * 0.99);
    }

    #[test]
    fn the_meshlets_of_a_level_are_made_from_its_own_indices() {
        let mesh = sphere(24, 32);
        let levels = build(
            &mesh.indices,
            &mesh.positions,
            &settings(3, 16),
            (Limits::MAX, 0.5, false),
        );

        assert!(levels.len() >= 2);
        for level in &levels {
            assert!(!level.meshlets.is_empty());
            // The meshlets have the triangles of the level (the same set, in another order).
            assert_eq!(level.meshlets.indices.len(), level.indices.len());
            let mut a: Vec<[u32; 3]> = level.indices.as_chunks::<3>().0.to_vec();
            let mut b: Vec<[u32; 3]> = level.meshlets.indices.as_chunks::<3>().0.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
        assert!(levels[1].meshlets.indices.len() < levels[0].meshlets.indices.len());
    }
}
