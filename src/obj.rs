use std::{fs::File, io::BufReader, path::Path};

use crate::{
    Error,
    parse::{FaceData, GroupingData},
};

#[derive(Debug)]
/// A representation of a .obj file.
///
/// This library interprets the .obj format with the following hierarchy:
///
/// A single .obj file can contain multiple objects (o).
/// If there are no explicit objects contained in the file, a unnamed object for the whole file is assumed.
///
/// Every object (o) can contain multiple groups (g).
/// If there are no explicit groups contained in the object, a unnamed group for the whole object (o) is assumed.
///
/// Every group (g) can contain multiple faces (f).
///
/// And each face (f) contains 3 vertices. If more then 3 vertices are specified for any face (f), they are automatically triangularized.
/// A maximum of 4 vertices in a face can be triangularized.
///
/// # Example
/// ```rust
/// # use polypath::ObjObject;
/// let mesh = "./meshes/cubes.obj";
/// let obj = ObjObject::read_from_file(mesh).unwrap();
///
/// // read .obj file
/// // automatic triangulates faces (from max 4 vertices per face)
/// for o in obj.objects_iter() {
///     println!("Object name: {}", o.name());
///     println!("Object material: {:?}", o.mtllib());
///
///     // an .obj file can contain multiple groups
///     for g in o.group_iter() {
///         println!("Group name: {}", g.name());
///         println!("Group material: {:?}", g.mtluse());
///
///         // each group can contain multiple faces
///         for f in g.faces_iter() {
///             // extract set of 3 vertices from each face
///            let [v1, v2, v3] = f.vertices();
///
///             println!("Positions: ");
///             println!("{:?}", v1.position);
///             println!("{:?}", v2.position);
///             println!("{:?}", v3.position);
///         }
///     }
/// }
/// ```
pub struct ObjObject {
    pub(crate) vertices: Vec<(f32, f32, f32)>,
    pub(crate) vertex_colors: Vec<(f32, f32, f32)>,
    pub(crate) vertex_normals: Vec<(f32, f32, f32)>,
    pub(crate) texture_coords: Vec<(f32, f32)>,

    pub(crate) faces: Vec<FaceData>,

    pub(crate) groups: Vec<GroupingData>,
    pub(crate) objects: Vec<GroupingData>,
}

impl ObjObject {
    /// Reads a .obj file and returns a ObjObject.
    ///
    /// # Error
    /// - Returns an [Error][std::io::Error] if reading from file fails
    /// - Returns other errors encountered when parsing the file
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let buffer = BufReader::new(file);

        Self::parse(buffer)
    }

    #[inline]
    /// Returns the number of individual objects contained in the .obj file.
    pub const fn object_count(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    /// Returns the number of individual groups contained in the .obj file.
    pub const fn group_count(&self) -> usize {
        self.groups.len()
    }

    #[inline]
    /// Returns the number of individual faces contained in the .obj file.
    pub const fn face_count(&self) -> usize {
        self.faces.len()
    }

    #[inline]
    /// Returns the raw number of individual verticies contained in the .obj file.
    ///
    /// This function is not 100% prezise, as it just calculates 3 verticices for each face.
    /// Vertices that are shared are not considered.
    pub const fn vert_count(&self) -> usize {
        self.faces.len() * 3
    }

    /// Returns an [Iterator][std::iter::Iterator] over each object.
    pub fn objects_iter(&self) -> impl Iterator<Item = ObjectRef> {
        self.objects.iter().map(|obj| ObjectRef {
            vertices: &self.vertices,
            vertex_colors: vec_to_option(&self.vertex_colors),
            vertex_normals: &self.vertex_normals,
            texture_coords: &self.texture_coords,

            faces: &self.faces,

            name: &obj.name,
            mtllib: obj.mtl.as_ref(),

            groups: &self.groups[obj.start..obj.finish],
        })
    }

    /// Returns:
    ///     - a [Vec][std::vec::Vec] containing 3 vertices for each face. Vertices that are shared are duplicated. Every 3 vertices build a face.
    ///     - a [Vec][std::vec::Vec] containing [`MaterialIdent`]. Each returned vertex contains a `material_index` that can be used to index into this list, to retrive the [`MaterialIdent`].
    /// This ignores any grouping done via objects (o) or groups (g).
    /// If keeping these groupings is important, consider iterating manually over each object/group/face.
    pub fn vertices(&self) -> (Vec<VertexTextureData>, Vec<MaterialIdent>) {
        let mut vertices = Vec::with_capacity(self.vert_count());
        let mut materials = Vec::<MaterialIdent>::new();

        for obj in self.objects_iter() {
            let mtllib = obj.mtllib.map(String::as_str);

            for group in obj.group_iter() {
                let mtluse = group.mtluse.map(String::as_str);

                let t = MaterialIdent { mtllib, mtluse };
                let texture_index = materials.iter().position(|m| *m == t).unwrap_or_else(|| {
                    materials.push(t);
                    materials.len() - 1
                });

                for f in group.faces_iter() {
                    for v in f.vertices() {
                        let vert = VertexTextureData {
                            material_index: texture_index,
                            vertex: v,
                        };

                        vertices.push(vert);
                    }
                }
            }
        }

        (vertices, materials)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Identifies a material.
///
/// Consists of
/// - a material library (mtllib)
/// - a material use (mtluse)
pub struct MaterialIdent<'a> {
    pub mtllib: Option<&'a str>,
    pub mtluse: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectRef<'a> {
    vertices: &'a [(f32, f32, f32)],
    vertex_colors: Option<&'a [(f32, f32, f32)]>,
    vertex_normals: &'a [(f32, f32, f32)],
    texture_coords: &'a [(f32, f32)],

    faces: &'a [FaceData],

    name: &'a str,
    mtllib: Option<&'a String>,

    groups: &'a [GroupingData],
}

impl<'a> ObjectRef<'a> {
    #[inline]
    pub const fn name(&self) -> &str {
        self.name
    }

    #[inline]
    pub fn mtllib(&self) -> Option<&str> {
        self.mtllib.map(String::as_str)
    }

    #[inline]
    pub const fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn group_iter(&self) -> impl Iterator<Item = GroupRef<'a>> {
        self.groups.iter().map(|group| GroupRef {
            vertices: self.vertices,
            vertex_colors: self.vertex_colors,
            vertex_normals: self.vertex_normals,
            texture_coords: self.texture_coords,

            name: &group.name,
            mtluse: group.mtl.as_ref(),
            faces: &self.faces[group.start..group.finish],
        })
    }

    #[inline]
    pub fn faces(&self) -> Vec<&[FaceData]> {
        let mut faces = 0;
        for g in self.groups {
            faces += g.finish - g.start;
        }

        let mut out = Vec::with_capacity(faces);

        for g in self.groups {
            out.push(&self.faces[g.start..g.finish]);
        }

        out
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GroupRef<'a> {
    vertices: &'a [(f32, f32, f32)],
    vertex_colors: Option<&'a [(f32, f32, f32)]>,
    vertex_normals: &'a [(f32, f32, f32)],
    texture_coords: &'a [(f32, f32)],

    name: &'a str,
    mtluse: Option<&'a String>,
    faces: &'a [FaceData],
}

impl GroupRef<'_> {
    #[inline]
    pub const fn name(&self) -> &str {
        self.name
    }

    #[inline]
    pub fn mtluse(&self) -> Option<&str> {
        self.mtluse.map(String::as_str)
    }

    #[inline]
    pub const fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn faces_iter(&self) -> impl Iterator<Item = Face> {
        self.faces.iter().map(|face| {
            let (i1, i2, i3) = face.indicies;

            Face {
                vert_positions: [
                    self.vertices[i1 as usize - 1],
                    self.vertices[i2 as usize - 1],
                    self.vertices[i3 as usize - 1],
                ],

                vert_colors: self.vertex_colors.map(|colors| {
                    [
                        colors[i1 as usize - 1],
                        colors[i2 as usize - 1],
                        colors[i3 as usize - 1],
                    ]
                }),
                vert_normals: face.normal_indicies.map(|(n1, n2, n3)| {
                    [
                        self.vertex_normals[n1 as usize - 1],
                        self.vertex_normals[n2 as usize - 1],
                        self.vertex_normals[n3 as usize - 1],
                    ]
                }),

                vert_uv_coords: face.texture_indcicies.map(|(t1, t2, t3)| {
                    [
                        self.texture_coords[t1 as usize - 1],
                        self.texture_coords[t2 as usize - 1],
                        self.texture_coords[t3 as usize - 1],
                    ]
                }),
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
/// Represents 3 vertices.
///
/// Contains:
///     - the vertex position for each vertex
///     - the vertex color for each vertex (optional)
///     - the vertex normals for each vertex (optional)
///     - the vertex uv coordinates for each vertex (optional)
///
/// # Examples
/// ```rust
/// # use polypath::Face;
/// let face = Face {
///     vert_positions: [
///         (0.0, 0.0, 0.0),
///         (1.0, 0.0, 0.0),
///         (1.0, 1.0, 0.0),
///     ],
///     vert_colors: None,
///     vert_normals: None,
///     vert_uv_coords: None,
/// };
/// ```
pub struct Face {
    pub vert_positions: [(f32, f32, f32); 3],
    pub vert_colors: Option<[(f32, f32, f32); 3]>,
    pub vert_normals: Option<[(f32, f32, f32); 3]>,
    pub vert_uv_coords: Option<[(f32, f32); 3]>,
}

impl Face {
    pub const fn vertices(&self) -> [VertexData; 3] {
        let [v1p, v2p, v3p] = self.vert_positions;

        let [v1c, v2c, v3c] = option_to_array(self.vert_colors);

        let [v1n, v2n, v3n] = option_to_array(self.vert_normals);

        let [v1t, v2t, v3t] = option_to_array(self.vert_uv_coords);

        [
            VertexData {
                position: v1p,
                color: v1c,
                normal: v1n,
                texture_coord: v1t,
            },
            VertexData {
                position: v2p,
                color: v2c,
                normal: v2n,
                texture_coord: v2t,
            },
            VertexData {
                position: v3p,
                color: v3c,
                normal: v3n,
                texture_coord: v3t,
            },
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// Represents the data of a single vertex.
///
/// Contains:
///     - the vertex position
///     - the vertex color (optional)
///     - the vertex normals (optional)
///     - the vertex uv coordinates (optional)
pub struct VertexData {
    pub position: (f32, f32, f32),
    pub color: Option<(f32, f32, f32)>,
    pub normal: Option<(f32, f32, f32)>,
    pub texture_coord: Option<(f32, f32)>,
}

impl Eq for VertexData {}

impl std::hash::Hash for VertexData {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let (x, y, z) = self.position;
        (f32::to_bits(x), f32::to_bits(y), f32::to_bits(z)).hash(state);

        self.color
            .map(|(r, g, b)| (f32::to_bits(r), f32::to_bits(g), f32::to_bits(b)))
            .hash(state);

        self.normal
            .map(|(u, v, w)| (f32::to_bits(u), f32::to_bits(v), f32::to_bits(w)))
            .hash(state);

        self.texture_coord
            .map(|(u, v)| (f32::to_bits(u), f32::to_bits(v)))
            .hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
/// Represents a single vertex, included associated material.
pub struct VertexTextureData {
    /// Can be used to index into a [Vec][std::vec::Vec] of [`MaterialIdent`].
    pub material_index: usize,
    pub vertex: VertexData,
}

#[inline]
fn vec_to_option<V: AsRef<[T]>, T>(vec: &V) -> Option<&[T]> {
    let slice = vec.as_ref();
    if slice.is_empty() { None } else { Some(slice) }
}

#[inline]
const fn option_to_array<T: Copy>(opt: Option<[T; 3]>) -> [Option<T>; 3] {
    match opt {
        Some([o1, o2, o3]) => [Some(o1), Some(o2), Some(o3)],
        None => [None, None, None],
    }
}
