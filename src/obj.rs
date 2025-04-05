use std::{
    collections::{HashMap, hash_map::Entry},
    fs::File,
    io::BufReader,
    path::Path,
};

use rustc_hash::FxBuildHasher;

use crate::{
    Error,
    parse::{FaceData, GroupingData},
};

#[derive(Debug)]
pub struct ObjObject {
    pub(crate) verticies: Vec<(f32, f32, f32)>,
    pub(crate) vertex_colors: Vec<(f32, f32, f32)>,
    pub(crate) vertex_normals: Vec<(f32, f32, f32)>,
    pub(crate) texture_coords: Vec<(f32, f32)>,

    pub(crate) faces: Vec<FaceData>,

    pub(crate) groups: Vec<GroupingData>,
    pub(crate) objects: Vec<GroupingData>,
}

impl ObjObject {
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let buffer = BufReader::new(file);

        Self::parse(buffer)
    }

    #[inline]
    pub const fn object_count(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    pub const fn group_count(&self) -> usize {
        self.groups.len()
    }

    #[inline]
    pub const fn face_count(&self) -> usize {
        self.faces.len()
    }

    #[inline]
    pub const fn vert_count(&self) -> usize {
        self.faces.len() * 3
    }

    pub fn objects_iter(&self) -> impl Iterator<Item = ObjectRef> {
        self.objects.iter().map(|obj| ObjectRef {
            verticies: &self.verticies,
            vertex_colors: vec_to_option(&self.vertex_colors),
            vertex_normals: &self.vertex_normals,
            texture_coords: &self.texture_coords,

            faces: &self.faces,

            name: &obj.name,
            mtllib: obj.mtl.as_ref(),

            groups: &self.groups[obj.start..obj.finish],
        })
    }

    pub fn verticies(&self) -> (Vec<VertexTextureData>, Vec<TextureIdent>) {
        let mut verticies = Vec::with_capacity(self.vert_count());
        let mut materials = Vec::<TextureIdent>::new();

        for obj in self.objects_iter() {
            let mtllib = obj.mtllib.map(String::as_str);

            for group in obj.group_iter() {
                let mtluse = group.mtluse.map(String::as_str);

                let t = TextureIdent { mtllib, mtluse };
                let texture_index = materials.iter().position(|m| *m == t).unwrap_or_else(|| {
                    materials.push(t);
                    materials.len() - 1
                });

                for f in group.faces_iter() {
                    for v in f.verticies() {
                        let vert = VertexTextureData {
                            material_index: texture_index,
                            vertex: v,
                        };

                        verticies.push(vert);
                    }
                }
            }
        }

        (verticies, materials)
    }

    pub fn verticies_indexed(&self) -> (Vec<usize>, Vec<VertexTextureData>, Vec<TextureIdent>) {
        let mut verticies = HashMap::<VertexDataComp, usize, _>::with_capacity_and_hasher(
            self.vert_count(),
            FxBuildHasher,
        );
        let mut indicies = Vec::with_capacity(self.vert_count());
        let mut materials = vec![TextureIdent {
            mtllib: None,
            mtluse: None,
        }];
        let mut counter = 0;

        // generate materials vector
        // and deduplicate verticies
        for obj in self.objects_iter() {
            let mtllib = obj.mtllib.map(String::as_str);

            for group in obj.group_iter() {
                let mtluse = group.mtluse.map(String::as_str);

                let t = TextureIdent { mtllib, mtluse };
                let texture_index = materials.iter().position(|m| *m == t).unwrap_or_else(|| {
                    materials.push(t);
                    materials.len() - 1
                });

                for f in group.faces_iter() {
                    for v in f.verticies() {
                        let vert = VertexTextureData {
                            material_index: texture_index,
                            vertex: v,
                        };

                        match verticies.entry(VertexDataComp::from(vert)) {
                            Entry::Occupied(occupied_entry) => indicies.push(*occupied_entry.get()),
                            Entry::Vacant(vacant_entry) => {
                                vacant_entry.insert(counter);
                                indicies.push(counter);
                                counter += 1;
                            }
                        }

                        debug_assert!(verticies.contains_key(&VertexDataComp::from(vert)));
                    }
                }
            }
        }

        // arrange vertex buffer
        let mut vertex_buffer = Vec::with_capacity(verticies.len());
        vertex_buffer.resize(verticies.len(), VertexTextureData::default());

        assert_eq!(counter, verticies.len());

        for (vert, pos) in verticies {
            vertex_buffer[pos] = VertexTextureData::from(vert);
        }

        (indicies, vertex_buffer, materials)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureIdent<'a> {
    pub mtllib: Option<&'a str>,
    pub mtluse: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectRef<'a> {
    verticies: &'a [(f32, f32, f32)],
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
            verticies: self.verticies,
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
    verticies: &'a [(f32, f32, f32)],
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
                    self.verticies[i1 as usize - 1],
                    self.verticies[i2 as usize - 1],
                    self.verticies[i3 as usize - 1],
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
pub struct Face {
    vert_positions: [(f32, f32, f32); 3],
    vert_colors: Option<[(f32, f32, f32); 3]>,
    vert_normals: Option<[(f32, f32, f32); 3]>,
    vert_uv_coords: Option<[(f32, f32); 3]>,
}

impl Face {
    pub const fn verticies(&self) -> [VertexData; 3] {
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
pub struct VertexData {
    pub position: (f32, f32, f32),
    pub color: Option<(f32, f32, f32)>,
    pub normal: Option<(f32, f32, f32)>,
    pub texture_coord: Option<(f32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VertexTextureData {
    pub material_index: usize,
    pub vertex: VertexData,
}

impl From<VertexDataComp> for VertexTextureData {
    #[inline]
    fn from(value: VertexDataComp) -> Self {
        let (x, y, z) = value.position;

        Self {
            material_index: value.texture_index,
            vertex: VertexData {
                position: (
                    f32::from_ne_bytes(x),
                    f32::from_ne_bytes(y),
                    f32::from_ne_bytes(z),
                ),
                color: value.color.map(|(r, g, b)| {
                    (
                        f32::from_ne_bytes(r),
                        f32::from_ne_bytes(g),
                        f32::from_ne_bytes(b),
                    )
                }),
                normal: value.normal.map(|(u, v, w)| {
                    (
                        f32::from_ne_bytes(u),
                        f32::from_ne_bytes(v),
                        f32::from_ne_bytes(w),
                    )
                }),
                texture_coord: value
                    .texture_coord
                    .map(|(u, v)| (f32::from_ne_bytes(u), f32::from_ne_bytes(v))),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VertexDataComp {
    pub texture_index: usize,
    pub position: ([u8; 4], [u8; 4], [u8; 4]),
    pub color: Option<([u8; 4], [u8; 4], [u8; 4])>,
    pub normal: Option<([u8; 4], [u8; 4], [u8; 4])>,
    pub texture_coord: Option<([u8; 4], [u8; 4])>,
}

impl From<VertexTextureData> for VertexDataComp {
    #[inline]
    fn from(value: VertexTextureData) -> Self {
        let (x, y, z) = value.vertex.position;

        Self {
            texture_index: value.material_index,
            position: (
                f32::to_ne_bytes(x),
                f32::to_ne_bytes(y),
                f32::to_ne_bytes(z),
            ),
            color: value.vertex.color.map(|(r, g, b)| {
                (
                    f32::to_ne_bytes(r),
                    f32::to_ne_bytes(g),
                    f32::to_ne_bytes(b),
                )
            }),
            normal: value.vertex.normal.map(|(u, v, w)| {
                (
                    f32::to_ne_bytes(u),
                    f32::to_ne_bytes(v),
                    f32::to_ne_bytes(w),
                )
            }),
            texture_coord: value
                .vertex
                .texture_coord
                .map(|(u, v)| (f32::to_ne_bytes(u), f32::to_ne_bytes(v))),
        }
    }
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
