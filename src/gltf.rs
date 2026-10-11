//! A loader for glTF 2.0 files.
//!
//! It reads `.gltf` (with its `.bin` files and pictures next to it) and the single file `.glb`, and
//! gives plain data: the triangles of every mesh, the materials, where the pictures are, and where
//! each node puts each mesh in the world.
//!
//! The file is read with the `gltf` crate; this module only turns what it finds into data that
//! does not depend on it, in the conventions of this crate and of the engine that uses it:
//!
//! - Matrices are `[[f32; 4]; 4]` in the **row-vector** convention (a point is a row, and is
//!   multiplied with the matrix on its left, so a node's matrix comes before its parent's). That is
//!   the same bytes glTF has (it stores column-major for column vectors), so nothing is turned.
//! - Positions, normals and tangents are in the space of the mesh; the node matrix moves them to
//!   the world. A scene is a list of [`GltfInstance`] with the matrix already multiplied down the
//!   hierarchy.
//! - Nothing is decoded: a picture is its file's bytes (or where to find the file), and the caller
//!   decodes it with whatever it has. Colours are linear, as glTF says; the picture's own
//!   colour space is by use: base colour and emissive are sRGB, the others are not.
//!
//! Not read (yet): cameras, skins, animations, morph targets, sparse accessors beyond what the
//! `gltf` crate resolves itself, texture samplers and `KHR_texture_transform`. Meshes that are
//! points or lines are left out (and counted in [`GltfScene::skipped_primitives`]).

use std::path::{Path, PathBuf};

use crate::Error;

/// A glTF file, read.
#[derive(Debug, Clone, Default)]
pub struct GltfScene {
    /// The meshes, by their number in the file.
    pub meshes: Vec<GltfMesh>,
    /// The materials, by their number in the file.
    pub materials: Vec<GltfMaterial>,
    /// The pictures, by their number in the file.
    pub images: Vec<GltfImage>,
    /// Every use of a mesh in the default scene (the first one if none is the default), with its
    /// transform in the world.
    pub instances: Vec<GltfInstance>,
    /// How many primitives were left out because they are not made of triangles.
    pub skipped_primitives: usize,
}

/// A mesh: one or more parts, each with its own material.
#[derive(Debug, Clone, Default)]
pub struct GltfMesh {
    pub name: Option<String>,
    pub primitives: Vec<GltfPrimitive>,
}

/// The triangles of one part of a mesh, with the same number of every attribute it has.
#[derive(Debug, Clone, Default)]
pub struct GltfPrimitive {
    pub positions: Vec<[f32; 3]>,
    /// `None` if the file has none (the normals are then to be made by the caller).
    pub normals: Option<Vec<[f32; 3]>>,
    /// The first set of texture coordinates, `v` growing **down** the picture (as glTF has it).
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Tangent (xyz) and the handedness of the bitangent (w, 1 or -1).
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Three for each triangle, counter clockwise when seen from the front.
    pub indices: Vec<u32>,
    /// The number of its material in [`GltfScene::materials`], `None` for the default one.
    pub material: Option<usize>,
}

/// A use of a mesh in the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfInstance {
    /// The number of the mesh in [`GltfScene::meshes`].
    pub mesh: usize,
    /// Takes the points of the mesh to the world (row-vector convention).
    pub transform: [[f32; 4]; 4],
    pub name: Option<String>,
}

/// A picture a material uses, and which texture coordinates it is read with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRef {
    /// The number of the picture in [`GltfScene::images`].
    pub image: usize,
    /// Which set of texture coordinates (0 is the first).
    pub tex_coord: u32,
}

/// How a material takes the alpha of what it draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Alpha is ignored.
    #[default]
    Opaque,
    /// Alpha below the cutoff is thrown away.
    Mask,
    /// Alpha blends with what is behind.
    Blend,
}

/// A metallic-roughness material.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfMaterial {
    pub name: Option<String>,
    /// Multiplied with the picture, linear.
    pub base_color_factor: [f32; 4],
    pub base_color: Option<TextureRef>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    /// One picture with the roughness in its **green** channel and the metalness in its **blue**
    /// one (the others are not used here: red can be the occlusion).
    pub metallic_roughness: Option<TextureRef>,
    pub normal: Option<TextureRef>,
    /// How much the normal map bends the normal.
    pub normal_scale: f32,
    /// The occlusion is in the **red** channel.
    pub occlusion: Option<TextureRef>,
    pub occlusion_strength: f32,
    pub emissive: Option<TextureRef>,
    pub emissive_factor: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

/// Where the bytes of a picture are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    /// A file, next to the glTF file or below it (the path is complete).
    File(PathBuf),
    /// The bytes of a file of a picture, from inside of the glTF file (a `.glb`, or a `data:` URI).
    Embedded(Vec<u8>),
}

/// A picture, not decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GltfImage {
    pub name: Option<String>,
    pub source: ImageSource,
    /// The media type the file says (`image/png`, `image/jpeg`), if it does.
    pub mime_type: Option<String>,
}

impl GltfScene {
    /// Reads a `.gltf` or `.glb` file, and the buffers it refers to.
    ///
    /// # Errors
    /// [`Error::Gltf`] if the file is not a valid glTF 2.0 file, a buffer can not be found or is
    /// too short, or an accessor is not what its use says.
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path.as_ref();
        let gltf = gltf::Gltf::open(path)
            .map_err(|err| Error::Gltf(format!("{}: {err}", path.display())))?;
        let folder = path.parent().unwrap_or_else(|| Path::new("."));
        let buffers = load_buffers(&gltf, folder)?;
        build(&gltf.document, &buffers, folder)
    }

    /// Reads a `.glb` (or a `.gltf` without files of its own) from memory. Files it refers to are
    /// looked for in `folder`.
    ///
    /// # Errors
    /// As [`read_from_file`](Self::read_from_file).
    pub fn read_from_slice(bytes: &[u8], folder: &Path) -> Result<Self, Error> {
        let gltf = gltf::Gltf::from_slice(bytes).map_err(|err| Error::Gltf(err.to_string()))?;
        let buffers = load_buffers(&gltf, folder)?;
        build(&gltf.document, &buffers, folder)
    }
}

fn load_buffers(gltf: &gltf::Gltf, folder: &Path) -> Result<Vec<Vec<u8>>, Error> {
    let mut buffers = Vec::new();
    for buffer in gltf.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf.blob.clone().ok_or_else(|| {
                Error::Gltf("a buffer is in the .glb file, which has none".into())
            })?,
            gltf::buffer::Source::Uri(uri) => read_uri(uri, folder)?.0,
        };
        if data.len() < buffer.length() {
            return Err(Error::Gltf(format!(
                "buffer {} is {} bytes, but should be {}",
                buffer.index(),
                data.len(),
                buffer.length()
            )));
        }
        buffers.push(data);
    }
    Ok(buffers)
}

/// The bytes a URI names: a `data:` URI is decoded, anything else is a path (percent-encoded)
/// relative to `folder`. The second value is the path, if it is one.
fn read_uri(uri: &str, folder: &Path) -> Result<(Vec<u8>, Option<PathBuf>), Error> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, payload) = rest
            .split_once(";base64,")
            .ok_or_else(|| Error::Gltf("a data: URI that is not base64".into()))?;
        return Ok((base64_decode(payload)?, None));
    }
    let path = folder.join(percent_decode(uri));
    let bytes =
        std::fs::read(&path).map_err(|err| Error::Gltf(format!("{}: {err}", path.display())))?;
    Ok((bytes, Some(path)))
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'%'
            && at + 2 < bytes.len()
            && let Some(value) = std::str::from_utf8(&bytes[at + 1..at + 3])
                .ok()
                .and_then(|hex| u8::from_str_radix(hex, 16).ok())
        {
            out.push(value);
            at += 3;
        } else {
            out.push(bytes[at]);
            at += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn base64_decode(text: &str) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut bits, mut count) = (0_u32, 0);
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => {
                return Err(Error::Gltf(
                    "a data: URI with a character that is not base64".into(),
                ));
            }
        };
        bits = (bits << 6) | u32::from(value);
        count += 6;
        if count >= 8 {
            count -= 8;
            #[allow(clippy::cast_possible_truncation)] // the low 8 bits
            out.push((bits >> count) as u8);
            bits &= (1 << count) - 1;
        }
    }
    Ok(out)
}

fn build(
    document: &gltf::Document,
    buffers: &[Vec<u8>],
    folder: &Path,
) -> Result<GltfScene, Error> {
    let mut scene = GltfScene::default();

    for image in document.images() {
        let (source, mime_type) = match image.source() {
            gltf::image::Source::Uri { uri, mime_type } => match read_uri(uri, folder)? {
                (bytes, None) => (ImageSource::Embedded(bytes), mime_type.map(str::to_owned)),
                (_, Some(path)) => (ImageSource::File(path), mime_type.map(str::to_owned)),
            },
            gltf::image::Source::View { view, mime_type } => {
                let data = buffers
                    .get(view.buffer().index())
                    .and_then(|buffer| buffer.get(view.offset()..view.offset() + view.length()))
                    .ok_or_else(|| Error::Gltf("a picture outside of its buffer".into()))?;
                (
                    ImageSource::Embedded(data.to_vec()),
                    Some(mime_type.to_owned()),
                )
            }
        };
        scene.images.push(GltfImage {
            name: image.name().map(str::to_owned),
            source,
            mime_type,
        });
    }

    for material in document.materials() {
        scene.materials.push(read_material(&material));
    }

    for mesh in document.meshes() {
        let mut out = GltfMesh {
            name: mesh.name().map(str::to_owned),
            primitives: Vec::new(),
        };
        for primitive in mesh.primitives() {
            match read_primitive(&primitive, buffers)? {
                Some(read) => out.primitives.push(read),
                None => scene.skipped_primitives += 1,
            }
        }
        scene.meshes.push(out);
    }

    let root_scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    if let Some(root_scene) = root_scene {
        for node in root_scene.nodes() {
            walk(&node, IDENTITY, &mut scene.instances);
        }
    }
    Ok(scene)
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// `a` then `b`: with row vectors, that is `a * b`.
fn multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (column, value) in out_row.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[row][k] * b[k][column]).sum();
        }
    }
    out
}

fn walk(node: &gltf::Node, parent: [[f32; 4]; 4], out: &mut Vec<GltfInstance>) {
    // glTF's `matrix()` is `m[column][row]`, for column vectors: read as rows it is the matrix for
    // row vectors. The node's own matrix comes first, then its parent's.
    let local = node.transform().matrix();
    let world = multiply(&local, &parent);
    if let Some(mesh) = node.mesh() {
        out.push(GltfInstance {
            mesh: mesh.index(),
            transform: world,
            name: node.name().map(str::to_owned),
        });
    }
    for child in node.children() {
        walk(&child, world, out);
    }
}

const fn texture_ref(image: usize, tex_coord: u32) -> TextureRef {
    TextureRef { image, tex_coord }
}

fn read_material(material: &gltf::Material) -> GltfMaterial {
    let pbr = material.pbr_metallic_roughness();
    let normal = material.normal_texture();
    let occlusion = material.occlusion_texture();
    GltfMaterial {
        name: material.name().map(str::to_owned),
        base_color_factor: pbr.base_color_factor(),
        base_color: pbr
            .base_color_texture()
            .map(|info| texture_ref(info.texture().source().index(), info.tex_coord())),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        metallic_roughness: pbr
            .metallic_roughness_texture()
            .map(|info| texture_ref(info.texture().source().index(), info.tex_coord())),
        normal_scale: normal
            .as_ref()
            .map_or(1.0, gltf::material::NormalTexture::scale),
        normal: normal.map(|info| texture_ref(info.texture().source().index(), info.tex_coord())),
        occlusion_strength: occlusion
            .as_ref()
            .map_or(1.0, gltf::material::OcclusionTexture::strength),
        occlusion: occlusion
            .map(|info| texture_ref(info.texture().source().index(), info.tex_coord())),
        emissive: material
            .emissive_texture()
            .map(|info| texture_ref(info.texture().source().index(), info.tex_coord())),
        emissive_factor: material.emissive_factor(),
        alpha_mode: match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        double_sided: material.double_sided(),
    }
}

/// The triangles of a primitive, or `None` if it is not made of triangles.
fn read_primitive(
    primitive: &gltf::Primitive,
    buffers: &[Vec<u8>],
) -> Result<Option<GltfPrimitive>, Error> {
    use gltf::mesh::Mode;

    let mode = primitive.mode();
    if !matches!(
        mode,
        Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan
    ) {
        return Ok(None);
    }
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| Error::Gltf("a primitive without positions".into()))?
        .collect();
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(Iterator::collect);
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(0)
        .map(|uvs| uvs.into_f32().collect());
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(Iterator::collect);

    let count = positions.len();
    for (name, len) in [
        ("normals", normals.as_ref().map(Vec::len)),
        ("texture coordinates", uvs.as_ref().map(Vec::len)),
        ("tangents", tangents.as_ref().map(Vec::len)),
    ] {
        if len.is_some_and(|len| len != count) {
            return Err(Error::Gltf(format!(
                "a primitive with {count} positions and {} {name}",
                len.unwrap_or(0)
            )));
        }
    }

    let listed: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..u32::try_from(count).map_err(|_| Error::Gltf("too many vertices".into()))?)
            .collect(),
    };
    if listed.iter().any(|&index| index as usize >= count) {
        return Err(Error::Gltf("an index past the last vertex".into()));
    }
    let indices = match mode {
        Mode::TriangleStrip => strip_to_triangles(&listed),
        Mode::TriangleFan => fan_to_triangles(&listed),
        _ => {
            let whole = listed.len() / 3 * 3;
            listed[..whole].to_vec()
        }
    };

    Ok(Some(GltfPrimitive {
        positions,
        normals,
        uvs,
        tangents,
        indices,
        material: primitive.material().index(),
    }))
}

/// A strip as triangles: every second one is turned over so that all face the same way.
fn strip_to_triangles(strip: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for (place, window) in strip.windows(3).enumerate() {
        if place % 2 == 0 {
            out.extend_from_slice(window);
        } else {
            out.extend_from_slice(&[window[1], window[0], window[2]]);
        }
    }
    out
}

fn fan_to_triangles(fan: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    if let Some((&first, rest)) = fan.split_first() {
        for pair in rest.windows(2) {
            out.extend_from_slice(&[first, pair[0], pair[1]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base64_encode(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk
                .iter()
                .enumerate()
                .fold(0_u32, |n, (i, &b)| n | (u32::from(b) << (16 - 8 * i)));
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(TABLE[((n >> (18 - 6 * i)) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    /// A triangle with normals, a quad node that moves it, and a child that moves it again.
    fn triangle_file() -> String {
        let mut data: Vec<u8> = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        for value in [0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0] {
            data.extend(value.to_le_bytes());
        }
        for index in [0_u16, 1, 2] {
            data.extend(index.to_le_bytes());
        }
        data.extend([0, 0]); // padding
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64_encode(&data)
        );
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [
    {{"name": "parent", "translation": [10, 0, 0], "children": [1]}},
    {{"name": "child", "translation": [0, 5, 0], "mesh": 0}}
  ],
  "meshes": [{{"name": "tri", "primitives": [{{"attributes": {{"POSITION": 0, "NORMAL": 1}}, "indices": 2, "material": 0}}]}}],
  "materials": [{{"name": "red", "pbrMetallicRoughness": {{"baseColorFactor": [1, 0, 0, 1], "metallicFactor": 0.25, "roughnessFactor": 0.75}}, "doubleSided": true}}],
  "buffers": [{{"byteLength": {length}, "uri": "{uri}"}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 72, "byteLength": 6}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}}
  ]
}}"#,
            length = data.len()
        )
    }

    #[test]
    fn base64_round_trips() {
        for bytes in [&b""[..], b"a", b"ab", b"abc", b"abcd", b"\x00\xff\x10\x80"] {
            assert_eq!(base64_decode(&base64_encode(bytes)).unwrap(), bytes);
        }
        assert!(base64_decode("ab$d").is_err());
    }

    #[test]
    fn percent_encoded_paths_are_decoded() {
        assert_eq!(percent_decode("my%20texture.png"), "my texture.png");
        assert_eq!(percent_decode("plain.png"), "plain.png");
        assert_eq!(percent_decode("odd%2"), "odd%2");
    }

    #[test]
    fn a_gltf_with_one_triangle_is_read() {
        let scene = GltfScene::read_from_slice(triangle_file().as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.meshes.len(), 1);
        let primitive = &scene.meshes[0].primitives[0];
        assert_eq!(
            primitive.positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(primitive.normals.as_ref().unwrap().len(), 3);
        assert!(primitive.uvs.is_none() && primitive.tangents.is_none());
        assert_eq!(primitive.indices, [0, 1, 2]);
        assert_eq!(primitive.material, Some(0));

        let material = &scene.materials[0];
        assert_eq!(material.base_color_factor, [1.0, 0.0, 0.0, 1.0]);
        assert!((material.metallic_factor - 0.25).abs() < 1e-6);
        assert!((material.roughness_factor - 0.75).abs() < 1e-6);
        assert!(material.double_sided);
        assert_eq!(material.alpha_mode, AlphaMode::Opaque);
    }

    #[test]
    fn a_node_is_moved_by_its_own_matrix_and_then_its_parents() {
        let scene = GltfScene::read_from_slice(triangle_file().as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.instances.len(), 1);
        let instance = &scene.instances[0];
        assert_eq!(instance.mesh, 0);
        assert_eq!(instance.name.as_deref(), Some("child"));
        // Row vectors: the translation is the last row, the sum of both nodes'.
        assert_eq!(instance.transform[3], [10.0, 5.0, 0.0, 1.0]);
        assert_eq!(instance.transform[0], [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn matrices_multiply_in_the_order_of_row_vectors() {
        let mut scale = IDENTITY;
        scale[0][0] = 2.0;
        let mut translate = IDENTITY;
        translate[3][0] = 1.0;

        // Scaled first, then moved: x = 1 goes to 2, then 3.
        let both = multiply(&scale, &translate);

        assert_eq!(both[0][0], 2.0);
        assert_eq!(both[3][0], 1.0);
        // The point (1, 0, 0, 1) as a row: x' = 1 * 2 + 1 = 3.
        assert_eq!(both[0][0] + both[3][0], 3.0);
    }

    #[test]
    fn strips_and_fans_become_triangles_that_face_the_same_way() {
        assert_eq!(strip_to_triangles(&[0, 1, 2, 3]), [0, 1, 2, 2, 1, 3]);
        assert_eq!(fan_to_triangles(&[0, 1, 2, 3]), [0, 1, 2, 0, 2, 3]);
        assert_eq!(strip_to_triangles(&[0, 1]), Vec::<u32>::new());
        assert_eq!(fan_to_triangles(&[]), Vec::<u32>::new());
    }

    #[test]
    fn a_file_that_is_not_gltf_is_an_error() {
        let result = GltfScene::read_from_slice(b"not a gltf", Path::new("."));

        assert!(matches!(result, Err(Error::Gltf(_))));
    }
}
