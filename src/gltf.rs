//! A loader for glTF 2.0 files.
//!
//! It reads `.gltf` (with its `.bin` files and pictures next to it) and the single file `.glb`, and
//! gives plain data: the triangles of every mesh, the materials, where the pictures are, where each
//! node puts each mesh in the world, the scene's lights (`KHR_lights_punctual`), cameras and the
//! nodes that carry properties of their own (glTF's `extras`: Blender's custom properties), and what
//! a skinned mesh needs: the nodes as a hierarchy, the skins (the joints and their inverse bind
//! matrices), the joints and weights of every vertex, and the animations (the keys of each moved
//! node's translation, rotation and scale). Playing an animation is the caller's.
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
//! Not read (yet): morph targets (and the channels of an animation that move their weights, which
//! are counted in [`GltfScene::skipped_channels`]), sparse accessors beyond what the `gltf` crate
//! resolves itself, texture samplers and `KHR_texture_transform`. Meshes that are points or lines are
//! left out (and counted in [`GltfScene::skipped_primitives`]).

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
    /// The lights of the default scene, where their nodes put them.
    pub lights: Vec<GltfLight>,
    /// The cameras of the default scene, where their nodes put them.
    pub cameras: Vec<GltfCamera>,
    /// The nodes of the default scene that have properties of their own and are not a light or a
    /// camera: the places the scene marks for the program.
    pub markers: Vec<GltfMarker>,
    /// Every node of the file, by its number, as a hierarchy: what the joints of a skin are, and what
    /// an animation moves.
    pub nodes: Vec<GltfNode>,
    /// The skins, by their number in the file.
    pub skins: Vec<GltfSkin>,
    /// The animations, by their number in the file.
    pub animations: Vec<GltfAnimation>,
    /// How many primitives were left out because they are not made of triangles.
    pub skipped_primitives: usize,
    /// How many channels of the animations were left out because they move the weights of morph
    /// targets, which are not read.
    pub skipped_channels: usize,
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
    /// For a skinned mesh, the joints that move each vertex, as places in the `joints` of the skin of
    /// the node that uses it (`JOINTS_0`), with [`weights`](Self::weights) in the same order. A file
    /// that gives more than four (`JOINTS_1`, `WEIGHTS_1`, ...) has the four with the most weight kept.
    /// `None` if the mesh is not skinned.
    pub joints: Option<Vec<[u16; 4]>>,
    /// How much each of the joints of a vertex moves it (`WEIGHTS_0`), adding up to 1 (the file's are
    /// scaled to, as its rounding or the four that were kept may not; a vertex with no weight at all
    /// follows its first joint). `None` if the mesh is not skinned.
    pub weights: Option<Vec<[f32; 4]>>,
}

/// A use of a mesh in the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfInstance {
    /// The number of the mesh in [`GltfScene::meshes`].
    pub mesh: usize,
    /// Takes the points of the mesh to the world (row-vector convention). A skinned mesh is not moved
    /// by it (glTF says the joints place its vertices, and the node's own transform is not used).
    pub transform: [[f32; 4]; 4],
    pub name: Option<String>,
    /// The number of its node in [`GltfScene::nodes`].
    pub node: usize,
    /// The number of the skin that bends it in [`GltfScene::skins`], `None` for a mesh that is not
    /// skinned.
    pub skin: Option<usize>,
}

/// A node of the file, in the hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfNode {
    pub name: Option<String>,
    /// The node it hangs from, `None` for a root.
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Where it is, relative to its parent, before an animation moves it (a node whose file gives a
    /// matrix has it taken apart into these three).
    pub translation: [f32; 3],
    /// How it is turned, relative to its parent: a quaternion of length 1, (x, y, z, w).
    pub rotation: [f32; 4],
    /// How it is scaled, relative to its parent, on its own axes.
    pub scale: [f32; 3],
}

/// A skin: the joints that bend a mesh, and where each of them was when the mesh was bound to it.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfSkin {
    pub name: Option<String>,
    /// The joints, as numbers of [`GltfScene::nodes`]. The `joints` of a vertex are places in this list.
    pub joints: Vec<usize>,
    /// For each joint, the matrix that takes a point of the mesh into the space of the joint as it was
    /// when the mesh was bound (row-vector convention); the identity where the file gives none. A
    /// point of the mesh is where joint `j` puts it with this matrix and then the joint's matrix in the
    /// world.
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
    /// The node the file names as the root of the skeleton, if it names one.
    pub skeleton: Option<usize>,
}

/// An animation: keys for the translations, rotations and scales of some nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfAnimation {
    pub name: Option<String>,
    pub channels: Vec<GltfChannel>,
}

/// The keys of one property of one node.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfChannel {
    /// The node it moves, a number of [`GltfScene::nodes`].
    pub node: usize,
    pub property: AnimatedProperty,
    pub interpolation: Interpolation,
    /// When each key is, in seconds from the start of the animation, never going back.
    pub times: Vec<f32>,
    /// The values: one for each key, or with [`Interpolation::CubicSpline`] three for each key (the
    /// tangent coming in, the value, the tangent going out), in the order of the keys.
    pub values: ChannelValues,
}

/// What a channel moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimatedProperty {
    Translation,
    Rotation,
    Scale,
}

/// How a channel goes from one key to the next (glTF's sampler).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// In a straight line (a rotation along the shortest arc of the sphere).
    Linear,
    /// It keeps a key's value until the next key.
    Step,
    /// A cubic curve through the values with the tangents given for each key.
    CubicSpline,
}

/// The values of a channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelValues {
    /// Translations or scales.
    Vectors(Vec<[f32; 3]>),
    /// Rotations: quaternions (x, y, z, w).
    Rotations(Vec<[f32; 4]>),
}

impl ChannelValues {
    /// How many values there are.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Vectors(values) => values.len(),
            Self::Rotations(values) => values.len(),
        }
    }

    /// Whether there are none.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A light of the scene (glTF's `KHR_lights_punctual`), where its node puts it.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfLight {
    /// The node's name (or the light's, if the node has none).
    pub name: Option<String>,
    pub kind: GltfLightKind,
    /// Linear red, green and blue.
    pub color: [f32; 3],
    /// How bright it is, in glTF's units: candela (lumen per steradian) for a point or a spot light,
    /// lux (lumen per square metre) for a directional one.
    pub intensity: f32,
    /// How far it reaches, if the file says (it gives no light from there on); `None`: as far as its
    /// intensity lights anything.
    pub range: Option<f32>,
    /// Where it is in the world (nothing for a directional light).
    pub position: [f32; 3],
    /// The way it shines in the world (its node's -z), of length 1 (nothing for a point light).
    pub direction: [f32; 3],
    /// The properties of its node (see [`GltfMarker`]).
    pub properties: Vec<(String, ExtraValue)>,
}

/// What kind of light a [`GltfLight`] is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GltfLightKind {
    /// Infinitely far away: only its direction counts.
    Directional,
    /// From a point, the same every way.
    Point,
    /// From a point, in a cone: full inside `inner_cone_angle`, nothing outside `outer_cone_angle`
    /// (both from the middle of the cone, in radians).
    Spot {
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    },
}

/// A camera of the scene, where its node puts it.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfCamera {
    /// The node's name (or the camera's, if the node has none).
    pub name: Option<String>,
    /// Where it is in the world.
    pub position: [f32; 3],
    /// The way it looks (its node's -z), of length 1.
    pub forward: [f32; 3],
    /// Its up (its node's +y), of length 1.
    pub up: [f32; 3],
    /// The vertical field of view in radians, for a perspective camera (`None`: orthographic).
    pub yfov: Option<f32>,
    pub znear: f32,
    /// The far plane, if it has one (a perspective camera may have none).
    pub zfar: Option<f32>,
}

/// A node with properties of its own that is not a light or a camera: a place the scene marks.
///
/// The properties are glTF's `extras`, which Blender writes from an object's custom properties: what
/// the scene says to the program about the place (an effect that goes there, something that starts
/// there). A node that has a mesh is in [`GltfScene::instances`] too.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfMarker {
    pub name: Option<String>,
    /// The node's matrix in the world (row vectors).
    pub transform: [[f32; 4]; 4],
    /// The properties, in the order of the file.
    pub properties: Vec<(String, ExtraValue)>,
}

impl GltfMarker {
    /// Where the node is in the world.
    #[must_use]
    pub const fn position(&self) -> [f32; 3] {
        [
            self.transform[3][0],
            self.transform[3][1],
            self.transform[3][2],
        ]
    }

    /// The value of the property `key`, if the node has it.
    #[must_use]
    pub fn property(&self, key: &str) -> Option<&ExtraValue> {
        property(&self.properties, key)
    }
}

/// The value of a property of a node: what JSON has.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtraValue {
    Number(f64),
    Text(String),
    Bool(bool),
    /// Anything else (a list, an object, null), as its JSON.
    Other(String),
}

impl ExtraValue {
    /// The number, if it is one.
    #[must_use]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// The text, if it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The truth value, if it is one.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

/// The value of the property `key` among `properties`, if it is there.
#[must_use]
pub fn property<'a>(properties: &'a [(String, ExtraValue)], key: &str) -> Option<&'a ExtraValue> {
    properties
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
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

    scene.nodes = read_nodes(document);
    for skin in document.skins() {
        scene.skins.push(read_skin(&skin, buffers)?);
    }
    for animation in document.animations() {
        let (read, skipped) = read_animation(&animation, buffers)?;
        scene.animations.push(read);
        scene.skipped_channels += skipped;
    }

    let root_scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    if let Some(root_scene) = root_scene {
        for node in root_scene.nodes() {
            walk(&node, IDENTITY, &mut scene);
        }
    }
    check_skinned_instances(&scene)?;
    Ok(scene)
}

/// Every node of the document, with its parent found from the lists of children.
fn read_nodes(document: &gltf::Document) -> Vec<GltfNode> {
    let mut nodes: Vec<GltfNode> = document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            GltfNode {
                name: node.name().map(str::to_owned),
                parent: None,
                children: node.children().map(|child| child.index()).collect(),
                translation,
                rotation,
                scale,
            }
        })
        .collect();
    for index in 0..nodes.len() {
        for child in nodes[index].children.clone() {
            if let Some(child) = nodes.get_mut(child) {
                child.parent = Some(index);
            }
        }
    }
    nodes
}

/// A skin, with an inverse bind matrix for each joint (the identity if the file gives none).
fn read_skin(skin: &gltf::Skin, buffers: &[Vec<u8>]) -> Result<GltfSkin, Error> {
    let joints: Vec<usize> = skin.joints().map(|joint| joint.index()).collect();
    let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    // glTF's matrices are `m[column][row]` for column vectors: read as rows they are the matrices for
    // row vectors, as a node's are.
    let inverse_bind_matrices: Vec<[[f32; 4]; 4]> = reader
        .read_inverse_bind_matrices()
        .map_or_else(|| vec![IDENTITY; joints.len()], Iterator::collect);
    if inverse_bind_matrices.len() != joints.len() {
        return Err(Error::Gltf(format!(
            "skin {} has {} joints and {} inverse bind matrices",
            skin.index(),
            joints.len(),
            inverse_bind_matrices.len()
        )));
    }
    Ok(GltfSkin {
        name: skin.name().map(str::to_owned),
        joints,
        inverse_bind_matrices,
        skeleton: skin.skeleton().map(|node| node.index()),
    })
}

/// An animation's channels, and how many were left out (the ones that move the weights of morph
/// targets).
fn read_animation(
    animation: &gltf::Animation,
    buffers: &[Vec<u8>],
) -> Result<(GltfAnimation, usize), Error> {
    use gltf::animation::util::ReadOutputs;

    let mut channels = Vec::new();
    let mut skipped = 0;
    for channel in animation.channels() {
        let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
        let (property, values) = match reader.read_outputs() {
            Some(ReadOutputs::Translations(values)) => (
                AnimatedProperty::Translation,
                ChannelValues::Vectors(values.collect()),
            ),
            Some(ReadOutputs::Rotations(values)) => (
                AnimatedProperty::Rotation,
                ChannelValues::Rotations(values.into_f32().collect()),
            ),
            Some(ReadOutputs::Scales(values)) => (
                AnimatedProperty::Scale,
                ChannelValues::Vectors(values.collect()),
            ),
            Some(ReadOutputs::MorphTargetWeights(_)) => {
                skipped += 1;
                continue;
            }
            None => {
                return Err(Error::Gltf(format!(
                    "a channel of animation {} has no values",
                    animation.index()
                )));
            }
        };
        let times: Vec<f32> = reader
            .read_inputs()
            .ok_or_else(|| {
                Error::Gltf(format!(
                    "a channel of animation {} has no times",
                    animation.index()
                ))
            })?
            .collect();
        let interpolation = match channel.sampler().interpolation() {
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        };
        let per_key = if interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if times.is_empty() || values.len() != times.len() * per_key {
            return Err(Error::Gltf(format!(
                "a channel of animation {} has {} times and {} values",
                animation.index(),
                times.len(),
                values.len()
            )));
        }
        if times.iter().any(|time| !time.is_finite())
            || times.windows(2).any(|pair| pair[1] < pair[0])
        {
            return Err(Error::Gltf(format!(
                "a channel of animation {} has times that go back or are not numbers",
                animation.index()
            )));
        }
        channels.push(GltfChannel {
            node: channel.target().node().index(),
            property,
            interpolation,
            times,
            values,
        });
    }
    Ok((
        GltfAnimation {
            name: animation.name().map(str::to_owned),
            channels,
        },
        skipped,
    ))
}

/// Every joint a vertex of a skinned instance names has to be one of its skin's.
fn check_skinned_instances(scene: &GltfScene) -> Result<(), Error> {
    for instance in &scene.instances {
        let Some(skin) = instance.skin else {
            continue;
        };
        let count = scene.skins.get(skin).map_or(0, |skin| skin.joints.len());
        let Some(mesh) = scene.meshes.get(instance.mesh) else {
            continue;
        };
        for primitive in &mesh.primitives {
            let Some(joints) = &primitive.joints else {
                return Err(Error::Gltf(format!(
                    "mesh {} is skinned by skin {skin}, and a part of it has no joints",
                    instance.mesh
                )));
            };
            if let Some(joint) = joints
                .iter()
                .flatten()
                .find(|&&joint| usize::from(joint) >= count)
            {
                return Err(Error::Gltf(format!(
                    "a vertex of mesh {} names joint {joint}, and skin {skin} has {count}",
                    instance.mesh
                )));
            }
        }
    }
    Ok(())
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

fn walk(node: &gltf::Node, parent: [[f32; 4]; 4], out: &mut GltfScene) {
    // glTF's `matrix()` is `m[column][row]`, for column vectors: read as rows it is the matrix for
    // row vectors. The node's own matrix comes first, then its parent's.
    let local = node.transform().matrix();
    let world = multiply(&local, &parent);
    if let Some(mesh) = node.mesh() {
        out.instances.push(GltfInstance {
            mesh: mesh.index(),
            transform: world,
            name: node.name().map(str::to_owned),
            node: node.index(),
            skin: node.skin().map(|skin| skin.index()),
        });
    }
    // The rows of the matrix are where the node's axes go in the world, and the last is where it is.
    let position = [world[3][0], world[3][1], world[3][2]];
    let axis = |row: usize, sign: f32| {
        normalized([
            sign * world[row][0],
            sign * world[row][1],
            sign * world[row][2],
        ])
    };
    let properties = properties(node.extras());
    if let Some(light) = node.light() {
        out.lights.push(GltfLight {
            name: node.name().or_else(|| light.name()).map(str::to_owned),
            kind: match light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => GltfLightKind::Directional,
                gltf::khr_lights_punctual::Kind::Point => GltfLightKind::Point,
                gltf::khr_lights_punctual::Kind::Spot {
                    inner_cone_angle,
                    outer_cone_angle,
                } => GltfLightKind::Spot {
                    inner_cone_angle,
                    outer_cone_angle,
                },
            },
            color: light.color(),
            intensity: light.intensity(),
            range: light.range(),
            position,
            direction: axis(2, -1.0),
            properties,
        });
    } else if let Some(camera) = node.camera() {
        let (yfov, znear, zfar) = match camera.projection() {
            gltf::camera::Projection::Perspective(p) => (Some(p.yfov()), p.znear(), p.zfar()),
            gltf::camera::Projection::Orthographic(o) => (None, o.znear(), Some(o.zfar())),
        };
        out.cameras.push(GltfCamera {
            name: node.name().or_else(|| camera.name()).map(str::to_owned),
            position,
            forward: axis(2, -1.0),
            up: axis(1, 1.0),
            yfov,
            znear,
            zfar,
        });
    } else if !properties.is_empty() {
        out.markers.push(GltfMarker {
            name: node.name().map(str::to_owned),
            transform: world,
            properties,
        });
    }
    for child in node.children() {
        walk(&child, world, out);
    }
}

/// `v` made of length 1 (as it is if it has no length).
fn normalized(v: [f32; 3]) -> [f32; 3] {
    let length = v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1])).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        v
    }
}

/// The properties of a node (its `extras`, a JSON object), in the order of the file; none if it has
/// none or they are not an object.
fn properties(extras: &gltf::json::Extras) -> Vec<(String, ExtraValue)> {
    use gltf::json::Value;
    let Some(raw) = extras else {
        return Vec::new();
    };
    match gltf::json::deserialize::from_str::<Value>(raw.get()) {
        Ok(Value::Object(map)) => map
            .into_iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::Number(number) => {
                        ExtraValue::Number(number.as_f64().unwrap_or(f64::NAN))
                    }
                    Value::String(text) => ExtraValue::Text(text),
                    Value::Bool(truth) => ExtraValue::Bool(truth),
                    other => ExtraValue::Other(other.to_string()),
                };
                (key, value)
            })
            .collect(),
        _ => Vec::new(),
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

    let (joints, weights) = read_influences(&reader, count)?;

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
        joints,
        weights,
    }))
}

/// The joints and weights of the `count` vertices of a primitive, from every set the file has
/// (`JOINTS_0` with `WEIGHTS_0`, then `JOINTS_1` with `WEIGHTS_1`, ...): for each vertex the four with
/// the most weight (a joint named twice has its weights added), scaled to add up to 1. `None` for both
/// if it has none.
#[allow(
    clippy::type_complexity,
    reason = "the two lists, as the primitive has them"
)]
fn read_influences<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    count: usize,
) -> Result<(Option<Vec<[u16; 4]>>, Option<Vec<[f32; 4]>>), Error>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let mut sets: Vec<(Vec<[u16; 4]>, Vec<[f32; 4]>)> = Vec::new();
    for set in 0.. {
        let (joints, weights) = (reader.read_joints(set), reader.read_weights(set));
        let (joints, weights) = match (joints, weights) {
            (None, None) => break,
            (Some(joints), Some(weights)) => (
                joints.into_u16().collect::<Vec<_>>(),
                weights.into_f32().collect::<Vec<_>>(),
            ),
            _ => {
                return Err(Error::Gltf(format!(
                    "a primitive has JOINTS_{set} or WEIGHTS_{set} without the other"
                )));
            }
        };
        if joints.len() != count || weights.len() != count {
            return Err(Error::Gltf(format!(
                "a primitive with {count} positions has {} joints and {} weights in set {set}",
                joints.len(),
                weights.len()
            )));
        }
        sets.push((joints, weights));
    }
    if sets.is_empty() {
        return Ok((None, None));
    }
    let mut joints = Vec::with_capacity(count);
    let mut weights = Vec::with_capacity(count);
    for vertex in 0..count {
        let mut influences: Vec<(u16, f32)> = Vec::with_capacity(4 * sets.len());
        for (set_joints, set_weights) in &sets {
            for (joint, weight) in set_joints[vertex].into_iter().zip(set_weights[vertex]) {
                if weight.is_nan() || weight <= 0.0 {
                    continue;
                }
                match influences
                    .iter_mut()
                    .find(|(existing, _)| *existing == joint)
                {
                    Some((_, total)) => *total += weight,
                    None => influences.push((joint, weight)),
                }
            }
        }
        let (vertex_joints, vertex_weights) = strongest_four(&mut influences);
        joints.push(vertex_joints);
        weights.push(vertex_weights);
    }
    Ok((Some(joints), Some(weights)))
}

/// The four influences with the most weight (the heaviest first), their weights scaled to add up to 1;
/// with none, the first joint, wholly.
fn strongest_four(influences: &mut [(u16, f32)]) -> ([u16; 4], [f32; 4]) {
    influences.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let kept = &influences[..influences.len().min(4)];
    let total: f32 = kept.iter().map(|(_, weight)| weight).sum();
    if total.is_nan() || total <= 0.0 {
        return ([0; 4], [1.0, 0.0, 0.0, 0.0]);
    }
    let mut joints = [0_u16; 4];
    let mut weights = [0.0_f32; 4];
    for (place, &(joint, weight)) in kept.iter().enumerate() {
        joints[place] = joint;
        weights[place] = weight / total;
    }
    (joints, weights)
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

    /// A spot light pointing down, a camera, and a node that marks a place with its properties.
    const LIGHT_CAMERA_MARKER: &str = r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0, 1, 2]}],
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {"KHR_lights_punctual": {"lights": [
    {"type": "spot", "color": [1, 0.5, 0.25], "intensity": 100, "range": 20,
     "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.5}}
  ]}},
  "cameras": [{"type": "perspective", "perspective": {"yfov": 0.8, "znear": 0.1, "zfar": 500}}],
  "nodes": [
    {"name": "lamp", "translation": [1, 2, 3], "rotation": [-0.70710678, 0, 0, 0.70710678],
     "extensions": {"KHR_lights_punctual": {"light": 0}}, "extras": {"shadows": "auto"}},
    {"name": "eye", "translation": [0, 1, 5], "camera": 0},
    {"name": "FX", "translation": [4, 0, -2], "extras": {"effect": "sparks", "rate": 0.5, "heat": true, "list": [1]}}
  ]
}"#;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn the_lights_are_read_where_their_nodes_put_them() {
        let scene =
            GltfScene::read_from_slice(LIGHT_CAMERA_MARKER.as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.lights.len(), 1);
        let light = &scene.lights[0];
        assert_eq!(light.name.as_deref(), Some("lamp"));
        assert_eq!(
            light.kind,
            GltfLightKind::Spot {
                inner_cone_angle: 0.2,
                outer_cone_angle: 0.5
            }
        );
        assert_eq!(light.color, [1.0, 0.5, 0.25]);
        assert!((light.intensity - 100.0).abs() < 1e-6);
        assert_eq!(light.range, Some(20.0));
        assert!(close(light.position, [1.0, 2.0, 3.0]));
        // Turned a quarter about x: its -z points down.
        assert!(
            close(light.direction, [0.0, -1.0, 0.0]),
            "{:?}",
            light.direction
        );
        assert_eq!(
            property(&light.properties, "shadows").and_then(ExtraValue::as_str),
            Some("auto")
        );
    }

    #[test]
    fn the_cameras_are_read_where_their_nodes_put_them() {
        let scene =
            GltfScene::read_from_slice(LIGHT_CAMERA_MARKER.as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.cameras.len(), 1);
        let camera = &scene.cameras[0];
        assert_eq!(camera.name.as_deref(), Some("eye"));
        assert!(close(camera.position, [0.0, 1.0, 5.0]));
        assert!(close(camera.forward, [0.0, 0.0, -1.0]));
        assert!(close(camera.up, [0.0, 1.0, 0.0]));
        assert_eq!(camera.yfov, Some(0.8));
        assert_eq!(camera.zfar, Some(500.0));
        assert!((camera.znear - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_node_with_properties_is_a_marker_and_its_values_keep_their_kind() {
        let scene =
            GltfScene::read_from_slice(LIGHT_CAMERA_MARKER.as_bytes(), Path::new(".")).unwrap();

        // The light's node has properties too, but it is a light: the marker is the one node that is
        // neither.
        assert_eq!(scene.markers.len(), 1);
        let marker = &scene.markers[0];
        assert_eq!(marker.name.as_deref(), Some("FX"));
        assert!(close(marker.position(), [4.0, 0.0, -2.0]));
        assert_eq!(
            marker.property("effect").and_then(ExtraValue::as_str),
            Some("sparks")
        );
        assert_eq!(
            marker.property("rate").and_then(ExtraValue::as_f64),
            Some(0.5)
        );
        assert_eq!(
            marker.property("heat").and_then(ExtraValue::as_bool),
            Some(true)
        );
        assert_eq!(
            marker.property("list"),
            Some(&ExtraValue::Other("[1]".to_owned()))
        );
        assert_eq!(marker.property("missing"), None);
        // No mesh anywhere.
        assert_eq!(scene.instances.len(), 0);
    }

    #[test]
    fn a_scene_without_lights_cameras_or_properties_has_none() {
        let scene = GltfScene::read_from_slice(triangle_file().as_bytes(), Path::new(".")).unwrap();

        assert!(scene.lights.is_empty() && scene.cameras.is_empty() && scene.markers.is_empty());
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

    /// The buffer of a test file being made, and the views and accessors of what is put in it.
    #[derive(Default)]
    struct Blob {
        data: Vec<u8>,
        views: Vec<String>,
        accessors: Vec<String>,
    }

    impl Blob {
        /// Puts `bytes` in (from a multiple of 4), with a view and an accessor of `count` elements of `kind`
        /// (`SCALAR`, `VEC4`, ...) of the glTF component type `component`; `extra` is more of the accessor's
        /// JSON (`min` and `max`). Returns the number of the accessor.
        fn add(
            &mut self,
            bytes: &[u8],
            (component, normalized): (u32, bool),
            (count, kind): (usize, &str),
            extra: &str,
        ) -> usize {
            while !self.data.len().is_multiple_of(4) {
                self.data.push(0);
            }
            let offset = self.data.len();
            self.data.extend_from_slice(bytes);
            self.views.push(format!(
                r#"{{"buffer": 0, "byteOffset": {offset}, "byteLength": {}}}"#,
                bytes.len()
            ));
            self.accessors.push(format!(
                r#"{{"bufferView": {}, "componentType": {component}, "normalized": {normalized}, "count": {count}, "type": "{kind}"{extra}}}"#,
                self.views.len() - 1
            ));
            self.accessors.len() - 1
        }

        fn floats(&mut self, values: &[f32], (count, kind): (usize, &str), extra: &str) -> usize {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            self.add(&bytes, (5126, false), (count, kind), extra)
        }

        /// The `buffers`, `bufferViews` and `accessors` of the file.
        fn json(&self) -> String {
            format!(
                r#""buffers": [{{"byteLength": {}, "uri": "data:application/octet-stream;base64,{}"}}],
"bufferViews": [{}],
"accessors": [{}]"#,
                self.data.len(),
                base64_encode(&self.data),
                self.views.join(", "),
                self.accessors.join(", ")
            )
        }
    }

    /// A column-major glTF matrix (as a file has it) that moves a point by `(x, y, z)`.
    fn translation_matrix(x: f32, y: f32, z: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
        ]
    }

    /// A skinned triangle: a mesh node and two joints (`root`, and `tip` under it) under a node that has a
    /// matrix. Two sets of joints and weights (the second in bytes that stand for 0 to 1), inverse bind
    /// matrices, and two animations: `Wave` (the tip turns along a line, the root steps, the tip's scale
    /// is a cubic spline) and `Turn` (a rotation in 16 bit numbers that stand for -1 to 1). `joint` is the
    /// joint the third vertex names first (1 is a joint of the skin; 2 is not).
    #[allow(clippy::too_many_lines, reason = "the buffer and the JSON of one file")]
    fn skinned_file(joint: u8) -> String {
        let mut blob = Blob::default();
        let positions = blob.floats(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            (3, "VEC3"),
            r#", "min": [0, 0, 0], "max": [1, 1, 0]"#,
        );
        // Vertex 0: joints 0 and 1 by halves, in two sets (0.4 and 0.3, then 0.1 and 0.2: added, half and half).
        // Vertex 1: joint 1 alone. Vertex 2: `joint` with 0.25 (and nothing else): scaled to 1.
        let joints_0 = blob.add(
            &[0, 1, 0, 0, 1, 0, 0, 0, joint, 0, 0, 0],
            (5121, false),
            (3, "VEC4"),
            "",
        );
        let weights_0 = blob.floats(
            &[0.4, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
            (3, "VEC4"),
            "",
        );
        let joints_1 = blob.add(
            &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (5121, false),
            (3, "VEC4"),
            "",
        );
        // 0.1 and 0.2 as bytes of 255ths (26 and 51).
        let weights_1 = blob.add(
            &[26, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (5121, true),
            (3, "VEC4"),
            "",
        );
        let indices = blob.add(
            &[0_u16, 1, 2]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            (5123, false),
            (3, "SCALAR"),
            "",
        );
        let mut matrices = translation_matrix(0.0, -1.0, 0.0).to_vec();
        matrices.extend(translation_matrix(0.0, -3.0, 0.0));
        let inverse_bind = blob.floats(&matrices, (2, "MAT4"), "");
        let two_times = blob.floats(&[0.0, 1.0], (2, "SCALAR"), r#", "min": [0], "max": [1]"#);
        let three_times = blob.floats(
            &[0.0, 0.5, 1.0],
            (3, "SCALAR"),
            r#", "min": [0], "max": [1]"#,
        );
        let one_time = blob.floats(&[0.0], (1, "SCALAR"), r#", "min": [0], "max": [0]"#);
        let turns = blob.floats(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2],
            (2, "VEC4"),
            "",
        );
        let steps = blob.floats(
            &[0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.0, 0.0],
            (3, "VEC3"),
            "",
        );
        // Two keys, each the tangent in, the value and the tangent out.
        let spline = blob.floats(
            &[
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 0.0,
                0.0, 0.0,
            ],
            (6, "VEC3"),
            "",
        );
        // Half a turn round y: (0, 1, 0, 0) as 16 bit numbers of 32767ths.
        let half_turn = blob.add(
            &[0_i16, 32767, 0, 0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            (5122, true),
            (1, "VEC4"),
            "",
        );
        format!(
            r#"{{
"asset": {{"version": "2.0"}},
"scene": 0,
"scenes": [{{"nodes": [0, 3]}}],
"nodes": [
  {{"name": "body", "mesh": 0, "skin": 0, "translation": [100, 0, 0]}},
  {{"name": "root", "translation": [0, 1, 0], "children": [2]}},
  {{"name": "tip", "translation": [0, 2, 0]}},
  {{"name": "armature", "matrix": [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 5, 0, 0, 1], "children": [1]}}
],
"meshes": [{{"primitives": [{{"attributes": {{"POSITION": {positions}, "JOINTS_0": {joints_0}, "WEIGHTS_0": {weights_0}, "JOINTS_1": {joints_1}, "WEIGHTS_1": {weights_1}}}, "indices": {indices}}}]}}],
"skins": [{{"name": "rig", "joints": [1, 2], "inverseBindMatrices": {inverse_bind}, "skeleton": 3}}],
"animations": [
  {{"name": "Wave",
   "samplers": [
     {{"input": {two_times}, "output": {turns}, "interpolation": "LINEAR"}},
     {{"input": {three_times}, "output": {steps}, "interpolation": "STEP"}},
     {{"input": {two_times}, "output": {spline}, "interpolation": "CUBICSPLINE"}}
   ],
   "channels": [
     {{"sampler": 0, "target": {{"node": 2, "path": "rotation"}}}},
     {{"sampler": 1, "target": {{"node": 1, "path": "translation"}}}},
     {{"sampler": 2, "target": {{"node": 2, "path": "scale"}}}}
   ]}},
  {{"name": "Turn",
   "samplers": [{{"input": {one_time}, "output": {half_turn}, "interpolation": "STEP"}}],
   "channels": [{{"sampler": 0, "target": {{"node": 1, "path": "rotation"}}}}]}}
],
{}
}}"#,
            blob.json()
        )
    }

    #[test]
    fn the_nodes_are_a_hierarchy_with_their_own_translation_rotation_and_scale() {
        let scene = GltfScene::read_from_slice(skinned_file(1).as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.nodes.len(), 4);
        let names: Vec<Option<&str>> = scene
            .nodes
            .iter()
            .map(|node| node.name.as_deref())
            .collect();
        assert_eq!(
            names,
            [Some("body"), Some("root"), Some("tip"), Some("armature")]
        );
        let parents: Vec<Option<usize>> = scene.nodes.iter().map(|node| node.parent).collect();
        assert_eq!(parents, [None, Some(3), Some(1), None]);
        assert_eq!(scene.nodes[1].children, [2]);
        assert_eq!(scene.nodes[2].translation, [0.0, 2.0, 0.0]);
        assert_eq!(scene.nodes[2].rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(scene.nodes[2].scale, [1.0, 1.0, 1.0]);
        // A node with a matrix has it taken apart.
        let armature = &scene.nodes[3];
        assert_eq!(armature.translation, [5.0, 0.0, 0.0]);
        assert_eq!(armature.scale, [2.0, 2.0, 2.0]);
        assert!(close(
            [
                armature.rotation[0],
                armature.rotation[1],
                armature.rotation[2]
            ],
            [0.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn a_skin_has_its_joints_its_inverse_bind_matrices_and_its_skeleton() {
        let scene = GltfScene::read_from_slice(skinned_file(1).as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.skins.len(), 1);
        let skin = &scene.skins[0];
        assert_eq!(skin.name.as_deref(), Some("rig"));
        assert_eq!(skin.joints, [1, 2]);
        assert_eq!(skin.skeleton, Some(3));
        // Row vectors: the translation of a matrix is its last row.
        assert_eq!(skin.inverse_bind_matrices[0][3], [0.0, -1.0, 0.0, 1.0]);
        assert_eq!(skin.inverse_bind_matrices[1][3], [0.0, -3.0, 0.0, 1.0]);
        assert_eq!(skin.inverse_bind_matrices[1][1], [0.0, 1.0, 0.0, 0.0]);
        // The instance says its node and its skin.
        assert_eq!(scene.instances.len(), 1);
        assert_eq!(
            (scene.instances[0].node, scene.instances[0].skin),
            (0, Some(0))
        );
    }

    #[test]
    fn the_joints_and_weights_of_every_set_are_four_that_add_up_to_one() {
        let scene = GltfScene::read_from_slice(skinned_file(1).as_bytes(), Path::new(".")).unwrap();
        let primitive = &scene.meshes[0].primitives[0];
        let joints = primitive.joints.as_ref().unwrap();
        let weights = primitive.weights.as_ref().unwrap();

        // Vertex 0: 0.4 + 0.1 for joint 0 and 0.3 + 0.2 for joint 1 (a byte of 26 and 51, out of 255).
        assert_eq!(joints[0][..2], [0, 1]);
        assert!(
            (weights[0][0] - 0.5).abs() < 0.01 && (weights[0][1] - 0.5).abs() < 0.01,
            "{:?}",
            weights[0]
        );
        // Vertex 1: joint 1 wholly. Vertex 2: one weight of 0.25, scaled to 1.
        assert_eq!((joints[1][0], weights[1]), (1, [1.0, 0.0, 0.0, 0.0]));
        assert_eq!((joints[2][0], weights[2]), (1, [1.0, 0.0, 0.0, 0.0]));
        for weight in weights {
            assert!((weight.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        // A mesh that is not skinned has neither.
        let plain = GltfScene::read_from_slice(triangle_file().as_bytes(), Path::new(".")).unwrap();
        let primitive = &plain.meshes[0].primitives[0];
        assert!(primitive.joints.is_none() && primitive.weights.is_none());
        assert_eq!(plain.instances[0].skin, None);
        assert!(plain.skins.is_empty() && plain.animations.is_empty());
    }

    #[test]
    fn of_more_than_four_influences_the_four_heaviest_are_kept() {
        let mut influences = vec![(7, 0.05), (1, 0.3), (2, 0.1), (3, 0.25), (4, 0.2), (5, 0.1)];
        let (joints, weights) = strongest_four(&mut influences);
        assert_eq!(joints, [1, 3, 4, 2]);
        let total = 0.3 + 0.25 + 0.2 + 0.1;
        for (weight, expected) in weights.iter().zip([0.3, 0.25, 0.2, 0.1]) {
            assert!((weight - expected / total).abs() < 1e-6);
        }
        // None at all: the first joint, wholly.
        assert_eq!(strongest_four(&mut []), ([0; 4], [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn the_channels_of_an_animation_have_their_node_property_interpolation_times_and_values() {
        let scene = GltfScene::read_from_slice(skinned_file(1).as_bytes(), Path::new(".")).unwrap();

        assert_eq!(scene.animations.len(), 2);
        assert_eq!(scene.skipped_channels, 0);
        let wave = &scene.animations[0];
        assert_eq!(wave.name.as_deref(), Some("Wave"));
        assert_eq!(wave.channels.len(), 3);

        let turn = &wave.channels[0];
        assert_eq!(
            (turn.node, turn.property, turn.interpolation),
            (2, AnimatedProperty::Rotation, Interpolation::Linear)
        );
        assert_eq!(turn.times, [0.0, 1.0]);
        assert_eq!(
            turn.values,
            ChannelValues::Rotations(vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2]
            ])
        );

        let step = &wave.channels[1];
        assert_eq!(
            (step.node, step.property, step.interpolation),
            (1, AnimatedProperty::Translation, Interpolation::Step)
        );
        assert_eq!(step.times, [0.0, 0.5, 1.0]);
        assert_eq!(
            step.values,
            ChannelValues::Vectors(vec![[0.0, 1.0, 0.0], [0.0, 1.5, 0.0], [0.0, 2.0, 0.0]])
        );

        // A cubic spline has three values for each key.
        let spline = &wave.channels[2];
        assert_eq!(
            (spline.property, spline.interpolation),
            (AnimatedProperty::Scale, Interpolation::CubicSpline)
        );
        assert_eq!((spline.times.len(), spline.values.len()), (2, 6));

        // A rotation in 16 bit numbers that stand for -1 to 1.
        let half = &scene.animations[1].channels[0];
        let ChannelValues::Rotations(rotations) = &half.values else {
            panic!("a rotation channel has rotations");
        };
        assert!(
            (rotations[0][1] - 1.0).abs() < 1e-6 && rotations[0][0].abs() < 1e-6,
            "{rotations:?}"
        );
    }

    #[test]
    fn a_vertex_that_names_a_joint_its_skin_does_not_have_is_an_error() {
        let Err(Error::Gltf(message)) =
            GltfScene::read_from_slice(skinned_file(2).as_bytes(), Path::new("."))
        else {
            panic!("joint 2 of a skin of 2 is an error");
        };
        assert!(message.contains("names joint 2"), "{message}");
    }

    #[test]
    fn times_that_go_back_are_an_error() {
        // A node moved by three keys whose times are 0, 1 and then 0.5.
        let mut blob = Blob::default();
        let times = blob.floats(
            &[0.0, 1.0, 0.5],
            (3, "SCALAR"),
            r#", "min": [0], "max": [1]"#,
        );
        let values = blob.floats(&[0.0; 9], (3, "VEC3"), "");
        let file = format!(
            r#"{{"asset": {{"version": "2.0"}}, "scene": 0, "scenes": [{{"nodes": [0]}}],
"nodes": [{{"name": "moved"}}],
"animations": [{{"samplers": [{{"input": {times}, "output": {values}}}], "channels": [{{"sampler": 0, "target": {{"node": 0, "path": "translation"}}}}]}}],
{}}}"#,
            blob.json()
        );
        let Err(Error::Gltf(message)) = GltfScene::read_from_slice(file.as_bytes(), Path::new("."))
        else {
            panic!("times that go back are an error");
        };
        assert!(message.contains("go back"), "{message}");
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
