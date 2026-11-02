//! A loader for glTF 2.0 files.
//!
//! It reads `.gltf` (with its `.bin` files and pictures next to it) and the single file `.glb`, and
//! gives plain data: the triangles of every mesh, the materials, where the pictures are, where each
//! node puts each mesh in the world (with what makes a material a glass: `KHR_materials_transmission`,
//! `KHR_materials_ior` and `KHR_materials_volume`; `KHR_materials_specular`; how strong what it gives off is,
//! `KHR_materials_emissive_strength`; whether it is lit at all, `KHR_materials_unlit`; and the clear layer over it, the
//! sheen of cloth and the stretched highlight of brushed metal, `KHR_materials_clearcoat`, `KHR_materials_sheen` and
//! `KHR_materials_anisotropy`, which the `gltf` crate does not know and this reads from the material's JSON), the scene's
//! lights
//! (`KHR_lights_punctual`), cameras and the
//! nodes that carry properties of their own (glTF's `extras`: Blender's custom properties), the numbers a
//! program gives the vertices for its own use (glTF's attributes whose names start with an underscore), and what
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
//! resolves itself, and `KHR_texture_transform`. Meshes that are points or lines are
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
    /// The files it was read from, the pictures apart: the `.gltf` or `.glb`, and the buffers it refers to.
    pub files: Vec<PathBuf>,
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
    /// The sets after the first (`TEXCOORD_1`, `TEXCOORD_2`, ...: set `n` is `extra_uvs[n - 1]`), the same way round;
    /// empty if the file has only the first. A map says which set it reads ([`TextureRef::tex_coord`]); a file often
    /// has copies of the first set here, which the caller may tell apart.
    pub extra_uvs: Vec<Vec<[f32; 2]>>,
    /// Tangent (xyz) and the handedness of the bitangent (w, 1 or -1).
    pub tangents: Option<Vec<[f32; 4]>>,
    /// The first set of vertex colours (`COLOR_0`), red, green, blue and alpha from 0 to 1 (an RGB
    /// colour has an alpha of 1), linear as the file has them: a colour, which glTF multiplies into the
    /// base colour (numbers of a program's own are in [`attributes`](Self::attributes)). `None` if the
    /// file has none.
    pub colors: Option<Vec<[f32; 4]>>,
    /// The attributes of the application (glTF's attributes whose names start with an underscore, as
    /// `_WIND`): numbers a program gives the vertices for its own use, which glTF gives no meaning. In
    /// the order of their names; empty if the file has none. [`attribute`](Self::attribute) finds one.
    pub attributes: Vec<GltfAttribute>,
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

impl GltfPrimitive {
    /// The attribute of the application named `name`, the underscore included (`"_WIND"`), if the
    /// primitive has it.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<&GltfAttribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
    }
}

/// An attribute of the application on the vertices of a primitive: up to four numbers for each.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfAttribute {
    /// Its name as the file has it, the underscore included (`_WIND`).
    pub name: String,
    /// How many numbers each vertex has: 1 to 4 (glTF's `SCALAR` to `VEC4`; an attribute of
    /// matrices is left out).
    pub components: u8,
    /// The numbers of each vertex, 0 past [`components`](Self::components). Integers the file marks
    /// normalised are as glTF reads them (unsigned from 0 to 1, signed from -1 to 1), others are their
    /// values.
    pub values: Vec<[f32; 4]>,
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

/// A picture a material uses, which texture coordinates it is read with, and how (its texture's sampler).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRef {
    /// The number of the picture in [`GltfScene::images`].
    pub image: usize,
    /// Which set of texture coordinates (0 is the first).
    pub tex_coord: u32,
    /// How the picture is read: its texture's sampler (glTF's default where the texture names none).
    pub sampler: GltfSampler,
}

/// How a texture is read (glTF's sampler).
///
/// What is outside 0 to 1 along each axis, and how a texel is taken where the picture is bigger on the screen than it is
/// and where it is smaller. A filter the file does not say is the viewer's to choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GltfSampler {
    /// Along `u` (glTF's `wrapS`).
    pub wrap_s: Wrap,
    /// Along `v` (glTF's `wrapT`).
    pub wrap_t: Wrap,
    /// Where the picture is magnified (glTF's `magFilter`).
    pub mag_filter: Option<Filter>,
    /// Where it is minified (glTF's `minFilter`).
    pub min_filter: Option<MinFilter>,
}

/// What a picture is outside 0 to 1 along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    /// It starts again (glTF's `REPEAT`, its default).
    #[default]
    Repeat,
    /// It starts again mirrored, every other time (`MIRRORED_REPEAT`).
    MirroredRepeat,
    /// The texel at the edge (`CLAMP_TO_EDGE`).
    ClampToEdge,
}

/// How a texel is taken between the texels of a picture, or between two of its mip levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    /// The nearest one (`NEAREST`): the texels show as squares.
    Nearest,
    /// A blend of the nearest ones (`LINEAR`).
    Linear,
}

/// glTF's `minFilter`: how a texel is taken within a level, and from which mip levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinFilter {
    /// Within a level.
    pub filter: Filter,
    /// Between levels: from the nearest (`Nearest`) or blended between two (`Linear`); `None` reads the first level
    /// alone (glTF's `NEAREST` and `LINEAR`, which have no mip levels).
    pub mip: Option<Filter>,
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
    /// Linear, multiplied with the picture; each channel 0 to 1, as glTF's core holds it.
    pub emissive_factor: [f32; 3],
    /// How many times brighter than `emissive_factor` says the surface glows (`KHR_materials_emissive_strength`: 1
    /// where the file does not say, and not held to 1): what it gives off is the picture times the factor times this.
    pub emissive_strength: f32,
    /// The surface is its base colour whatever lights it (`KHR_materials_unlit`): what a renderer draws of it is the
    /// base colour (its factor, its picture and its alpha) and nothing else of the material.
    pub unlit: bool,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// How much of the light that the surface does not reflect goes through it rather than into it, 0 to 1
    /// (`KHR_materials_transmission`: 0 where the file does not say), multiplied with the **red** channel of its picture.
    pub transmission_factor: f32,
    pub transmission: Option<TextureRef>,
    /// The index of refraction (`KHR_materials_ior`): 1.5 where the file does not say.
    pub ior: f32,
    /// How thick the volume under the surface is, in the units of the mesh (`KHR_materials_volume`): 0 for a surface as
    /// thin as a sheet, which bends nothing (and where the file does not say); multiplied with the **green** channel of
    /// its picture.
    pub thickness_factor: f32,
    pub thickness: Option<TextureRef>,
    /// How far light goes through the volume before only `attenuation_color` of it is left (infinite where the file
    /// does not say: none is absorbed), and that colour, linear (white where the file does not say).
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
    /// How strong the specular layer of what is not a metal is, 0 to 1 (`KHR_materials_specular`: 1 where the file does
    /// not say), multiplied with the **alpha** channel of its picture.
    pub specular_factor: f32,
    pub specular: Option<TextureRef>,
    /// The colour of the reflection of what is not a metal at a right angle, a factor of what its index makes it,
    /// linear (white where the file does not say), multiplied with the red, green and blue of its picture (in the sRGB
    /// curve).
    pub specular_color_factor: [f32; 3],
    pub specular_color: Option<TextureRef>,
    /// How strong the clear layer over the surface is, 0 to 1 (`KHR_materials_clearcoat`: 0 where the file does not say,
    /// no layer), multiplied with the **red** channel of its picture.
    pub clearcoat_factor: f32,
    pub clearcoat: Option<TextureRef>,
    /// How rough the clear layer is, 0 to 1 (0 where the file does not say), multiplied with the **green** channel of
    /// its picture.
    pub clearcoat_roughness_factor: f32,
    pub clearcoat_roughness: Option<TextureRef>,
    /// The normal map of the clear layer, and how much it bends the normal (1 where the file does not say). Without one
    /// the layer has the surface's own normal, not bent by the material's normal map (glTF).
    pub clearcoat_normal: Option<TextureRef>,
    pub clearcoat_normal_scale: f32,
    /// The colour of the sheen of cloth and velvet at grazing angles (`KHR_materials_sheen`), linear: black where the
    /// file does not say, no sheen. Multiplied with the red, green and blue of its picture (in the sRGB curve).
    pub sheen_color_factor: [f32; 3],
    pub sheen_color: Option<TextureRef>,
    /// How rough the sheen is, 0 to 1 (0 where the file does not say), multiplied with the **alpha** channel of its
    /// picture.
    pub sheen_roughness_factor: f32,
    pub sheen_roughness: Option<TextureRef>,
    /// How strongly the highlight is stretched along a direction of the surface, 0 to 1 (`KHR_materials_anisotropy`: 0
    /// where the file does not say, none), multiplied with the **blue** channel of its picture.
    pub anisotropy_strength: f32,
    /// How far that direction is turned from the tangent, counter-clockwise, in radians (0 where the file does not say).
    pub anisotropy_rotation: f32,
    /// The picture of the direction, in the frame of the tangent (red and green, -1 to 1 stored as 0 to 1; along the
    /// tangent without one) and of the strength (blue), linear.
    pub anisotropy: Option<TextureRef>,
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
        let bytes =
            std::fs::read(path).map_err(|err| Error::Gltf(format!("{}: {err}", path.display())))?;
        let gltf = open(&bytes).map_err(|err| Error::Gltf(format!("{}: {err}", path.display())))?;
        let folder = path.parent().unwrap_or_else(|| Path::new("."));
        let (buffers, mut files) = load_buffers(&gltf, folder)?;
        files.insert(0, path.to_path_buf());
        let mut scene = build(&gltf.document, &buffers, folder)?;
        scene.files = files;
        Ok(scene)
    }

    /// Reads a `.glb` (or a `.gltf` without files of its own) from memory. Files it refers to are
    /// looked for in `folder`.
    ///
    /// # Errors
    /// As [`read_from_file`](Self::read_from_file).
    pub fn read_from_slice(bytes: &[u8], folder: &Path) -> Result<Self, Error> {
        let gltf = open(bytes).map_err(Error::Gltf)?;
        let (buffers, files) = load_buffers(&gltf, folder)?;
        let mut scene = build(&gltf.document, &buffers, folder)?;
        scene.files = files;
        Ok(scene)
    }
}

/// The extensions this reads itself from a material's JSON (the `gltf` crate does not know them): a file that requires
/// one of them is read, where the crate would refuse it.
const OWN_EXTENSIONS: [&str; 3] = [
    "KHR_materials_anisotropy",
    "KHR_materials_clearcoat",
    "KHR_materials_sheen",
];

/// A `.gltf` or `.glb` from memory, validated as the `gltf` crate validates it, but for an extension it does not know
/// that is required and that this reads itself ([`OWN_EXTENSIONS`]).
fn open(bytes: &[u8]) -> Result<gltf::Gltf, String> {
    use gltf::json::validation::{Error as Invalid, Validate as _};
    let gltf = gltf::Gltf::from_slice_without_validation(bytes).map_err(|err| err.to_string())?;
    let root = gltf.document.as_json();
    let mut errors = Vec::new();
    root.validate(root, gltf::json::Path::new, &mut |path, error| {
        let path = path();
        let own = matches!(error, Invalid::Unsupported)
            && path.as_str().starts_with("extensionsRequired")
            && OWN_EXTENSIONS
                .iter()
                .any(|extension| path.as_str().ends_with(&format!("= \"{extension}\"")));
        if !own {
            errors.push(format!("{path}: {error}"));
        }
    });
    if errors.is_empty() {
        Ok(gltf)
    } else {
        Err(format!("invalid glTF: {}", errors.join("; ")))
    }
}

/// The buffers of a file, and the files they were in.
fn load_buffers(gltf: &gltf::Gltf, folder: &Path) -> Result<(Vec<Vec<u8>>, Vec<PathBuf>), Error> {
    let mut buffers = Vec::new();
    let mut files = Vec::new();
    for buffer in gltf.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf.blob.clone().ok_or_else(|| {
                Error::Gltf("a buffer is in the .glb file, which has none".into())
            })?,
            gltf::buffer::Source::Uri(uri) => {
                let (data, file) = read_uri(uri, folder)?;
                files.extend(file);
                data
            }
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
    Ok((buffers, files))
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
        scene.materials.push(read_material(&material, document)?);
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

/// The picture of `texture`, read with the set `tex_coord` and its sampler.
fn texture_ref(texture: &gltf::Texture<'_>, tex_coord: u32) -> TextureRef {
    TextureRef {
        image: texture.source().index(),
        tex_coord,
        sampler: sampler_of(&texture.sampler()),
    }
}

/// What glTF's `sampler` says.
fn sampler_of(sampler: &gltf::texture::Sampler<'_>) -> GltfSampler {
    use gltf::texture::{MagFilter, MinFilter as Min, WrappingMode};
    let wrap = |mode| match mode {
        WrappingMode::Repeat => Wrap::Repeat,
        WrappingMode::MirroredRepeat => Wrap::MirroredRepeat,
        WrappingMode::ClampToEdge => Wrap::ClampToEdge,
    };
    let min = |filter, mip| MinFilter { filter, mip };
    GltfSampler {
        wrap_s: wrap(sampler.wrap_s()),
        wrap_t: wrap(sampler.wrap_t()),
        mag_filter: sampler.mag_filter().map(|filter| match filter {
            MagFilter::Nearest => Filter::Nearest,
            MagFilter::Linear => Filter::Linear,
        }),
        min_filter: sampler.min_filter().map(|filter| match filter {
            Min::Nearest => min(Filter::Nearest, None),
            Min::Linear => min(Filter::Linear, None),
            Min::NearestMipmapNearest => min(Filter::Nearest, Some(Filter::Nearest)),
            Min::LinearMipmapNearest => min(Filter::Linear, Some(Filter::Nearest)),
            Min::NearestMipmapLinear => min(Filter::Nearest, Some(Filter::Linear)),
            Min::LinearMipmapLinear => min(Filter::Linear, Some(Filter::Linear)),
        }),
    }
}

/// An extension of a material that the `gltf` crate does not know ([`OWN_EXTENSIONS`]), as its JSON: what it says is
/// read here, as the crate reads the ones it knows (a number that is not one, or a texture the file does not have, is an
/// error; what it does not say is glTF's default).
struct OwnExtension<'a> {
    name: &'static str,
    value: Option<&'a gltf::json::Value>,
    material: &'a gltf::Material<'a>,
    document: &'a gltf::Document,
}

impl<'a> OwnExtension<'a> {
    fn of(
        material: &'a gltf::Material<'a>,
        document: &'a gltf::Document,
        name: &'static str,
    ) -> Self {
        Self {
            name,
            value: material.extension_value(name),
            material,
            document,
        }
    }

    fn error(&self, what: &str) -> Error {
        let material = match (self.material.name(), self.material.index()) {
            (Some(name), _) => format!("the material {name}"),
            (None, Some(index)) => format!("material {index}"),
            (None, None) => "the default material".to_owned(),
        };
        Error::Gltf(format!("{material}: {}'s {what}", self.name))
    }

    /// The number of `key` in `object` (the extension, or a texture of it), or `default` where it does not say.
    fn number_in(
        &self,
        object: Option<&gltf::json::Value>,
        key: &str,
        default: f32,
    ) -> Result<f32, Error> {
        object
            .and_then(|object| object.get(key))
            .map_or(Ok(default), |value| {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "glTF's numbers are single floats"
                )]
                let number = value.as_f64().map(|number| number as f32);
                number.ok_or_else(|| self.error(&format!("{key} is not a number")))
            })
    }

    /// The number `key` of the extension, or `default`.
    fn number(&self, key: &str, default: f32) -> Result<f32, Error> {
        self.number_in(self.value, key, default)
    }

    /// The three numbers `key` of the extension (a colour), or `default`.
    fn numbers3(&self, key: &str, default: [f32; 3]) -> Result<[f32; 3], Error> {
        let Some(value) = self.value.and_then(|value| value.get(key)) else {
            return Ok(default);
        };
        let numbers: Option<Vec<f32>> = value.as_array().and_then(|items| {
            items
                .iter()
                .map(|item| {
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "glTF's numbers are single floats"
                    )]
                    let number = item.as_f64().map(|number| number as f32);
                    number
                })
                .collect()
        });
        numbers
            .and_then(|numbers| <[f32; 3]>::try_from(numbers).ok())
            .ok_or_else(|| self.error(&format!("{key} is not three numbers")))
    }

    /// The texture `key` of the extension (glTF's `textureInfo`: the texture, and the set of texture coordinates), as
    /// its picture, and its JSON (a normal map's has its scale).
    fn texture(&self, key: &str) -> Result<Option<(TextureRef, &'a gltf::json::Value)>, Error> {
        let Some(info) = self.value.and_then(|value| value.get(key)) else {
            return Ok(None);
        };
        let index = info
            .get("index")
            .and_then(gltf::json::Value::as_u64)
            .ok_or_else(|| self.error(&format!("{key} has no index of a texture")))?;
        let texture = usize::try_from(index)
            .ok()
            .and_then(|index| self.document.textures().nth(index))
            .ok_or_else(|| {
                self.error(&format!(
                    "{key} names texture {index}, which the file does not have"
                ))
            })?;
        let tex_coord = match info.get("texCoord") {
            None => 0,
            Some(set) => set
                .as_u64()
                .and_then(|set| u32::try_from(set).ok())
                .ok_or_else(|| self.error(&format!("{key}'s texCoord is not a set")))?,
        };
        Ok(Some((texture_ref(&texture, tex_coord), info)))
    }
}

/// What `KHR_materials_clearcoat` says of `material`: the strength of the layer and its picture, its roughness and its
/// picture, and its normal map and that map's scale.
struct Clearcoat {
    factor: f32,
    texture: Option<TextureRef>,
    roughness_factor: f32,
    roughness: Option<TextureRef>,
    normal: Option<TextureRef>,
    normal_scale: f32,
}

fn read_clearcoat(
    material: &gltf::Material,
    document: &gltf::Document,
) -> Result<Clearcoat, Error> {
    let extension = OwnExtension::of(material, document, "KHR_materials_clearcoat");
    let normal = extension.texture("clearcoatNormalTexture")?;
    Ok(Clearcoat {
        factor: extension.number("clearcoatFactor", 0.0)?,
        texture: extension.texture("clearcoatTexture")?.map(|(map, _)| map),
        roughness_factor: extension.number("clearcoatRoughnessFactor", 0.0)?,
        roughness: extension
            .texture("clearcoatRoughnessTexture")?
            .map(|(map, _)| map),
        normal_scale: extension.number_in(normal.map(|(_, info)| info), "scale", 1.0)?,
        normal: normal.map(|(map, _)| map),
    })
}

/// What `KHR_materials_sheen` says of `material`: the sheen's colour and its picture, its roughness and its picture.
struct Sheen {
    color_factor: [f32; 3],
    color: Option<TextureRef>,
    roughness_factor: f32,
    roughness: Option<TextureRef>,
}

fn read_sheen(material: &gltf::Material, document: &gltf::Document) -> Result<Sheen, Error> {
    let extension = OwnExtension::of(material, document, "KHR_materials_sheen");
    Ok(Sheen {
        color_factor: extension.numbers3("sheenColorFactor", [0.0; 3])?,
        color: extension.texture("sheenColorTexture")?.map(|(map, _)| map),
        roughness_factor: extension.number("sheenRoughnessFactor", 0.0)?,
        roughness: extension
            .texture("sheenRoughnessTexture")?
            .map(|(map, _)| map),
    })
}

/// What `KHR_materials_anisotropy` says of `material`: the strength, the turn and the picture.
struct Anisotropy {
    strength: f32,
    rotation: f32,
    texture: Option<TextureRef>,
}

fn read_anisotropy(
    material: &gltf::Material,
    document: &gltf::Document,
) -> Result<Anisotropy, Error> {
    let extension = OwnExtension::of(material, document, "KHR_materials_anisotropy");
    Ok(Anisotropy {
        strength: extension.number("anisotropyStrength", 0.0)?,
        rotation: extension.number("anisotropyRotation", 0.0)?,
        texture: extension.texture("anisotropyTexture")?.map(|(map, _)| map),
    })
}

fn read_material(
    material: &gltf::Material,
    document: &gltf::Document,
) -> Result<GltfMaterial, Error> {
    let clearcoat = read_clearcoat(material, document)?;
    let sheen = read_sheen(material, document)?;
    let anisotropy = read_anisotropy(material, document)?;
    let pbr = material.pbr_metallic_roughness();
    let normal = material.normal_texture();
    let occlusion = material.occlusion_texture();
    let transmission = material.transmission();
    let volume = material.volume();
    let specular = material.specular();
    Ok(GltfMaterial {
        name: material.name().map(str::to_owned),
        base_color_factor: pbr.base_color_factor(),
        base_color: pbr
            .base_color_texture()
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        metallic_roughness: pbr
            .metallic_roughness_texture()
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        normal_scale: normal
            .as_ref()
            .map_or(1.0, gltf::material::NormalTexture::scale),
        normal: normal.map(|info| texture_ref(&info.texture(), info.tex_coord())),
        occlusion_strength: occlusion
            .as_ref()
            .map_or(1.0, gltf::material::OcclusionTexture::strength),
        occlusion: occlusion.map(|info| texture_ref(&info.texture(), info.tex_coord())),
        emissive: material
            .emissive_texture()
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        emissive_factor: material.emissive_factor(),
        emissive_strength: material.emissive_strength().unwrap_or(1.0),
        unlit: material.unlit(),
        alpha_mode: match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        double_sided: material.double_sided(),
        transmission_factor: transmission
            .as_ref()
            .map_or(0.0, gltf::material::Transmission::transmission_factor),
        transmission: transmission
            .and_then(|transmission| transmission.transmission_texture())
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        ior: material.ior().unwrap_or(1.5),
        thickness_factor: volume
            .as_ref()
            .map_or(0.0, gltf::material::Volume::thickness_factor),
        thickness: volume
            .as_ref()
            .and_then(gltf::material::Volume::thickness_texture)
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        attenuation_distance: volume
            .as_ref()
            .map_or(f32::INFINITY, gltf::material::Volume::attenuation_distance),
        attenuation_color: volume
            .as_ref()
            .map_or([1.0; 3], gltf::material::Volume::attenuation_color),
        specular_factor: specular
            .as_ref()
            .map_or(1.0, gltf::material::Specular::specular_factor),
        specular: specular
            .as_ref()
            .and_then(gltf::material::Specular::specular_texture)
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        specular_color_factor: specular
            .as_ref()
            .map_or([1.0; 3], gltf::material::Specular::specular_color_factor),
        specular_color: specular
            .as_ref()
            .and_then(gltf::material::Specular::specular_color_texture)
            .map(|info| texture_ref(&info.texture(), info.tex_coord())),
        clearcoat_factor: clearcoat.factor,
        clearcoat: clearcoat.texture,
        clearcoat_roughness_factor: clearcoat.roughness_factor,
        clearcoat_roughness: clearcoat.roughness,
        clearcoat_normal: clearcoat.normal,
        clearcoat_normal_scale: clearcoat.normal_scale,
        sheen_color_factor: sheen.color_factor,
        sheen_color: sheen.color,
        sheen_roughness_factor: sheen.roughness_factor,
        sheen_roughness: sheen.roughness,
        anisotropy_strength: anisotropy.strength,
        anisotropy_rotation: anisotropy.rotation,
        anisotropy: anisotropy.texture,
    })
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
    // glTF numbers the sets from 0 with no gaps: the first one the file does not have is the end.
    let mut extra_uvs: Vec<Vec<[f32; 2]>> = Vec::new();
    while let Some(set) = u32::try_from(extra_uvs.len() + 1)
        .ok()
        .and_then(|set| reader.read_tex_coords(set))
    {
        extra_uvs.push(set.into_f32().collect());
    }
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(Iterator::collect);
    let colors: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect());

    let count = positions.len();
    for (name, len) in [
        ("normals", normals.as_ref().map(Vec::len)),
        ("texture coordinates", uvs.as_ref().map(Vec::len)),
        ("tangents", tangents.as_ref().map(Vec::len)),
        ("colours", colors.as_ref().map(Vec::len)),
    ]
    .into_iter()
    .chain(
        extra_uvs
            .iter()
            .map(|set| ("texture coordinates of a later set", Some(set.len()))),
    ) {
        if len.is_some_and(|len| len != count) {
            return Err(Error::Gltf(format!(
                "a primitive with {count} positions and {} {name}",
                len.unwrap_or(0)
            )));
        }
    }

    let (joints, weights) = read_influences(&reader, count)?;
    let attributes = read_attributes(primitive, buffers, count)?;

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
        extra_uvs,
        tangents,
        colors,
        attributes,
        indices,
        material: primitive.material().index(),
        joints,
        weights,
    }))
}

/// The attributes of the application of a primitive of `count` vertices (names that start with an
/// underscore), in the order of their names. An attribute of matrices is left out.
fn read_attributes(
    primitive: &gltf::Primitive,
    buffers: &[Vec<u8>],
    count: usize,
) -> Result<Vec<GltfAttribute>, Error> {
    use gltf::accessor::{DataType, Dimensions};
    use gltf::mesh::Semantic;

    let get = |buffer: gltf::Buffer| buffers.get(buffer.index()).map(Vec::as_slice);
    let mut attributes = Vec::new();
    for (semantic, accessor) in primitive.attributes() {
        let Semantic::Extras(name) = semantic else {
            continue;
        };
        let components = match accessor.dimensions() {
            Dimensions::Scalar => 1,
            Dimensions::Vec2 => 2,
            Dimensions::Vec3 => 3,
            Dimensions::Vec4 => 4,
            Dimensions::Mat2 | Dimensions::Mat3 | Dimensions::Mat4 => continue,
        };
        let values = match accessor.data_type() {
            DataType::I8 => read_numbers::<i8, _>(accessor, get, components),
            DataType::U8 => read_numbers::<u8, _>(accessor, get, components),
            DataType::I16 => read_numbers::<i16, _>(accessor, get, components),
            DataType::U16 => read_numbers::<u16, _>(accessor, get, components),
            DataType::U32 => read_numbers::<u32, _>(accessor, get, components),
            DataType::F32 => read_numbers::<f32, _>(accessor, get, components),
        }
        .ok_or_else(|| Error::Gltf(format!("the attribute _{name} can not be read")))?;
        if values.len() != count {
            return Err(Error::Gltf(format!(
                "a primitive with {count} positions and {} values of _{name}",
                values.len()
            )));
        }
        attributes.push(GltfAttribute {
            name: format!("_{name}"),
            components,
            values,
        });
    }
    Ok(attributes)
}

/// The elements of `accessor`, each `components` numbers of type `N` (1 to 4), as four floats (0 past
/// `components`), normalised as the accessor says. `None` if its data is not in the buffers.
fn read_numbers<'a, 's, N, F>(
    accessor: gltf::Accessor<'a>,
    get: F,
    components: u8,
) -> Option<Vec<[f32; 4]>>
where
    N: Number,
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    use gltf::accessor::Iter;

    let normalized = accessor.normalized();
    let float = |number: N| number.to_f32(normalized);
    Some(match components {
        1 => Iter::<N>::new(accessor, get)?
            .map(|x| [float(x), 0.0, 0.0, 0.0])
            .collect(),
        2 => Iter::<[N; 2]>::new(accessor, get)?
            .map(|[x, y]| [float(x), float(y), 0.0, 0.0])
            .collect(),
        3 => Iter::<[N; 3]>::new(accessor, get)?
            .map(|[x, y, z]| [float(x), float(y), float(z), 0.0])
            .collect(),
        _ => Iter::<[N; 4]>::new(accessor, get)?
            .map(|numbers| numbers.map(float))
            .collect(),
    })
}

/// A type of the numbers of an accessor, read as a float: a normalised integer as glTF has it
/// (unsigned from 0 to 1, signed from -1 to 1, the smallest signed one -1 as well), any other as its
/// value.
trait Number: gltf::accessor::Item + Copy {
    fn to_f32(self, normalized: bool) -> f32;
}

impl Number for f32 {
    fn to_f32(self, _normalized: bool) -> f32 {
        self
    }
}

impl Number for u8 {
    fn to_f32(self, normalized: bool) -> f32 {
        if normalized {
            f32::from(self) / f32::from(Self::MAX)
        } else {
            f32::from(self)
        }
    }
}

impl Number for u16 {
    fn to_f32(self, normalized: bool) -> f32 {
        if normalized {
            f32::from(self) / f32::from(Self::MAX)
        } else {
            f32::from(self)
        }
    }
}

impl Number for i8 {
    fn to_f32(self, normalized: bool) -> f32 {
        if normalized {
            (f32::from(self) / f32::from(Self::MAX)).max(-1.0)
        } else {
            f32::from(self)
        }
    }
}

impl Number for i16 {
    fn to_f32(self, normalized: bool) -> f32 {
        if normalized {
            (f32::from(self) / f32::from(Self::MAX)).max(-1.0)
        } else {
            f32::from(self)
        }
    }
}

impl Number for u32 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "a value past 2^24 is rounded; glTF allows no u32 in an attribute of the application, and no normalised one anywhere"
    )]
    fn to_f32(self, _normalized: bool) -> f32 {
        self as f32
    }
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

    /// Two materials: a glass with what all three of glTF's extensions of glass say, each with a picture; and a plain
    /// one, which has what they say where a file does not.
    const GLASS: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior", "KHR_materials_volume"],
  "images": [{"uri": "transmission.png"}, {"uri": "thickness.png"}],
  "textures": [{"source": 0}, {"source": 1}],
  "materials": [
    {"name": "glass", "extensions": {
      "KHR_materials_transmission": {"transmissionFactor": 0.9, "transmissionTexture": {"index": 0}},
      "KHR_materials_ior": {"ior": 1.33},
      "KHR_materials_volume": {"thicknessFactor": 0.25, "thicknessTexture": {"index": 1, "texCoord": 1},
                               "attenuationDistance": 2.0, "attenuationColor": [0.5, 0.8, 0.6]}}},
    {"name": "plain"}
  ]
}"#;

    #[test]
    fn a_glass_has_its_transmission_its_index_and_its_volume_and_a_plain_material_glass_s_defaults()
    {
        // The pictures are where the file says (they are not decoded: a byte each is enough).
        let folder = std::env::temp_dir().join(format!("polypath_glass_{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        for name in ["transmission.png", "thickness.png"] {
            std::fs::write(folder.join(name), [0_u8]).unwrap();
        }
        let scene = GltfScene::read_from_slice(GLASS.as_bytes(), &folder).unwrap();
        let _ = std::fs::remove_dir_all(&folder);

        let glass = &scene.materials[0];
        assert!((glass.transmission_factor - 0.9).abs() < 1e-6);
        assert_eq!(glass.transmission.map(|map| map.image), Some(0));
        assert!((glass.ior - 1.33).abs() < 1e-6);
        assert!((glass.thickness_factor - 0.25).abs() < 1e-6);
        assert_eq!(
            glass.thickness.map(|map| (map.image, map.tex_coord)),
            Some((1, 1))
        );
        assert!((glass.attenuation_distance - 2.0).abs() < 1e-6);
        assert!(close(glass.attenuation_color, [0.5, 0.8, 0.6]));

        let plain = &scene.materials[1];
        assert_eq!(plain.transmission_factor, 0.0);
        assert_eq!(plain.transmission, None);
        assert_eq!(plain.ior, 1.5);
        assert_eq!(plain.thickness_factor, 0.0);
        assert_eq!(plain.thickness, None);
        assert_eq!(plain.attenuation_distance, f32::INFINITY);
        assert_eq!(plain.attenuation_color, [1.0; 3]);
    }

    /// A material with what `KHR_materials_specular` says, each with a picture, and a plain one.
    const SPECULAR: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_specular"],
  "images": [{"uri": "specular.png"}, {"uri": "specular_color.png"}],
  "textures": [{"source": 0}, {"source": 1}],
  "materials": [
    {"name": "coated", "extensions": {"KHR_materials_specular": {
      "specularFactor": 0.25, "specularTexture": {"index": 0},
      "specularColorFactor": [1.0, 0.5, 0.25], "specularColorTexture": {"index": 1, "texCoord": 1}}}},
    {"name": "plain"}
  ]
}"#;

    #[test]
    fn a_specular_material_has_its_strength_and_its_colour_and_a_plain_one_the_defaults() {
        let folder = std::env::temp_dir().join(format!("polypath_specular_{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        for name in ["specular.png", "specular_color.png"] {
            std::fs::write(folder.join(name), [0_u8]).unwrap();
        }
        let scene = GltfScene::read_from_slice(SPECULAR.as_bytes(), &folder).unwrap();
        let _ = std::fs::remove_dir_all(&folder);

        let coated = &scene.materials[0];
        assert!((coated.specular_factor - 0.25).abs() < 1e-6);
        assert_eq!(coated.specular.map(|map| map.image), Some(0));
        assert!(close(coated.specular_color_factor, [1.0, 0.5, 0.25]));
        assert_eq!(
            coated.specular_color.map(|map| (map.image, map.tex_coord)),
            Some((1, 1))
        );

        let plain = &scene.materials[1];
        assert_eq!(plain.specular_factor, 1.0);
        assert_eq!(plain.specular, None);
        assert_eq!(plain.specular_color_factor, [1.0; 3]);
        assert_eq!(plain.specular_color, None);
    }

    /// A material with what `KHR_materials_clearcoat` says, each with a picture (the textures name the pictures the other
    /// way round, so that a texture's picture is not its own number), and a plain one.
    const CLEARCOAT: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_clearcoat"],
  "images": [{"uri": "coat.png"}, {"uri": "coat_roughness.png"}, {"uri": "coat_normal.png"}],
  "textures": [{"source": 2}, {"source": 1}, {"source": 0}],
  "materials": [
    {"name": "varnished", "extensions": {"KHR_materials_clearcoat": {
      "clearcoatFactor": 0.75, "clearcoatTexture": {"index": 2},
      "clearcoatRoughnessFactor": 0.25, "clearcoatRoughnessTexture": {"index": 1, "texCoord": 1},
      "clearcoatNormalTexture": {"index": 0, "scale": 0.5}}}},
    {"name": "plain"}
  ]
}"#;

    /// Four samplers (each wrap, every filter, none said) and a texture without one, read by the core's base colour and
    /// by an extension the crate does not know (the clear coat).
    const SAMPLERS: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_clearcoat"],
  "images": [{"uri": "data:image/png;base64,AA=="}],
  "samplers": [
    {"wrapS": 33071, "wrapT": 33648, "magFilter": 9728, "minFilter": 9986},
    {"magFilter": 9729, "minFilter": 9729},
    {"wrapT": 33071, "minFilter": 9984},
    {}
  ],
  "textures": [{"source": 0, "sampler": 0}, {"source": 0, "sampler": 1}, {"source": 0, "sampler": 2},
    {"source": 0, "sampler": 3}, {"source": 0}],
  "materials": [
    {"name": "clamped", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
      "extensions": {"KHR_materials_clearcoat": {"clearcoatTexture": {"index": 2}}}},
    {"name": "blended", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}},
    {"name": "unsaid", "pbrMetallicRoughness": {"baseColorTexture": {"index": 3}}},
    {"name": "none", "pbrMetallicRoughness": {"baseColorTexture": {"index": 4}}}
  ]
}"#;

    #[test]
    fn a_texture_is_read_with_its_sampler_by_the_core_and_by_an_extension() {
        let scene = GltfScene::read_from_slice(SAMPLERS.as_bytes(), Path::new(".")).unwrap();
        let sampler = |material: usize| scene.materials[material].base_color.unwrap().sampler;

        assert_eq!(
            sampler(0),
            GltfSampler {
                wrap_s: Wrap::ClampToEdge,
                wrap_t: Wrap::MirroredRepeat,
                mag_filter: Some(Filter::Nearest),
                min_filter: Some(MinFilter {
                    filter: Filter::Nearest,
                    mip: Some(Filter::Linear),
                }),
            }
        );
        // Minified without mip levels.
        assert_eq!(
            sampler(1).min_filter,
            Some(MinFilter {
                filter: Filter::Linear,
                mip: None,
            })
        );
        assert_eq!(sampler(1).mag_filter, Some(Filter::Linear));
        // A sampler that says nothing, and a texture without one: repeat, and the filters the viewer's.
        assert_eq!(sampler(2), GltfSampler::default());
        assert_eq!(sampler(3), GltfSampler::default());
        assert_eq!(sampler(3).wrap_s, Wrap::Repeat);
        // The clear coat's texture, which the crate does not read, has its sampler too.
        assert_eq!(
            scene.materials[0].clearcoat.unwrap().sampler,
            GltfSampler {
                wrap_s: Wrap::Repeat,
                wrap_t: Wrap::ClampToEdge,
                mag_filter: None,
                min_filter: Some(MinFilter {
                    filter: Filter::Nearest,
                    mip: Some(Filter::Nearest),
                }),
            }
        );
    }

    #[test]
    fn a_clearcoat_material_has_its_layer_and_a_plain_one_none() {
        let folder =
            std::env::temp_dir().join(format!("polypath_clearcoat_{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        for name in ["coat.png", "coat_roughness.png", "coat_normal.png"] {
            std::fs::write(folder.join(name), [0_u8]).unwrap();
        }
        let scene = GltfScene::read_from_slice(CLEARCOAT.as_bytes(), &folder).unwrap();
        let _ = std::fs::remove_dir_all(&folder);

        let varnished = &scene.materials[0];
        assert!((varnished.clearcoat_factor - 0.75).abs() < 1e-6);
        assert_eq!(varnished.clearcoat.map(|map| map.image), Some(0));
        assert!((varnished.clearcoat_roughness_factor - 0.25).abs() < 1e-6);
        assert_eq!(
            varnished
                .clearcoat_roughness
                .map(|map| (map.image, map.tex_coord)),
            Some((1, 1))
        );
        assert_eq!(varnished.clearcoat_normal.map(|map| map.image), Some(2));
        assert!((varnished.clearcoat_normal_scale - 0.5).abs() < 1e-6);

        let plain = &scene.materials[1];
        assert_eq!(plain.clearcoat_factor, 0.0);
        assert_eq!(plain.clearcoat, None);
        assert_eq!(plain.clearcoat_roughness_factor, 0.0);
        assert_eq!(plain.clearcoat_roughness, None);
        assert_eq!(plain.clearcoat_normal, None);
        assert_eq!(plain.clearcoat_normal_scale, 1.0);
    }

    #[test]
    fn a_clearcoat_that_names_a_texture_the_file_does_not_have_or_a_number_that_is_not_one_is_an_error()
     {
        let with = |clearcoat: &str| {
            format!(
                r#"{{"asset": {{"version": "2.0"}}, "extensionsUsed": ["KHR_materials_clearcoat"],
  "images": [{{"uri": "data:image/png;base64,AA=="}}], "textures": [{{"source": 0}}],
  "materials": [{{"name": "varnished", "extensions": {{"KHR_materials_clearcoat": {clearcoat}}}}}]}}"#
            )
        };
        for (clearcoat, says) in [
            (r#"{"clearcoatTexture": {"index": 3}}"#, "names texture 3"),
            (
                r#"{"clearcoatFactor": "much"}"#,
                "clearcoatFactor is not a number",
            ),
            (
                r#"{"clearcoatNormalTexture": {"index": 0, "scale": [1]}}"#,
                "scale is not a number",
            ),
            (r#"{"clearcoatRoughnessTexture": {}}"#, "has no index"),
        ] {
            let read = GltfScene::read_from_slice(with(clearcoat).as_bytes(), Path::new("."));
            assert!(
                matches!(&read, Err(Error::Gltf(message)) if message.contains("varnished") && message.contains(says)),
                "{clearcoat}: {read:?}"
            );
        }
    }

    /// A material with what `KHR_materials_sheen` says, each with a picture, and a plain one.
    const SHEEN: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_sheen"],
  "images": [{"uri": "data:image/png;base64,AA=="}, {"uri": "data:image/png;base64,AA=="}],
  "textures": [{"source": 1}, {"source": 0}],
  "materials": [
    {"name": "velvet", "extensions": {"KHR_materials_sheen": {
      "sheenColorFactor": [1.0, 0.5, 0.25], "sheenColorTexture": {"index": 0},
      "sheenRoughnessFactor": 0.75, "sheenRoughnessTexture": {"index": 1, "texCoord": 1}}}},
    {"name": "plain"}
  ]
}"#;

    #[test]
    fn a_sheen_material_has_its_colour_and_roughness_and_a_plain_one_none() {
        let scene = GltfScene::read_from_slice(SHEEN.as_bytes(), Path::new(".")).unwrap();
        let velvet = &scene.materials[0];
        assert!(close(velvet.sheen_color_factor, [1.0, 0.5, 0.25]));
        assert_eq!(velvet.sheen_color.map(|map| map.image), Some(1));
        assert!((velvet.sheen_roughness_factor - 0.75).abs() < 1e-6);
        assert_eq!(
            velvet.sheen_roughness.map(|map| (map.image, map.tex_coord)),
            Some((0, 1))
        );
        let plain = &scene.materials[1];
        assert_eq!(plain.sheen_color_factor, [0.0; 3]);
        assert_eq!(plain.sheen_color, None);
        assert_eq!(plain.sheen_roughness_factor, 0.0);
        assert_eq!(plain.sheen_roughness, None);
        // A colour that is not three numbers is an error.
        let broken = SHEEN.replace("[1.0, 0.5, 0.25]", "[1.0, 0.5]");
        let read = GltfScene::read_from_slice(broken.as_bytes(), Path::new("."));
        assert!(
            matches!(&read, Err(Error::Gltf(message)) if message.contains("sheenColorFactor is not three numbers")),
            "{read:?}"
        );
    }

    #[test]
    fn an_anisotropic_material_has_its_strength_turn_and_picture_and_a_plain_one_none() {
        let file = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_anisotropy"],
  "images": [{"uri": "data:image/png;base64,AA=="}, {"uri": "data:image/png;base64,AA=="}],
  "textures": [{"source": 1}, {"source": 0}],
  "materials": [
    {"name": "brushed", "extensions": {"KHR_materials_anisotropy": {
      "anisotropyStrength": 0.6, "anisotropyRotation": 1.5, "anisotropyTexture": {"index": 0}}}},
    {"name": "plain"}
  ]
}"#;
        let scene = GltfScene::read_from_slice(file.as_bytes(), Path::new(".")).unwrap();
        let brushed = &scene.materials[0];
        assert!((brushed.anisotropy_strength - 0.6).abs() < 1e-6);
        assert!((brushed.anisotropy_rotation - 1.5).abs() < 1e-6);
        assert_eq!(brushed.anisotropy.map(|map| map.image), Some(1));
        let plain = &scene.materials[1];
        assert_eq!(plain.anisotropy_strength, 0.0);
        assert_eq!(plain.anisotropy_rotation, 0.0);
        assert_eq!(plain.anisotropy, None);
    }

    /// A material that glows four times as strong as its factor says, and a plain one.
    const EMISSIVE_STRENGTH: &str = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_emissive_strength"],
  "materials": [
    {"name": "lamp", "emissiveFactor": [1.0, 0.5, 0.25],
     "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}}},
    {"name": "plain", "emissiveFactor": [1.0, 0.5, 0.25]}
  ]
}"#;

    #[test]
    fn a_file_may_require_an_extension_this_reads_itself_and_not_one_it_does_not_know() {
        let requiring = |extension: &str| {
            format!(
                r#"{{"asset": {{"version": "2.0"}}, "extensionsUsed": ["{extension}"], "extensionsRequired": ["{extension}"]}}"#
            )
        };
        for extension in OWN_EXTENSIONS {
            let file = requiring(extension);
            assert!(
                GltfScene::read_from_slice(file.as_bytes(), Path::new(".")).is_ok(),
                "{extension}"
            );
        }
        let unknown = requiring("EXT_not_an_extension");
        let refused = GltfScene::read_from_slice(unknown.as_bytes(), Path::new("."));
        assert!(
            matches!(refused, Err(Error::Gltf(message)) if message.contains("EXT_not_an_extension"))
        );
        // What the crate refuses besides is still refused: an accessor of a buffer view that is not there.
        let broken = r#"{"asset": {"version": "2.0"}, "accessors": [{"bufferView": 3, "componentType": 5126, "count": 1, "type": "SCALAR"}]}"#;
        assert!(GltfScene::read_from_slice(broken.as_bytes(), Path::new(".")).is_err());
    }

    #[test]
    fn a_material_of_khr_materials_unlit_is_unlit_and_a_plain_one_is_not() {
        let file = r#"{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_materials_unlit"],
  "materials": [
    {"name": "flat", "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]},
     "extensions": {"KHR_materials_unlit": {}}},
    {"name": "plain"}
  ]
}"#;
        let scene = GltfScene::read_from_slice(file.as_bytes(), Path::new(".")).unwrap();
        assert!(scene.materials[0].unlit);
        assert!(!scene.materials[1].unlit);
    }

    #[test]
    fn a_material_has_its_emissive_strength_and_a_plain_one_a_strength_of_1() {
        let scene =
            GltfScene::read_from_slice(EMISSIVE_STRENGTH.as_bytes(), Path::new(".")).unwrap();
        let (lamp, plain) = (&scene.materials[0], &scene.materials[1]);
        // The factor is as the file has it, and the strength beside it.
        assert!(close(lamp.emissive_factor, [1.0, 0.5, 0.25]));
        assert!((lamp.emissive_strength - 4.0).abs() < 1e-6);
        assert!(close(plain.emissive_factor, [1.0, 0.5, 0.25]));
        assert_eq!(plain.emissive_strength, 1.0);
    }

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
            &[
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
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
                [
                    0.0,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2
                ]
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
        assert!(primitive.colors.is_none());
        assert_eq!(primitive.indices, [0, 1, 2]);
        assert_eq!(primitive.material, Some(0));

        let material = &scene.materials[0];
        assert_eq!(material.base_color_factor, [1.0, 0.0, 0.0, 1.0]);
        assert!((material.metallic_factor - 0.25).abs() < 1e-6);
        assert!((material.roughness_factor - 0.75).abs() < 1e-6);
        assert!(material.double_sided);
        assert_eq!(material.alpha_mode, AlphaMode::Opaque);
    }

    /// A triangle whose vertices have colours: four normalized 16-bit numbers each (what Blender writes),
    /// or three normalized bytes (an RGB colour, whose alpha is 1).
    fn coloured_triangle_file(rgba16: bool) -> String {
        let mut data: Vec<u8> = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        let colours_at = data.len();
        if rgba16 {
            for value in [
                0_u16, 65_535, 32_768, 65_535, 65_535, 0, 0, 13_107, 0, 0, 65_535, 52_428,
            ] {
                data.extend(value.to_le_bytes());
            }
        } else {
            // Each colour padded to four bytes, as glTF wants the elements of an accessor aligned.
            for colour in [[0_u8, 255, 51], [255, 0, 0], [0, 0, 255]] {
                data.extend(colour);
                data.push(0);
            }
        }
        let colours_length = data.len() - colours_at;
        let (component, kind, stride) = if rgba16 {
            (5123, "VEC4", "")
        } else {
            (5121, "VEC3", r#", "byteStride": 4"#)
        };
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64_encode(&data)
        );
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"name": "tri", "mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "COLOR_0": 1}}}}]}}],
  "buffers": [{{"byteLength": {length}, "uri": "{uri}"}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": {colours_at}, "byteLength": {colours_length}{stride}}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
    {{"bufferView": 1, "componentType": {component}, "normalized": true, "count": 3, "type": "{kind}"}}
  ]
}}"#,
            length = data.len()
        )
    }

    /// A triangle with three sets of texture coordinates: the first, a second of its own, and a third of bytes (normalised,
    /// as glTF allows a set to be). The second primitive has the first set alone.
    fn three_sets_file() -> String {
        let mut data: Vec<u8> = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        for value in [
            0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75,
        ] {
            data.extend(value.to_le_bytes());
        }
        // The third set as bytes, each pair padded to four.
        for pair in [[0_u8, 255], [255, 0], [51, 102]] {
            data.extend(pair);
            data.extend([0, 0]);
        }
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64_encode(&data)
        );
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"name": "tri", "mesh": 0}}],
  "meshes": [{{"primitives": [
    {{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2, "TEXCOORD_2": 3}}}},
    {{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1}}}}
  ]}}],
  "buffers": [{{"byteLength": {length}, "uri": "{uri}"}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 84, "byteLength": 12, "byteStride": 4}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}},
    {{"bufferView": 1, "byteOffset": 24, "componentType": 5126, "count": 3, "type": "VEC2"}},
    {{"bufferView": 2, "componentType": 5121, "normalized": true, "count": 3, "type": "VEC2"}}
  ]
}}"#,
            length = data.len()
        )
    }

    #[test]
    fn every_set_of_texture_coordinates_is_read_in_its_order() {
        let scene =
            GltfScene::read_from_slice(three_sets_file().as_bytes(), Path::new(".")).unwrap();
        let three = &scene.meshes[0].primitives[0];
        assert_eq!(
            three.uvs.as_deref(),
            Some(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]][..])
        );
        assert_eq!(three.extra_uvs.len(), 2);
        assert_eq!(three.extra_uvs[0], [[0.5, 0.5], [0.75, 0.5], [0.5, 0.75]]);
        let bytes = &three.extra_uvs[1];
        let wanted = [[0.0, 1.0], [1.0, 0.0], [0.2, 0.4]];
        for (got, wanted) in bytes.iter().zip(wanted) {
            assert!(
                (got[0] - wanted[0]).abs() < 1e-6 && (got[1] - wanted[1]).abs() < 1e-6,
                "{got:?}"
            );
        }
        // A primitive with the first set alone has none after it.
        assert_eq!(
            scene.meshes[0].primitives[1].extra_uvs,
            Vec::<Vec<[f32; 2]>>::new()
        );
    }

    #[test]
    fn the_colours_of_the_vertices_are_numbers_from_zero_to_one() {
        let scene =
            GltfScene::read_from_slice(coloured_triangle_file(true).as_bytes(), Path::new("."))
                .unwrap();
        let colours = scene.meshes[0].primitives[0].colors.as_ref().unwrap();
        let expected = [
            [0.0, 1.0, 0.5, 1.0],
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.8],
        ];
        for (colour, expected) in colours.iter().zip(expected) {
            for (value, wanted) in colour.iter().zip(expected) {
                assert!((value - wanted).abs() < 1e-4, "{colour:?} {expected:?}");
            }
        }

        // An RGB colour of bytes: alpha 1.
        let scene =
            GltfScene::read_from_slice(coloured_triangle_file(false).as_bytes(), Path::new("."))
                .unwrap();
        let colours = scene.meshes[0].primitives[0].colors.as_ref().unwrap();
        assert_eq!(colours.len(), 3);
        assert!((colours[0][1] - 1.0).abs() < 1e-6 && (colours[0][2] - 0.2).abs() < 1e-3);
        assert!(colours.iter().all(|colour| (colour[3] - 1.0).abs() < 1e-6));
        // The colours are not an attribute of the application.
        assert_eq!(
            scene.meshes[0].primitives[0].attributes,
            Vec::<GltfAttribute>::new()
        );
    }

    /// A triangle with attributes of the application: `_WIND` (four floats), `_TILT` (two normalised
    /// signed bytes, each pair padded to four), `_COUNT` (one u16, not normalised) and `_FRAME` (a 2 by 2
    /// matrix of floats, which is left out).
    fn attributes_file() -> String {
        let mut data: Vec<u8> = Vec::new();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        for value in [
            0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.25, 0.5, 0.75, 1.0,
        ] {
            data.extend(value.to_le_bytes());
        }
        for pair in [[127_i8, -127], [-128, 0], [64, 127]] {
            data.extend(pair.map(i8::to_le_bytes).concat());
            data.extend([0, 0]);
        }
        for count in [0_u16, 7, 65535] {
            data.extend(count.to_le_bytes());
        }
        data.extend([0, 0]);
        for value in 0..12_u8 {
            data.extend(f32::from(value).to_le_bytes());
        }
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64_encode(&data)
        );
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"name": "tri", "mesh": 0}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "_WIND": 1, "_TILT": 2, "_COUNT": 3, "_FRAME": 4}}}}]}}],
  "buffers": [{{"byteLength": {length}, "uri": "{uri}"}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 84, "byteLength": 12, "byteStride": 4}},
    {{"buffer": 0, "byteOffset": 96, "byteLength": 6}},
    {{"buffer": 0, "byteOffset": 104, "byteLength": 48}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4"}},
    {{"bufferView": 2, "componentType": 5120, "normalized": true, "count": 3, "type": "VEC2"}},
    {{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}},
    {{"bufferView": 4, "componentType": 5126, "count": 3, "type": "MAT2"}}
  ]
}}"#,
            length = data.len()
        )
    }

    #[test]
    fn the_attributes_of_the_application_are_read_by_their_names() {
        let scene =
            GltfScene::read_from_slice(attributes_file().as_bytes(), Path::new(".")).unwrap();
        let primitive = &scene.meshes[0].primitives[0];
        assert!(primitive.colors.is_none());
        // In the order of their names; the matrices left out.
        let names: Vec<&str> = primitive
            .attributes
            .iter()
            .map(|attribute| attribute.name.as_str())
            .collect();
        assert_eq!(names, ["_COUNT", "_TILT", "_WIND"]);
        assert!(primitive.attribute("_FRAME").is_none());
        assert!(primitive.attribute("WIND").is_none());

        let wind = primitive.attribute("_WIND").unwrap();
        assert_eq!(wind.components, 4);
        assert_eq!(
            wind.values,
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.25, 0.5, 0.75, 1.0]
            ]
        );

        // Normalised signed bytes: 127 is 1, -127 and -128 are -1.
        let tilt = primitive.attribute("_TILT").unwrap();
        assert_eq!(tilt.components, 2);
        let wanted = [[1.0, -1.0], [-1.0, 0.0], [64.0 / 127.0, 1.0]];
        for (got, wanted) in tilt.values.iter().zip(wanted) {
            assert!(
                (got[0] - wanted[0]).abs() < 1e-6 && (got[1] - wanted[1]).abs() < 1e-6,
                "{got:?}"
            );
            assert_eq!(got[2..], [0.0, 0.0]);
        }

        // Integers not normalised are their values.
        let count = primitive.attribute("_COUNT").unwrap();
        assert_eq!(count.components, 1);
        assert_eq!(
            count.values,
            [
                [0.0, 0.0, 0.0, 0.0],
                [7.0, 0.0, 0.0, 0.0],
                [65535.0, 0.0, 0.0, 0.0]
            ]
        );
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
