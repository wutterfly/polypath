//! Reading `.mtl` files: the materials that the `mtllib` and `usemtl` lines of an `.obj` file name.
//!
//! A `.mtl` file is a list of materials. Each starts with `newmtl name` and is followed by lines
//! that set its properties: colours (`Ka`, `Kd`, `Ks`, `Ke`), the shininess (`Ns`), how opaque it
//! is (`d`, or `Tr`, which is `1 - d`), the illumination model (`illum`) and texture maps
//! (`map_Kd` and friends). The physically based extension is read too (the one Blender and others
//! write): `Pr` and `Pm` (how rough and how metallic the surface is), and the maps `map_Pr`,
//! `map_Pm` and `norm` (a normal map).
//!
//! This reads what a renderer commonly uses and skips the rest (`Tf`, `Ni`, `map_Ns`, `disp`,
//! `decal`, ...). Keywords are not case sensitive. A line with a keyword that is not known is
//! skipped, and not an error, since the format has many extensions.
//!
//! The paths of texture maps are given as they are in the file, relative to the folder of the
//! `.mtl` file and with whatever separators the exporter used: loading the image is up to the
//! caller.
//!
//! # Example
//! ```rust
//! # use polypath::MtlLibrary;
//! let mtl = "newmtl red\nKd 1 0 0\nmap_Kd -s 2 2 1 textures/red.png\n";
//! let library = MtlLibrary::parse(mtl.as_bytes()).unwrap();
//!
//! let red = library.get("red").unwrap();
//! assert_eq!(red.diffuse, Some((1.0, 0.0, 0.0)));
//! let map = red.diffuse_map.as_ref().unwrap();
//! assert_eq!(map.path, "textures/red.png");
//! assert_eq!(map.scale, (2.0, 2.0));
//! ```

use std::{fs::File, io::BufReader, path::Path};

use rustc_hash::FxHashMap;

use crate::Error;

/// A texture map of a material: the image, and how it is put on the surface.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureMap {
    /// The path of the image, as written in the file.
    pub path: String,
    /// `-o`: how far the texture is moved, as u and v. `(0, 0)` if not given.
    pub offset: (f32, f32),
    /// `-s`: how much the texture is scaled, as u and v. `(1, 1)` if not given.
    pub scale: (f32, f32),
    /// `-clamp on`: the texture does not repeat, its edge pixels are used outside of it. `false`
    /// (it repeats) if not given.
    pub clamp: bool,
    /// `-bm`: how much a bump map is applied. Only for bump maps; `None` if not given.
    pub bump_multiplier: Option<f32>,
}

impl TextureMap {
    const fn new(path: String) -> Self {
        Self {
            path,
            offset: (0.0, 0.0),
            scale: (1.0, 1.0),
            clamp: false,
            bump_multiplier: None,
        }
    }
}

/// A material: how a surface looks. Everything the file does not say is `None`, so that the user
/// of the material decides what a missing property means.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// The name the `usemtl` lines of an `.obj` file refer to it by.
    pub name: String,
    /// `Ka`: the colour in the shade, as red, green and blue.
    pub ambient: Option<(f32, f32, f32)>,
    /// `Kd`: the colour of the surface where light hits it.
    pub diffuse: Option<(f32, f32, f32)>,
    /// `Ks`: the colour of the highlight.
    pub specular: Option<(f32, f32, f32)>,
    /// `Ke`: the colour of the light the surface gives off.
    pub emissive: Option<(f32, f32, f32)>,
    /// `Ns`: how sharp the highlight is (a higher number is a smaller one).
    pub shininess: Option<f32>,
    /// `d`, or `1 - Tr`: how opaque the surface is, from 0 (see through) to 1 (opaque). When the
    /// file gives both, the last one counts.
    pub dissolve: Option<f32>,
    /// `illum`: the illumination model, a number from 0 to 10.
    pub illumination: Option<u32>,
    /// `Pr`: how rough the surface is, from 0 (a mirror) to 1. Physically based extension.
    pub roughness: Option<f32>,
    /// `Pm`: how metallic the surface is, from 0 (not at all) to 1. Physically based extension.
    pub metallic: Option<f32>,
    /// `map_Ka`: a texture for the ambient colour.
    pub ambient_map: Option<TextureMap>,
    /// `map_Kd`: a texture for the diffuse colour, which is the colour of the surface.
    pub diffuse_map: Option<TextureMap>,
    /// `map_Ks`: a texture for the specular colour.
    pub specular_map: Option<TextureMap>,
    /// `map_Ke`: a texture for the emitted light.
    pub emissive_map: Option<TextureMap>,
    /// `map_d`: a texture for the opacity.
    pub alpha_map: Option<TextureMap>,
    /// `map_bump` or `bump`: a bump map.
    pub bump_map: Option<TextureMap>,
    /// `map_Pr`: a texture for the roughness (in its red channel, or grey).
    pub roughness_map: Option<TextureMap>,
    /// `map_Pm`: a texture for the metallic amount (in its red channel, or grey).
    pub metallic_map: Option<TextureMap>,
    /// `norm`: a normal map (the direction of the surface, as a colour in tangent space).
    pub normal_map: Option<TextureMap>,
}

impl Material {
    const fn new(name: String) -> Self {
        Self {
            name,
            ambient: None,
            diffuse: None,
            specular: None,
            emissive: None,
            shininess: None,
            dissolve: None,
            illumination: None,
            roughness: None,
            metallic: None,
            ambient_map: None,
            diffuse_map: None,
            specular_map: None,
            emissive_map: None,
            alpha_map: None,
            bump_map: None,
            roughness_map: None,
            metallic_map: None,
            normal_map: None,
        }
    }
}

/// The materials of a `.mtl` file, in the order of the file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MtlLibrary {
    materials: Vec<Material>,
    /// The place in `materials` of the last material of every name.
    by_name: FxHashMap<String, usize>,
}

/// The keywords that are read, in lower case. A line with another keyword is skipped.
const KEYWORDS: [&str; 22] = [
    "ka",
    "kd",
    "ks",
    "ke",
    "ns",
    "d",
    "tr",
    "illum",
    "map_ka",
    "map_kd",
    "map_ks",
    "map_ke",
    "map_d",
    "map_bump",
    "bump",
    "map_opacity",
    "map_kn",
    "pr",
    "pm",
    "map_pr",
    "map_pm",
    "norm",
];

impl MtlLibrary {
    /// Reads a `.mtl` file.
    ///
    /// # Errors
    /// - [`Error::Io`] if the file can not be read.
    /// - The errors of [`parse`](Self::parse).
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        Self::parse(BufReader::new(file))
    }

    /// Parses the text of a `.mtl` file from a reader.
    ///
    /// # Errors
    /// - [`Error::EmptyMtl`] for a `newmtl` without a name.
    /// - [`Error::MaterialLineBeforeNewmtl`] for a property before the first `newmtl`.
    /// - [`Error::UnexpectedEoL`], [`Error::ParseF`] and [`Error::ParseI`] for a property that is
    ///   missing a value or has one that is not a number.
    /// - [`Error::UnkownLine`] for a texture map with an option that is not known.
    pub fn parse(mut reader: impl std::io::BufRead) -> Result<Self, Error> {
        let mut library = Self::default();
        let mut buffer = String::with_capacity(256);

        loop {
            buffer.clear();
            if reader.read_line(&mut buffer)? == 0 {
                break;
            }

            let line = buffer.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (keyword, rest) = split_word(line);
            let keyword = keyword.to_ascii_lowercase();

            if keyword == "newmtl" {
                if rest.is_empty() {
                    return Err(Error::EmptyMtl);
                }
                library
                    .by_name
                    .insert(rest.to_owned(), library.materials.len());
                library.materials.push(Material::new(rest.to_owned()));
                continue;
            }

            // Lines that are not understood are skipped, wherever they are.
            if !KEYWORDS.contains(&keyword.as_str()) {
                continue;
            }
            let Some(material) = library.materials.last_mut() else {
                return Err(Error::MaterialLineBeforeNewmtl(line.to_owned()));
            };
            apply(material, &keyword, rest, line)?;
        }

        Ok(library)
    }

    /// The materials, in the order of the file.
    #[inline]
    #[must_use]
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    /// How many materials there are.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.materials.len()
    }

    /// Whether there are no materials.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// The material called `name`. If the file defines a name more than once, the last one.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Material> {
        self.by_name.get(name).map(|&index| &self.materials[index])
    }
}

/// Splits off the first word of `text`: the word, and what follows it with the space around it
/// taken off.
fn split_word(text: &str) -> (&str, &str) {
    let text = text.trim_start();
    text.split_once(char::is_whitespace)
        .map_or((text, ""), |(word, rest)| (word, rest.trim()))
}

/// Sets the property that the line `keyword rest` is of `material`.
fn apply(material: &mut Material, keyword: &str, rest: &str, line: &str) -> Result<(), Error> {
    match keyword {
        "ka" => material.ambient = parse_color(rest)?,
        "kd" => material.diffuse = parse_color(rest)?,
        "ks" => material.specular = parse_color(rest)?,
        "ke" => material.emissive = parse_color(rest)?,
        "ns" => material.shininess = Some(parse_number(rest)?),
        "d" => material.dissolve = Some(parse_dissolve(rest)?),
        "tr" => material.dissolve = Some(1.0 - parse_number(rest)?),
        "illum" => {
            material.illumination = Some(split_word(rest).0.parse::<u32>().map_err(Error::from)?);
        }
        "map_ka" => material.ambient_map = Some(parse_map(rest, line)?),
        "map_kd" => material.diffuse_map = Some(parse_map(rest, line)?),
        "map_ks" => material.specular_map = Some(parse_map(rest, line)?),
        "map_ke" => material.emissive_map = Some(parse_map(rest, line)?),
        "map_d" | "map_opacity" => material.alpha_map = Some(parse_map(rest, line)?),
        "map_bump" | "bump" | "map_kn" => material.bump_map = Some(parse_map(rest, line)?),
        "pr" => material.roughness = Some(parse_number(rest)?),
        "pm" => material.metallic = Some(parse_number(rest)?),
        "map_pr" => material.roughness_map = Some(parse_map(rest, line)?),
        "map_pm" => material.metallic_map = Some(parse_map(rest, line)?),
        "norm" => material.normal_map = Some(parse_map(rest, line)?),
        _ => unreachable!("only the keywords of KEYWORDS get here"),
    }
    Ok(())
}

/// The first number of `text`.
fn parse_number(text: &str) -> Result<f32, Error> {
    let word = split_word(text).0;
    if word.is_empty() {
        return Err(Error::UnexpectedEoL);
    }
    Ok(word.parse::<f32>()?)
}

/// A colour: three numbers, or one for a grey. `None` for the forms that give a colour as a
/// spectrum or in another colour space (`spectral`, `xyz`), which are not supported.
fn parse_color(text: &str) -> Result<Option<(f32, f32, f32)>, Error> {
    let mut words = text.split_whitespace();
    let Some(first) = words.next() else {
        return Err(Error::UnexpectedEoL);
    };
    if first.eq_ignore_ascii_case("spectral") || first.eq_ignore_ascii_case("xyz") {
        return Ok(None);
    }

    let r = first.parse::<f32>()?;
    let Some(g) = words.next() else {
        return Ok(Some((r, r, r)));
    };
    let g = g.parse::<f32>()?;
    let b = words.next().ok_or(Error::UnexpectedEoL)?.parse::<f32>()?;
    Ok(Some((r, g, b)))
}

/// The opacity of `d`, which can have a `-halo` in front of the number.
fn parse_dissolve(text: &str) -> Result<f32, Error> {
    let (word, rest) = split_word(text);
    if word.eq_ignore_ascii_case("-halo") {
        return parse_number(rest);
    }
    parse_number(text)
}

/// A texture map: options (that start with `-` and have values) and then the path, which is the
/// rest of the line (a path can have spaces in it).
fn parse_map(text: &str, line: &str) -> Result<TextureMap, Error> {
    let mut map = TextureMap::new(String::new());
    let mut rest = text;

    loop {
        let (word, after) = split_word(rest);
        if !word.starts_with('-') {
            break;
        }
        let option = word.to_ascii_lowercase();
        rest = after;
        match option.as_str() {
            "-blendu" | "-blendv" | "-cc" | "-imfchan" | "-type" | "-texres" => {
                rest = split_word(rest).1;
            }
            "-clamp" => {
                let (value, after) = split_word(rest);
                map.clamp = value.eq_ignore_ascii_case("on");
                rest = after;
            }
            "-bm" => {
                let (value, after) = split_word(rest);
                map.bump_multiplier = Some(value.parse::<f32>()?);
                rest = after;
            }
            "-mm" => {
                // Two numbers: a base and a gain. Not used.
                rest = split_word(split_word(rest).1).1;
            }
            "-o" | "-s" | "-t" => {
                let (values, after) = take_numbers(rest);
                rest = after;
                let u = *values.first().ok_or(Error::UnexpectedEoL)?;
                // v and w are optional: v is 0 for an offset and 1 for a scale if not given. (w
                // is for 3D textures, and not used.)
                match option.as_str() {
                    "-o" => map.offset = (u, values.get(1).copied().unwrap_or(0.0)),
                    "-s" => map.scale = (u, values.get(1).copied().unwrap_or(1.0)),
                    // `-t` is for turbulence: read and not used.
                    _ => {}
                }
            }
            _ => return Err(Error::UnkownLine(line.to_owned())),
        }
    }

    if rest.is_empty() {
        return Err(Error::UnexpectedEoL);
    }
    rest.clone_into(&mut map.path);
    Ok(map)
}

/// Up to three numbers from the start of `text`, and what follows them. Stops at the first word
/// that is not a number (which is the path, or the next option).
fn take_numbers(text: &str) -> (Vec<f32>, &str) {
    let mut numbers = Vec::with_capacity(3);
    let mut rest = text;
    while numbers.len() < 3 {
        let (word, after) = split_word(rest);
        match word.parse::<f32>() {
            Ok(number) if !word.is_empty() => {
                numbers.push(number);
                rest = after;
            }
            _ => break,
        }
    }
    (numbers, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> MtlLibrary {
        MtlLibrary::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn a_material_with_colours_shininess_and_illumination() {
        let library = parse(
            "newmtl steel\n\tNs 32\n\td 1\n\tTr 0\n\tTf 1 1 1\n\tillum 2\n\tKa 0.1 0.2 0.3\n\tKd 0.4 0.5 0.6\n\tKs 0.7 0.8 0.9\n",
        );

        let steel = library.get("steel").unwrap();
        assert_eq!(steel.name, "steel");
        assert_eq!(steel.shininess, Some(32.0));
        assert_eq!(steel.dissolve, Some(1.0));
        assert_eq!(steel.illumination, Some(2));
        assert_eq!(steel.ambient, Some((0.1, 0.2, 0.3)));
        assert_eq!(steel.diffuse, Some((0.4, 0.5, 0.6)));
        assert_eq!(steel.specular, Some((0.7, 0.8, 0.9)));
        assert_eq!(steel.emissive, None);
        assert_eq!(steel.diffuse_map, None);
    }

    #[test]
    fn the_materials_keep_the_order_of_the_file_and_are_found_by_name() {
        let library = parse("newmtl b\nKd 1 0 0\nnewmtl a\nKd 0 1 0\n");

        let names: Vec<_> = library
            .materials()
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(names, ["b", "a"]);
        assert_eq!(library.len(), 2);
        assert_eq!(library.get("a").unwrap().diffuse, Some((0.0, 1.0, 0.0)));
        assert!(library.get("c").is_none());
    }

    #[test]
    fn a_name_that_is_defined_twice_finds_the_last_one() {
        let library = parse("newmtl m\nKd 1 0 0\nnewmtl m\nKd 0 0 1\n");

        assert_eq!(library.len(), 2);
        assert_eq!(library.get("m").unwrap().diffuse, Some((0.0, 0.0, 1.0)));
    }

    #[test]
    fn a_name_can_have_spaces() {
        let library = parse("newmtl wood  planks\nKd 1 1 1\n");

        assert!(library.get("wood  planks").is_some());
    }

    #[test]
    fn a_single_number_is_a_grey_colour() {
        let library = parse("newmtl m\nKd 0.5\n");

        assert_eq!(library.get("m").unwrap().diffuse, Some((0.5, 0.5, 0.5)));
    }

    #[test]
    fn colours_in_other_forms_are_left_out() {
        let library = parse("newmtl m\nKd spectral file.rfl 1\nKa xyz 0.1 0.2 0.3\nKs 1 1 1\n");

        let m = library.get("m").unwrap();
        assert_eq!(m.diffuse, None);
        assert_eq!(m.ambient, None);
        assert_eq!(m.specular, Some((1.0, 1.0, 1.0)));
    }

    #[test]
    fn transparency_is_one_minus_dissolve_and_the_last_line_counts() {
        assert_eq!(
            parse("newmtl m\nTr 0.25\n").get("m").unwrap().dissolve,
            Some(0.75)
        );
        assert_eq!(
            parse("newmtl m\nd 0.5\n").get("m").unwrap().dissolve,
            Some(0.5)
        );
        assert_eq!(
            parse("newmtl m\nd -halo 0.3\n").get("m").unwrap().dissolve,
            Some(0.3)
        );
        assert_eq!(
            parse("newmtl m\nd 1\nTr 0.5\n").get("m").unwrap().dissolve,
            Some(0.5)
        );
    }

    #[test]
    fn a_texture_map_is_a_path() {
        let library = parse("newmtl m\nmap_Kd textures/wood.png\n");

        let map = library.get("m").unwrap().diffuse_map.clone().unwrap();
        assert_eq!(map.path, "textures/wood.png");
        assert_eq!(map.offset, (0.0, 0.0));
        assert_eq!(map.scale, (1.0, 1.0));
        assert!(!map.clamp);
        assert_eq!(map.bump_multiplier, None);
    }

    #[test]
    fn a_path_can_have_spaces_and_backslashes() {
        let library = parse("newmtl m\nmap_Kd My Textures\\old wood.png\n");

        let map = library.get("m").unwrap().diffuse_map.clone().unwrap();
        assert_eq!(map.path, "My Textures\\old wood.png");
    }

    #[test]
    fn the_options_of_a_texture_map_come_before_the_path() {
        let library =
            parse("newmtl m\nmap_Kd -o 0.5 0.25 -s 2 3 1 -clamp on -blendu off tile.png\n");

        let map = library.get("m").unwrap().diffuse_map.clone().unwrap();
        assert_eq!(map.path, "tile.png");
        assert_eq!(map.offset, (0.5, 0.25));
        assert_eq!(map.scale, (2.0, 3.0));
        assert!(map.clamp);
    }

    #[test]
    fn an_option_with_one_number_sets_only_u() {
        let library = parse("newmtl m\nmap_Kd -s 4 a.png\nmap_Ks -o 1 b.png\n");

        let m = library.get("m").unwrap();
        assert_eq!(m.diffuse_map.as_ref().unwrap().scale, (4.0, 1.0));
        assert_eq!(m.specular_map.as_ref().unwrap().offset, (1.0, 0.0));
    }

    #[test]
    fn a_path_that_starts_with_a_digit_is_not_taken_for_a_number() {
        let library = parse("newmtl m\nmap_Kd -s 2 2 1.png\n");

        let map = library.get("m").unwrap().diffuse_map.clone().unwrap();
        // The two numbers are the scale, and `1.png` is not a third one.
        assert_eq!(map.path, "1.png");
        assert_eq!(map.scale, (2.0, 2.0));
    }

    #[test]
    fn a_bump_map_has_a_multiplier() {
        let library = parse("newmtl m\nbump -bm 0.5 bumps.png\nmap_bump other.png\n");

        // The last of `bump` and `map_bump` counts.
        let map = library.get("m").unwrap().bump_map.clone().unwrap();
        assert_eq!(map.path, "other.png");
        let library = parse("newmtl m\nbump -bm 0.5 bumps.png\n");
        let map = library.get("m").unwrap().bump_map.clone().unwrap();
        assert_eq!(map.bump_multiplier, Some(0.5));
        assert_eq!(map.path, "bumps.png");
    }

    #[test]
    fn every_kind_of_map_goes_where_it_belongs() {
        let library =
            parse("newmtl m\nmap_Ka a\nmap_Kd d\nmap_Ks s\nmap_Ke e\nmap_d o\nmap_bump b\n");

        let m = library.get("m").unwrap();
        let path = |map: &Option<TextureMap>| map.as_ref().unwrap().path.clone();
        assert_eq!(path(&m.ambient_map), "a");
        assert_eq!(path(&m.diffuse_map), "d");
        assert_eq!(path(&m.specular_map), "s");
        assert_eq!(path(&m.emissive_map), "e");
        assert_eq!(path(&m.alpha_map), "o");
        assert_eq!(path(&m.bump_map), "b");
    }

    #[test]
    fn keywords_are_not_case_sensitive() {
        let library = parse("NEWMTL m\nKD 1 0 0\nMAP_KD x.png\nILLUM 2\n");

        let m = library.get("m").unwrap();
        assert_eq!(m.diffuse, Some((1.0, 0.0, 0.0)));
        assert_eq!(m.diffuse_map.as_ref().unwrap().path, "x.png");
        assert_eq!(m.illumination, Some(2));
    }

    #[test]
    fn comments_blank_lines_windows_line_ends_and_unknown_keywords_are_skipped() {
        let library = parse(
            "# made by a tool\r\n\r\nnewmtl m\r\nNi 1.45\r\nmap_Ns rough.png\r\nfancy_extension 1 2 3\r\nKd 1 1 1\r\n",
        );

        assert_eq!(library.len(), 1);
        assert_eq!(library.get("m").unwrap().diffuse, Some((1.0, 1.0, 1.0)));
    }

    #[test]
    fn an_unknown_keyword_before_any_material_is_skipped_too() {
        let library = parse("fancy_header 1\nnewmtl m\n");

        assert_eq!(library.len(), 1);
    }

    #[test]
    fn a_property_before_the_first_material_is_an_error() {
        let result = MtlLibrary::parse(b"Kd 1 0 0\nnewmtl m\n".as_slice());

        assert!(matches!(result, Err(Error::MaterialLineBeforeNewmtl(line)) if line == "Kd 1 0 0"));
    }

    #[test]
    fn a_material_without_a_name_is_an_error() {
        let result = MtlLibrary::parse(b"newmtl\n".as_slice());

        assert!(matches!(result, Err(Error::EmptyMtl)));
    }

    #[test]
    fn missing_or_bad_values_are_errors() {
        let bad = |text: &str| MtlLibrary::parse(format!("newmtl m\n{text}\n").as_bytes());

        assert!(matches!(bad("Kd"), Err(Error::UnexpectedEoL)));
        assert!(matches!(bad("Kd 1 2"), Err(Error::UnexpectedEoL)));
        assert!(matches!(bad("Kd 1 x 3"), Err(Error::ParseF(_))));
        assert!(matches!(bad("Ns"), Err(Error::UnexpectedEoL)));
        assert!(matches!(bad("illum two"), Err(Error::ParseI(_))));
        assert!(matches!(bad("map_Kd"), Err(Error::UnexpectedEoL)));
        assert!(matches!(bad("map_Kd -s 2"), Err(Error::UnexpectedEoL)));
        assert!(matches!(
            bad("map_Kd -wibble 2 a.png"),
            Err(Error::UnkownLine(_))
        ));
    }

    #[test]
    fn an_empty_file_has_no_materials() {
        let library = parse("");

        assert!(library.is_empty());
        assert_eq!(library.materials().len(), 0);
    }

    #[test]
    fn a_material_file_of_the_kind_3ds_max_writes() {
        // The file next to the tank model of the workspace.
        let library = parse(
            "# 3ds Max Wavefront OBJ Exporter v0.97b - (c)2007 guruware\n# File Created: 24.11.2019 01:41:08\n\nnewmtl wire_086086086\n\tNs 32\n\td 1\n\tTr 0\n\tTf 1 1 1\n\tillum 2\n\tKa 0.3373 0.3373 0.3373\n\tKd 0.3373 0.3373 0.3373\n\tKs 0.3500 0.3500 0.3500\n",
        );

        let m = library.get("wire_086086086").unwrap();
        assert_eq!(m.diffuse, Some((0.3373, 0.3373, 0.3373)));
        assert_eq!(m.specular, Some((0.35, 0.35, 0.35)));
        assert_eq!(m.shininess, Some(32.0));
        assert!(m.diffuse_map.is_none());
    }

    #[test]
    fn the_physically_based_extension_is_read() {
        let library = parse(
            "newmtl pbr\nPr 0.25\nPm 1\nmap_Pr rough.png\nmap_Pm -clamp on metal.png\nnorm -bm 2 normal.png\n",
        );

        let pbr = library.get("pbr").unwrap();
        assert_eq!(pbr.roughness, Some(0.25));
        assert_eq!(pbr.metallic, Some(1.0));
        assert_eq!(pbr.roughness_map.as_ref().unwrap().path, "rough.png");
        let metal = pbr.metallic_map.as_ref().unwrap();
        assert_eq!(metal.path, "metal.png");
        assert!(metal.clamp);
        let normal = pbr.normal_map.as_ref().unwrap();
        assert_eq!(normal.path, "normal.png");
        assert_eq!(normal.bump_multiplier, Some(2.0));
        // The older maps are not touched by it.
        assert_eq!(pbr.bump_map, None);
        assert_eq!(pbr.specular_map, None);
    }

    #[test]
    fn the_keywords_of_the_extension_are_not_case_sensitive_and_the_old_ones_still_work() {
        let library = parse("newmtl m\nPR 0.5\nMAP_PM m.png\nNORM n.png\nmap_Kd d.png\nns 8\n");

        let m = library.get("m").unwrap();
        assert_eq!(m.roughness, Some(0.5));
        assert_eq!(m.metallic_map.as_ref().unwrap().path, "m.png");
        assert_eq!(m.normal_map.as_ref().unwrap().path, "n.png");
        assert_eq!(m.diffuse_map.as_ref().unwrap().path, "d.png");
        assert_eq!(m.shininess, Some(8.0));
        assert_eq!(m.metallic, None);
    }

    #[test]
    fn a_number_that_is_missing_or_not_a_number_is_an_error_for_the_extension_too() {
        assert!(MtlLibrary::parse(&b"newmtl m\nPr\n"[..]).is_err());
        assert!(MtlLibrary::parse(&b"newmtl m\nPm rough\n"[..]).is_err());
    }
}
