use crate::{Error, ObjObject};

use std::mem;

impl ObjObject {
    pub fn parse(mut reader: impl std::io::BufRead) -> Result<Self, Error> {
        let mut buffer = String::with_capacity(256);

        let mut vertices = Vec::with_capacity(64);
        let mut vertex_colors = Vec::new();
        let mut vertex_normals = Vec::new();
        let mut texture_coords = Vec::new();
        let mut faces = Vec::with_capacity(32);

        let mut groups = Vec::new();
        let mut objects = Vec::new();

        let mut current_group = GroupingData::default();
        let mut current_object = GroupingData::default();

        loop {
            let read = reader.read_line(&mut buffer)?;

            if read == 0 {
                break;
            }

            let v_count = vertices.len() as u32;
            let t_count = texture_coords.len() as u32;
            let n_count = vertex_normals.len() as u32;

            let line = Self::parse_line(&buffer[..read], v_count, t_count, n_count)?;
            buffer.clear();

            match line {
                Line::Empty | Line::Comment => continue,
                Line::Vertex(vertex_data) => {
                    vertices.push(vertex_data.position);
                    if let Some(color) = vertex_data.color {
                        vertex_colors.push(color);
                    }
                }
                Line::Normal(normal) => vertex_normals.push(normal),
                Line::TextureCoord(tex) => texture_coords.push(tex),
                Line::Face(face_data) => {
                    faces.push(face_data);
                    current_group.finish += 1;
                }
                Line::DoubleFace(f1, f2) => {
                    faces.push(f1);
                    faces.push(f2);
                    current_group.finish += 2;
                }
                Line::Group(data) => {
                    if current_group.start == current_group.finish {
                        current_group.name = data;
                    } else {
                        let finished = mem::take(&mut current_group);
                        groups.push(finished);

                        current_group.name = data;
                        current_group.start = faces.len();
                        current_group.finish = faces.len();

                        current_object.finish += 1;
                    }
                }
                Line::Object(data) => {
                    if current_object.start == current_object.finish && faces.is_empty() {
                        current_object.name = data;
                    } else {
                        if current_group.start != current_group.finish {
                            current_object.finish += 1;

                            let finished = mem::take(&mut current_group);
                            groups.push(finished);

                            current_group.start = faces.len();
                            current_group.finish = faces.len();
                        }

                        let finished = mem::take(&mut current_object);
                        objects.push(finished);

                        current_object.name = data;
                        current_object.start = groups.len();
                        current_object.finish = groups.len();
                    }
                }

                Line::MaterialLib(data) => {
                    if current_object.mtl.is_none() {
                        current_object.mtl = Some(data);
                    } else {
                        return Err(Error::OjectMultipleMtl(current_object.name));
                    }
                }
                Line::MaterialUse(data) => {
                    if current_group.mtl.is_none() {
                        current_group.mtl = Some(data);
                    } else {
                        return Err(Error::GroupMultipleMTl(current_group.name));
                    }
                }
            }
        }

        // store current group
        if current_group.start != current_group.finish {
            current_object.finish += 1;
            let finished = std::mem::take(&mut current_group);
            groups.push(finished);
        }

        // store current object
        if current_object.start != current_object.finish {
            let finished = std::mem::take(&mut current_object);
            objects.push(finished);
        }

        Ok(Self {
            vertices,
            vertex_colors,
            vertex_normals,
            texture_coords,
            faces,

            groups,
            objects,
        })
    }

    fn parse_line(line: &str, v_count: u32, t_count: u32, n_count: u32) -> Result<Line, Error> {
        let line = line.trim();

        if line.is_empty() {
            return Ok(Line::Empty);
        }

        if line.starts_with('#') {
            return Ok(Line::Comment);
        }

        let t: &[u8] = line.as_bytes();
        let out = match t {
            [b'v', b' ', ..] => Line::Vertex(Self::parse_vertex(line[2..].trim())?),
            [b'v', b'n', b' ', ..] => Line::Normal(Self::parse_normal(line[3..].trim())?),
            [b'v', b't', b' ', ..] => {
                Line::TextureCoord(Self::parse_texture_coord(line[3..].trim())?)
            }
            [b'f', b' ', ..] => {
                let (f1, f2) = Self::parse_face(line[2..].trim(), v_count, t_count, n_count)?;
                f2.map_or(Line::Face(f1), |f2| Line::DoubleFace(f1, f2))
            }
            [b'o', b' ', ..] => Line::Object(Self::parse_grouping(line[2..].trim())),
            [b'g', b' ', ..] => Line::Group(Self::parse_grouping(line[2..].trim())),
            [b's', b' ', ..] => todo!("smooth"),
            [b'm', b't', b'l', b'l', b'i', b'b', b' ', ..] => {
                Line::MaterialLib(Self::parse_mtl(line[7..].trim())?)
            }
            [b'u', b's', b'e', b'm', b't', b'l', b' ', ..] => {
                Line::MaterialUse(Self::parse_mtl(line[7..].trim())?)
            }
            _ => return Err(Error::UnkownLine(String::from(line))),
        };

        Ok(out)
    }

    fn parse_vertex(data: &str) -> Result<VertexData, Error> {
        let mut split = data.split_whitespace();

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let x = str.parse::<f32>()?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let y = str.parse::<f32>()?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let z = str.parse::<f32>()?;

        if let Some(str) = split.next() {
            let a = str.parse::<f32>()?;

            let str = split.next().ok_or(Error::NonUniformColors)?;
            let b = str.parse::<f32>()?;

            let str = split.next().ok_or(Error::NonUniformColors)?;
            let c = str.parse::<f32>()?;

            return Ok(VertexData {
                position: (x, y, z),
                color: Some((a, b, c)),
            });
        }

        Ok(VertexData {
            position: (x, y, z),
            color: None,
        })
    }

    fn parse_normal(data: &str) -> Result<(f32, f32, f32), Error> {
        let mut split = data.split_whitespace();

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let x = str.parse::<f32>()?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let y = str.parse::<f32>()?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let z = str.parse::<f32>()?;

        Ok((x, y, z))
    }

    fn parse_texture_coord(data: &str) -> Result<(f32, f32), Error> {
        let mut split = data.split_whitespace();

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let x = str.parse::<f32>()?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let y = str.parse::<f32>()?;

        Ok((x, y))
    }

    fn parse_face(
        data: &str,
        v_count: u32,
        t_count: u32,
        n_count: u32,
    ) -> Result<(FaceData, Option<FaceData>), Error> {
        // i t n
        fn parse_single(
            data: &str,
            v_count: u32,
            t_count: u32,
            n_count: u32,
        ) -> Result<(u32, Option<u32>, Option<u32>), Error> {
            let mut split = data.split('/');

            // vertex index
            let str = split.next().ok_or(Error::UnexpectedEoL)?;
            let i = str.parse::<i32>()?;
            let i = if i < 0 {
                // negativ index, meaning
                // => -1 = v_count
                // => -2 = v_count - 1
                (v_count as i32 + (i + 1)) as u32
            } else {
                i as u32
            };

            // texture index
            let t = match split.next() {
                None => return Ok((i, None, None)),

                // ....//0980
                Some("") => None,

                // 986/0980...
                Some(str) => {
                    let t = str.parse::<i32>()?;
                    let t = if t < 0 {
                        // negativ index
                        (t_count as i32 + (t + 1)) as u32
                    } else {
                        t as u32
                    };

                    Some(t)
                }
            };

            // normal index
            let n = match split.next() {
                None => return Ok((i, t, None)),

                // .../.../1231
                Some(str) => {
                    let n = str.parse::<i32>()?;
                    let n = if n < 0 {
                        // negativ index
                        (n_count as i32 + (n + 1)) as u32
                    } else {
                        n as u32
                    };

                    Some(n)
                }
            };

            Ok((i, t, n))
        }

        let mut split = data.split_whitespace();

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let (i1, t1, n1) = parse_single(str, v_count, t_count, n_count)?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let (i2, t2, n2) = parse_single(str, v_count, t_count, n_count)?;

        let str = split.next().ok_or(Error::UnexpectedEoL)?;
        let (i3, t3, n3) = parse_single(str, v_count, t_count, n_count)?;

        let normal = match (n1, n2, n3) {
            (None, None, None) => None,
            (Some(n1), Some(n2), Some(n3)) => Some((n1, n2, n3)),
            _ => unreachable!(""),
        };

        let texture = match (t1, t2, t3) {
            (None, None, None) => None,
            (Some(t1), Some(t2), Some(t3)) => Some((t1, t2, t3)),
            _ => unreachable!(""),
        };

        // check for 4th vertex
        if let Some(str) = split.next() {
            let (i4, t4, n4) = parse_single(str, v_count, t_count, n_count)?;

            let normals = normal.map(|(n1, n2, n3)| [n1, n2, n3, n4.unwrap()]);
            let texture = texture.map(|(t1, t2, t3)| [t1, t2, t3, t4.unwrap()]);
            let [f1, f2] = Self::triangulate([i1, i2, i3, i4], normals, texture);

            return Ok((f1, Some(f2)));
        }

        Ok((
            FaceData {
                indicies: (i1, i2, i3),
                normal_indicies: normal,
                texture_indcicies: texture,
            },
            None,
        ))
    }

    fn parse_grouping(data: &str) -> String {
        let trimmed = data.trim();
        trimmed.to_string()
    }

    fn parse_mtl(data: &str) -> Result<String, Error> {
        let str = data.trim();
        Ok(str.to_owned())
    }

    const fn triangulate(
        index: [u32; 4],
        normals: Option<[u32; 4]>,
        texture: Option<[u32; 4]>,
    ) -> [FaceData; 2] {
        let i1 = (index[0], index[1], index[2]);
        let i2 = (index[0], index[2], index[3]);

        let (n1, n2) = match normals {
            Some(normals) => {
                let n1 = (normals[0], normals[1], normals[2]);
                let n2 = (normals[0], normals[2], normals[3]);

                (Some(n1), Some(n2))
            }
            None => (None, None),
        };

        let (t1, t2) = match texture {
            Some(texture) => {
                let n1 = (texture[0], texture[1], texture[2]);
                let n2 = (texture[0], texture[2], texture[3]);

                (Some(n1), Some(n2))
            }
            None => (None, None),
        };

        [
            FaceData {
                indicies: i1,
                texture_indcicies: t1,
                normal_indicies: n1,
            },
            FaceData {
                indicies: i2,
                texture_indcicies: t2,
                normal_indicies: n2,
            },
        ]
    }
}

#[derive(Debug, Clone)]
pub enum Line {
    Empty,
    Comment,
    Vertex(VertexData),
    Normal((f32, f32, f32)),
    TextureCoord((f32, f32)),
    Face(FaceData),
    DoubleFace(FaceData, FaceData),
    MaterialLib(String),
    MaterialUse(String),
    Group(String),
    Object(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexData {
    pub position: (f32, f32, f32),
    pub color: Option<(f32, f32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceData {
    // for vertex & color
    pub(crate) indicies: (u32, u32, u32),
    pub(crate) texture_indcicies: Option<(u32, u32, u32)>,
    pub(crate) normal_indicies: Option<(u32, u32, u32)>,
}

#[derive(Debug, Clone, Default)]
pub struct GroupingData {
    pub(crate) name: String,
    pub(crate) mtl: Option<String>,
    pub(crate) start: usize,
    pub(crate) finish: usize,
}

#[cfg(test)]
mod tests {
    use crate::ObjObject;
    use crate::parse::FaceData;

    #[test]
    fn test_vertex_no_color() {
        let line = "1.0  1.2  0.0";

        let res = ObjObject::parse_vertex(line).unwrap();

        assert_eq!(res.position, (1.0, 1.2, 0.0));
        assert!(res.color.is_none());
    }

    #[test]
    fn test_vertex_with_color() {
        let line = "1.0  1.2  0.0  255.0 123.0 90.0";

        let res = ObjObject::parse_vertex(line).unwrap();

        assert_eq!(res.position, (1.0, 1.2, 0.0));
        assert_eq!(res.color, Some((255.0, 123.0, 90.0)));
    }

    #[test]
    fn test_normal() {
        let line = "0.5 0.0 -1.0";

        let res = ObjObject::parse_normal(line).unwrap();

        assert_eq!(res, (0.5, 0.0, -1.0));
    }

    #[test]
    fn test_face_itn() {
        let line = "123/5445/123 456/123/1231 789/113/12";

        let (res, f2) = ObjObject::parse_face(line, 0, 0, 0).unwrap();
        assert!(f2.is_none());
        assert_eq!(
            res,
            FaceData {
                indicies: (123, 456, 789),
                texture_indcicies: Some((5445, 123, 113)),
                normal_indicies: Some((123, 1231, 12))
            }
        );
    }

    #[test]
    fn test_face_it() {
        let line = "123/5445 456/123 789/113";

        let (res, f2) = ObjObject::parse_face(line, 0, 0, 0).unwrap();
        assert!(f2.is_none());
        assert_eq!(
            res,
            FaceData {
                indicies: (123, 456, 789),
                texture_indcicies: Some((5445, 123, 113)),
                normal_indicies: None,
            }
        );
    }

    #[test]
    fn test_face_i() {
        let line = "123 456 789";

        let (res, f2) = ObjObject::parse_face(line, 0, 0, 0).unwrap();
        assert!(f2.is_none());
        assert_eq!(
            res,
            FaceData {
                indicies: (123, 456, 789),
                texture_indcicies: None,
                normal_indicies: None,
            }
        );
    }

    #[test]
    fn test_face_in() {
        let line = "123//123 456//1231 789//12";

        let (res, f2) = ObjObject::parse_face(line, 0, 0, 0).unwrap();
        assert!(f2.is_none());
        assert_eq!(
            res,
            FaceData {
                indicies: (123, 456, 789),
                texture_indcicies: None,
                normal_indicies: Some((123, 1231, 12)),
            }
        );
    }

    #[test]
    fn test_face_negativ() {
        //                 i  t  n  i  t  n  i  t  n
        let line = "-2/-3/-1 -1/-1/-1 -5/-2/-3";

        let (res, f2) = ObjObject::parse_face(line, 10, 4, 7).unwrap();
        assert!(f2.is_none());
        assert_eq!(
            res,
            FaceData {
                indicies: (9, 10, 6),
                texture_indcicies: Some((2, 4, 3)),
                normal_indicies: Some((7, 7, 5)),
            }
        );
    }

    #[test]
    fn test_face_double() {
        let line = "123/5445/123 456/123/1231 789/113/12 509/111/576";

        let (f1, f2) = ObjObject::parse_face(line, 0, 0, 0).unwrap();
        assert_eq!(
            f1,
            FaceData {
                indicies: (123, 456, 789),
                texture_indcicies: Some((5445, 123, 113)),
                normal_indicies: Some((123, 1231, 12))
            }
        );

        assert_eq!(
            f2,
            Some(FaceData {
                indicies: (123, 789, 509),
                texture_indcicies: Some((5445, 113, 111)),
                normal_indicies: Some((123, 12, 576)),
            })
        );
    }
}
