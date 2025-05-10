#![warn(clippy::nursery)]
#![warn(clippy::cargo)]

pub mod meshlet;
pub mod opt;

mod obj;
mod parse;

pub use obj::Face;
pub use obj::MaterialIdent;
pub use obj::ObjObject;
pub use obj::VertexData;
pub use obj::VertexTextureData;

use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug)]
/// Represents different kind of errors that can happen while reading and parsing a .obj object.
pub enum Error {
    Io(std::io::Error),
    UnkownLine(String),
    UnexpectedEoL,
    ParseF(ParseFloatError),
    ParseI(ParseIntError),
    EmptyMtl,
    OjectMultipleMtl(String),
    GroupMultipleMTl(String),
    NonUniformColors,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => writeln!(f, "{error}"),
            Self::UnkownLine(line) => writeln!(f, "Encounterd a unknown line: [{line}]"),
            Self::UnexpectedEoL => writeln!(f, "Unexpected end-of-line"),
            Self::ParseF(error) => writeln!(f, "{error}"),
            Self::ParseI(error) => writeln!(f, "{error}"),
            Self::EmptyMtl => writeln!(f, "Empty material [lib/use]"),
            Self::OjectMultipleMtl(object) => {
                writeln!(f, "Multiple material lib defined for object [{object}]")
            }
            Self::GroupMultipleMTl(group) => {
                writeln!(f, "Multiple material uses defined for group [{group}]")
            }
            Self::NonUniformColors => {
                writeln!(
                    f,
                    "Vertex colors are specified for some vertices, but not all"
                )
            }
        }
    }
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ParseFloatError> for Error {
    #[inline]
    fn from(value: ParseFloatError) -> Self {
        Self::ParseF(value)
    }
}

impl From<ParseIntError> for Error {
    #[inline]
    fn from(value: ParseIntError) -> Self {
        Self::ParseI(value)
    }
}

pub trait Vertex {
    fn position(&self) -> (f32, f32, f32);
}

impl Vertex for VertexTextureData {
    #[inline]
    fn position(&self) -> (f32, f32, f32) {
        self.vertex.position
    }
}
