#[doc=include_str!("domain/docs/about_domain.md")]
pub mod domain;
pub mod geometry;
pub mod boundary;
pub mod solver;
pub mod time_stepping;
pub mod linalg;
//pub mod input_output;
pub mod tests;

pub use geometry::{Geometry, CellShape, FaceShape};
pub use linalg::{Vector, Matrix};
pub use domain::{Arrow, MeshEntity, Section, Sieve, mesh::Mesh};