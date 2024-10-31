// domain module
#[doc=include_str!("domain/docs/about_domain.md")]
pub mod domain;
// geometry module
pub mod geometry;
// boundary module
#[doc=include_str!("boundary/docs/about_boundary.md")]
pub mod boundary;
// solver module
#[doc=include_str!("solver/docs/about_solver.md")]
pub mod solver;
// timestepping module
#[doc=include_str!("time_stepping/docs/about_time_stepping.md")]
pub mod time_stepping;
// linear algebra module
#[doc=include_str!("linalg/docs/about_vector.md")]
#[doc=include_str!("linalg/docs/about_matrix.md")]
pub mod linalg;
// input-output module
#[doc=include_str!("input_output/docs/about_input_output.md")]
pub mod input_output;
// extrusion module
#[doc=include_str!("extrusion/docs/about_extrusion.md")]
pub mod extrusion;
#[doc=include_str!("equation/docs/about_equation.md")]
// equation module
pub mod equation;

// test module
pub mod tests;

// re-exports
pub use geometry::{Geometry, CellShape, FaceShape};
pub use linalg::{Vector, Matrix};
pub use domain::{Arrow, MeshEntity, Section, Sieve, mesh::Mesh};