// domain module
#[doc=include_str!("domain/docs/domain_userguide.md")]
pub mod domain;

// geometry module
pub mod geometry;

// boundary module
#[doc=include_str!("boundary/docs/boundary_userguide.md")]
pub mod boundary;

// solver module
#[doc=include_str!("solver/docs/solver_userguide.md")]
pub mod solver;

// timestepping module
#[doc=include_str!("time_stepping/docs/time_stepping_userguide.md")]
pub mod time_stepping;

// linear algebra module
#[doc=include_str!("linalg/docs/linalg_userguide.md")]
pub mod linalg;

// input-output module
#[doc=include_str!("input_output/docs/input_output_userguide.md")]
pub mod input_output;

// extrusion module
#[doc=include_str!("extrusion/docs/extrusion_userguide.md")]
pub mod extrusion;

// equation module
#[doc=include_str!("equation/docs/equation_userguide.md")]
pub mod equation;

// interface adapters
#[doc=include_str!("interface_adapters/docs/interface_adapters_userguide.md")]
pub mod interface_adapters;

// use cases
#[doc=include_str!("use_cases/docs/use_cases_userguide.md")]
pub mod use_cases;

// test module
pub mod tests;

// re-exports
pub use geometry::{Geometry, CellShape, FaceShape};
pub use linalg::{Vector, Matrix};
pub use domain::{Arrow, MeshEntity, Section, Sieve, mesh::Mesh};