// src/timestep/mod.rs

pub mod euler;
pub mod cranknicolson;

pub use crate::domain::Mesh;
pub use euler::ExplicitEuler;
pub use cranknicolson::CrankNicolson;
