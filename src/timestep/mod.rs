// src/timestep/mod.rs

pub mod euler;
pub mod cranknicolson;

pub use crate::domain::Mesh;
pub use euler::ExplicitEuler;
pub use cranknicolson::CrankNicolson;

/// Trait that defines the interface for time-stepping methods.
pub trait TimeStepper {
    /// Perform a time step on the given mesh.
    ///
    /// # Arguments
    /// * `mesh`: The computational mesh containing elements, faces, etc.
    /// * `dt`: The size of the time step to perform.
    fn step(&self, mesh: &mut Mesh, dt: f64);
}
