pub mod euler;
pub mod cranknicolson;
pub use crate::domain::mesh::Mesh;

pub use euler::ExplicitEuler;
pub use cranknicolson::CrankNicolson;
pub use crate::solver::Solver;


pub trait TimeStepper {
    fn step(&self, mesh: &mut Mesh, solver: &mut dyn Solver);
}