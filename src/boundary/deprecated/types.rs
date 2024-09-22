use crate::domain::{FlowField, Mesh};
use nalgebra::Vector3;

/// Enum to represent different types of boundary conditions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryType {
    Inflow,
    Outflow,
    NoSlip,
    FreeSurface, // Free surface boundary type
    Periodic,
    Reflective,
    Open,
}

/// Trait for boundary conditions.
pub trait BoundaryCondition {
    /// Updates the boundary condition based on simulation time or other parameters.
    fn update(&mut self, time: f64);

    /// Applies the boundary condition to the mesh and flow field.
    fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64);

    /// Retrieves the velocity associated with the boundary condition (if applicable).
    fn velocity(&self) -> Option<Vector3<f64>>;

    /// Retrieves the mass rate associated with the boundary condition (if applicable).
    fn mass_rate(&self) -> Option<f64>;

    /// Retrieves the boundary element IDs associated with this boundary condition.
    /// Can refer to either **faces** or **elements** based on the boundary condition.
    fn get_boundary_elements(&self, mesh: &Mesh) -> Vec<u32>;
}
