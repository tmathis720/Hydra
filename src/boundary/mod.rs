use std::collections::HashMap;

use crate::domain::{FlowField, Mesh};

pub mod types;
pub use types::{BoundaryType, BoundaryCondition};

// Import specific boundary condition implementations
pub mod inflow;
pub mod outflow;
pub mod no_slip;
pub mod periodic;
pub mod free_surface;
pub mod open;

pub use inflow::InflowBoundaryCondition;
pub use outflow::OutflowBoundaryCondition;
pub use no_slip::NoSlipBoundaryCondition;
pub use periodic::PeriodicBoundaryCondition;
pub use free_surface::FreeSurfaceBoundaryCondition;
pub use open::OpenBoundaryCondition;

#[derive(Default)]
/// Manages all boundary conditions in the simulation.
pub struct BoundaryManager {
    /// A hashmap storing boundary conditions, mapped to the BoundaryType.
    pub boundaries: HashMap<BoundaryType, Box<dyn BoundaryCondition>>,
}

impl BoundaryManager {
    /// Creates a new BoundaryManager.
    pub fn new() -> Self {
        Self {
            boundaries: HashMap::new(),
        }
    }

    /// Registers a new boundary condition with the manager.
    ///
    /// # Arguments
    /// - `boundary_type`: The type of the boundary (e.g., FreeSurface, Inflow, etc.)
    /// - `condition`: The boundary condition to register, implementing the `BoundaryCondition` trait.
    pub fn register_boundary(&mut self, boundary_type: BoundaryType, condition: Box<dyn BoundaryCondition>) {
        self.boundaries.insert(boundary_type, condition);
    }

    /// Applies all registered boundary conditions to the mesh and flow field.
    ///
    /// # Arguments
    /// - `mesh`: The computational mesh.
    /// - `flow_field`: The flow field representing velocities, pressures, etc.
    /// - `time_step`: The current simulation time step.
    pub fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        for condition in self.boundaries.values() {
            condition.apply(mesh, flow_field, time_step);
        }
    }

    /// Get a list of elements associated with inflow boundary conditions.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for inflow elements.
    ///
    /// Returns a vector of element IDs.
    pub fn get_inflow_elements(&self, mesh: &Mesh) -> Vec<u32> {
        self.get_boundary_elements(mesh, BoundaryType::Inflow)
    }

    /// Get a list of elements associated with outflow boundary conditions.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for outflow elements.
    ///
    /// Returns a vector of element IDs.
    pub fn get_outflow_elements(&self, mesh: &Mesh) -> Vec<u32> {
        self.get_boundary_elements(mesh, BoundaryType::Outflow)
    }

    /// Get a list of faces associated with the free surface boundary condition.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for free surface faces.
    ///
    /// Returns a vector of face IDs.
    pub fn get_free_surface_faces(&self, mesh: &Mesh) -> Vec<u32> {
        self.get_boundary_elements(mesh, BoundaryType::FreeSurface)
    }

    /// Retrieves elements (or faces) for a specific boundary type.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to query.
    /// - `boundary_type`: The type of boundary (e.g., Inflow, FreeSurface).
    ///
    /// Returns a vector of element or face IDs.
    fn get_boundary_elements(&self, mesh: &Mesh, boundary_type: BoundaryType) -> Vec<u32> {
        if let Some(boundary_condition) = self.boundaries.get(&boundary_type) {
            boundary_condition.get_boundary_elements(mesh)
        } else {
            Vec::new()  // Return an empty vector if no boundary of that type is registered.
        }
    }
}
