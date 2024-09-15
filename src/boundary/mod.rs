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
    pub boundaries: HashMap<BoundaryType, Box<dyn BoundaryCondition>>,
}

impl BoundaryManager {
    /// Creates a new BoundaryManager.
    pub fn new() -> Self {
        Self {
            boundaries: HashMap::new(),
        }
    }

    pub fn register_boundary(&mut self, boundary_type: BoundaryType, condition: Box<dyn BoundaryCondition>) {
        self.boundaries.insert(boundary_type, condition);
    }

    /// Applies all registered boundary conditions to the mesh and flow field.
    pub fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        for condition in self.boundaries.values() {
            condition.apply(mesh, flow_field, time_step);
        }
    }
}
