use crate::domain::{Element, Mesh};
use crate::boundary::BoundaryManager;
use nalgebra::Vector3;

/// Represents the flow field of the simulation, encapsulating the collection of elements
/// and managing the state of the simulation, including mass conservation.
pub struct FlowField {
    pub elements: Vec<Element>,  // Elements in the domain
    pub initial_mass: f64,  // Initial mass for mass conservation checks
    pub boundary_manager: BoundaryManager,  // Dynamically manage boundary conditions
}

impl FlowField {
    /// Creates a new FlowField, computes the initial mass.
    pub fn new(elements: Vec<Element>, boundary_manager: BoundaryManager) -> Self {
        let initial_mass: f64 = elements.iter().map(|e| e.mass).sum();
        Self {
            elements,
            initial_mass,
            boundary_manager,
        }
    }

    /// Updates boundary conditions via the BoundaryManager.
    pub fn update_boundary_conditions(&mut self, mesh: &mut Mesh, time: f64) {
        // Split the borrow to avoid conflicts
        let boundary_manager = std::mem::take(&mut self.boundary_manager);
        boundary_manager.apply(mesh, self, time);
        // Restore the boundary manager back to self after using it
        self.boundary_manager = boundary_manager;
    }

    /// Computes the mass of a given element.
    pub fn compute_mass(&self, element: &Element) -> f64 {
        element.mass
    }

    /// Computes the density of a given element.
    pub fn compute_density(&self, element: &Element) -> Option<f64> {
        if element.area > 0.0 {
            Some(element.mass / element.area)
        } else {
            None
        }
    }

    /// Checks mass conservation by comparing the current mass with the initial mass.
    pub fn check_mass_conservation(&self) -> Result<(), String> {
        let current_mass: f64 = self.elements.iter().map(|e| e.mass).sum();
        let relative_difference = ((current_mass - self.initial_mass) / self.initial_mass).abs();
        let tolerance = 1e-6;

        if relative_difference < tolerance {
            Ok(())
        } else {
            Err(format!(
                "Mass conservation failed: Initial mass = {}, Current mass = {}, Relative difference = {}",
                self.initial_mass, current_mass, relative_difference
            ))
        }
    }

    /// Retrieves element velocity.
    pub fn get_velocity(&self, element_id: u32) -> Option<Vector3<f64>> {
        self.elements.iter().find(|e| e.id == element_id).map(|e| e.velocity)
    }

    /// Retrieves element pressure.
    pub fn get_pressure(&self, element_id: u32) -> Option<f64> {
        self.elements.iter().find(|e| e.id == element_id).map(|e| e.pressure)
    }

    pub fn apply_boundary_conditions(&mut self, mesh: &mut Mesh, time_step: f64) {
        // Create a temporary reference to the boundary_manager
        let boundary_manager = std::mem::take(&mut self.boundary_manager);
    
        // Apply boundary conditions
        boundary_manager.apply(mesh, self, time_step);
    
        // Restore the boundary manager back to self after using it
        self.boundary_manager = boundary_manager;
    }
}
