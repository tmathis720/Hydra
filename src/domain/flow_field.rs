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

    pub fn get_inflow_velocity(&self, mesh: &mut Mesh) -> Option<Vector3<f64>> {
        if let Some(inflow_element_id) = self.boundary_manager.get_inflow_elements(&mesh).first() {
            self.get_velocity(*inflow_element_id)
        } else {
            None
        }
    }

    pub fn get_outflow_velocity(&self, mesh: &mut Mesh) -> Option<Vector3<f64>> {
        if let Some(outflow_element_id) = self.boundary_manager.get_outflow_elements(&mesh).first() {
            self.get_velocity(*outflow_element_id)
        } else {
            None
        }
    }

    pub fn get_inflow_mass_rate(&self, mesh: &mut Mesh) -> f64 {
        // Retrieve inflow elements and compute total inflow mass rate
        let inflow_elements = self.boundary_manager.get_inflow_elements(mesh);
        let velocity = self.get_inflow_velocity(mesh).unwrap_or(Vector3::zeros());
        let area = self.calculate_inflow_area(mesh, &inflow_elements); // Calculate inflow area based on inflow elements
        let density = self.calculate_inflow_density(&inflow_elements); // Calculate inflow density based on inflow elements
    
        // Return the mass flow rate: density * velocity * area
        density * velocity.magnitude() * area
    }

    pub fn get_outflow_mass_rate(&self, mesh: &mut Mesh) -> f64 {
        // Retrieve outflow elements and compute total outflow mass rate
        let outflow_elements = self.boundary_manager.get_outflow_elements(mesh);
        let velocity = self.get_outflow_velocity(mesh).unwrap_or(Vector3::zeros());
        let area = self.calculate_outflow_area(mesh, &outflow_elements); // Calculate outflow area based on outflow elements
        let density = self.calculate_outflow_density(&outflow_elements); // Calculate outflow density based on outflow elements
    
        // Return the mass flow rate: density * velocity * area
        density * velocity.magnitude() * area
    }

    pub fn calculate_inflow_area(&self, mesh: &Mesh, inflow_elements: &[u32]) -> f64 {
        inflow_elements.iter().fold(0.0, |total_area, &element_id| {
            if let Some(element) = mesh.get_element_by_id(element_id) {
                total_area + element.area // Sum the area of all inflow elements
            } else {
                total_area
            }
        })
    }

    pub fn calculate_outflow_area(&self, mesh: &Mesh, outflow_elements: &[u32]) -> f64 {
        outflow_elements.iter().fold(0.0, |total_area, &element_id| {
            if let Some(element) = mesh.get_element_by_id(element_id) {
                total_area + element.area // Sum the area of all outflow elements
            } else {
                total_area
            }
        })
    }

    pub fn calculate_inflow_density(&self, inflow_elements: &[u32]) -> f64 {
        let total_density: f64 = inflow_elements.iter().fold(0.0, |total_density, &element_id| {
            if let Some(element) = self.elements.iter().find(|e| e.id == element_id) {
                total_density + element.compute_density().unwrap_or(0.0)
            } else {
                total_density
            }
        });

        // Average density across inflow elements
        total_density / inflow_elements.len() as f64
    }

    pub fn calculate_outflow_density(&self, outflow_elements: &[u32]) -> f64 {
        let total_density: f64 = outflow_elements.iter().fold(0.0, |total_density, &element_id| {
            if let Some(element) = self.elements.iter().find(|e| e.id == element_id) {
                total_density + element.compute_density().unwrap_or(0.0)
            } else {
                total_density
            }
        });

        // Average density across outflow elements
        total_density / outflow_elements.len() as f64
    }
}
