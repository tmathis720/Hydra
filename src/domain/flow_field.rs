// src/domain/flow_field.rs

use crate::domain::{Element, Mesh};
use crate::boundary::{BoundaryCondition, InflowBoundaryCondition, OutflowBoundaryCondition};
use nalgebra::Vector3;

/// Represents the flow field of the simulation, encapsulating the collection of elements
/// and managing the state of the simulation, including mass conservation.
pub struct FlowField {
    pub elements: Vec<Element>,
    pub initial_mass: f64,
    pub inflow_boundary: InflowBoundaryCondition,
    pub outflow_boundary: OutflowBoundaryCondition,
}

impl FlowField {
    /// Creates a new flow field and computes the initial mass.
    ///
    /// # Parameters
    /// - `elements`: Vector of `Element` instances representing the simulation domain.
    /// - `inflow_boundary`: Inflow boundary condition.
    /// - `outflow_boundary`: Outflow boundary condition.
    ///
    /// # Returns
    /// A new `FlowField` instance.
    pub fn new(
        elements: Vec<Element>,
        inflow_boundary: InflowBoundaryCondition,
        outflow_boundary: OutflowBoundaryCondition,
    ) -> Self {
        let initial_mass: f64 = elements.iter().map(|e| e.mass).sum();
        Self {
            elements,
            initial_mass,
            inflow_boundary,
            outflow_boundary,
        }
    }

    /// Computes the mass of a given element.
    ///
    /// # Parameters
    /// - `element`: Reference to an `Element`.
    ///
    /// # Returns
    /// Mass of the element (units: kilograms).
    pub fn compute_mass(&self, element: &Element) -> f64 {
        element.mass
    }

    /// Computes the density of a given element.
    ///
    /// # Parameters
    /// - `element`: Reference to an `Element`.
    ///
    /// # Returns
    /// Density of the element (units: kg/m³) if area is positive; otherwise, `None`.
    pub fn compute_density(&self, element: &Element) -> Option<f64> {
        if element.area > 0.0 {
            Some(element.mass / element.area)
        } else {
            None
        }
    }

    /// Checks if mass conservation holds by comparing the current mass with the initial mass.
    ///
    /// # Returns
    /// `Ok(())` if mass conservation holds within the specified tolerance.
    /// `Err` with a message if mass conservation fails.
    pub fn check_mass_conservation(&self) -> Result<(), String> {
        let current_mass: f64 = self.elements.iter().map(|e| e.mass).sum();
        let relative_difference = ((current_mass - self.initial_mass) / self.initial_mass).abs();
        let tolerance = 1e-6; // Relative tolerance

        if relative_difference < tolerance {
            Ok(())
        } else {
            Err(format!(
                "Mass conservation check failed: Initial mass = {}, Current mass = {}, Relative difference = {}",
                self.initial_mass, current_mass, relative_difference
            ))
        }
    }

    /// Updates boundary conditions for the flow field.
    ///
    /// # Parameters
    /// - `time`: Current simulation time.
    pub fn update_boundary_conditions(&mut self, time: f64) {
        // Update inflow boundary condition based on simulation time or other parameters
        self.inflow_boundary.update(time);

        // Update outflow boundary condition
        self.outflow_boundary.update(time);
    }

    /// Applies boundary conditions to the elements in the flow field.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh containing elements and faces.
    pub fn apply_boundary_conditions(self, mesh: &mut Mesh) {
        // Apply inflow boundary condition
        self.inflow_boundary.apply(mesh, self, 0.0);  // Pass the entire FlowField

        // Apply outflow boundary condition
        self.outflow_boundary.apply(mesh, self, 0.0);  // Pass the entire FlowField
    }

    /// Retrieves the inflow velocity from the boundary condition.
    ///
    /// # Returns
    /// Inflow velocity vector (units: m/s).
    pub fn get_inflow_velocity(&self) -> Vector3<f64> {
        self.inflow_boundary.velocity().unwrap_or(Vector3::zeros())
    }

    /// Retrieves the outflow velocity from the boundary condition.
    ///
    /// # Returns
    /// Outflow velocity vector (units: m/s).
    pub fn get_outflow_velocity(&self) -> Vector3<f64> {
        self.outflow_boundary.velocity().unwrap_or(Vector3::zeros())
    }

    /// Retrieves the inflow mass rate from the boundary condition.
    ///
    /// # Returns
    /// Inflow mass rate (units: kg/s).
    pub fn get_inflow_mass_rate(&self) -> f64 {
        self.inflow_boundary.mass_rate().unwrap_or(0.0)
    }

    /// Retrieves the outflow mass rate from the boundary condition.
    ///
    /// # Returns
    /// Outflow mass rate (units: kg/s).
    pub fn get_outflow_mass_rate(&self) -> f64 {
        self.outflow_boundary.mass_rate().unwrap_or(0.0)
    }

    /// Computes the average nearby pressure for an element based on its neighbors.
    ///
    /// # Parameters
    /// - `element`: Reference to the `Element` for which to compute nearby pressure.
    /// - `mesh`: Reference to the `Mesh` containing element connectivity information.
    ///
    /// # Returns
    /// Average pressure of neighboring elements (units: Pascals).
    pub fn get_nearby_pressure(&self, element: &Element, mesh: &Mesh) -> f64 {
        let neighbor_ids = mesh.get_neighbors_of_element(element.id);
        let mut total_pressure = 0.0;
        let mut count = 0;

        for neighbor_id in neighbor_ids {
            if let Some(neighbor) = self.elements.iter().find(|e| e.id == neighbor_id) {
                total_pressure += neighbor.pressure;
                count += 1;
            }
        }

        if count > 0 {
            total_pressure / count as f64
        } else {
            element.pressure // If no neighbors, return the element's own pressure
        }
    }
}

impl Default for FlowField {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            initial_mass: 0.0,
            inflow_boundary: InflowBoundaryCondition::default(),
            outflow_boundary: OutflowBoundaryCondition::default(),
        }
    }
}