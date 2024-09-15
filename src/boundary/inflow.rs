// src/boundary/inflow.rs

use crate::boundary::BoundaryCondition;
use crate::domain::{Mesh, FlowField};
use nalgebra::Vector3;

/// Represents an inflow boundary condition that adds velocity and mass to elements.
#[derive(Default)]
pub struct InflowBoundaryCondition {
    pub velocity: Vector3<f64>,
    pub mass_rate: f64,
    pub boundary_elements: Vec<u32>, // IDs of boundary elements
}

impl InflowBoundaryCondition {
    /// Creates a new inflow boundary condition with the specified velocity and mass rate.
    pub fn new(velocity: Vector3<f64>, mass_rate: f64) -> Self {
        Self {
            velocity,
            mass_rate,
            boundary_elements: Vec::new(),
        }
    }

    /// Adds a boundary element to the inflow condition.
    pub fn add_boundary_element(&mut self, element_id: u32) {
        self.boundary_elements.push(element_id);
    }
}

impl BoundaryCondition for InflowBoundaryCondition {
    /// Updates the inflow boundary condition based on time or other factors.
    fn update(&mut self, _time: f64) {
        // Update logic based on time or other factors
    }

    /// Applies the inflow boundary condition to the elements in the flow field.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh (if additional mesh information is needed).
    /// - `flow_field`: Reference to the flow field, which contains the elements.
    /// - `time_step`: The current simulation time step (if needed).
    fn apply(&self, _mesh: &mut Mesh, flow_field: &mut FlowField, _time_step: f64) {
        for &element_id in &self.boundary_elements {
            if let Some(element) = flow_field.elements.iter_mut().find(|e| e.id == element_id) {
                // Apply inflow conditions to the element
                element.velocity = self.velocity;
                element.mass += self.mass_rate;  // Add mass at inflow
                element.momentum += self.velocity * element.mass;  // Update momentum
            }
        }
    }

    /// Retrieves the velocity associated with the inflow boundary condition.
    fn velocity(&self) -> Option<Vector3<f64>> {
        Some(self.velocity)
    }

    /// Retrieves the mass rate associated with the inflow boundary condition.
    fn mass_rate(&self) -> Option<f64> {
        Some(self.mass_rate)
    }
}
