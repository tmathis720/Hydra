use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::Vector3; // Use nalgebra for vector operations

pub mod flow;
pub mod free_surface;
pub mod no_slip;
pub mod periodic;
pub mod open;

pub use flow::{FlowBoundary, Inflow, Outflow};
pub use free_surface::FreeSurfaceBoundary;
pub use no_slip::NoSlipBoundary;
pub use periodic::PeriodicBoundary;
pub use open::OpenBoundary;

use crate::domain::{Mesh, Element};
use crate::domain::FlowField;  // Updated import for FlowField

#[derive(PartialEq, Clone)]
pub enum BoundaryType {
    Inflow,
    Outflow,
    NoSlip,
    FreeSurface,
    Periodic,
    Reflective,
    Open,  // Open boundary type added
}

/// Struct representing boundary conditions in the simulation
pub struct Boundary {
    pub boundary_elements: Vec<BoundaryElement>,
    pub condition_type: BoundaryType,
}

impl Boundary {
    /// Apply the appropriate boundary condition based on the boundary type.
    pub fn apply(&self, _mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        for boundary_element in &self.boundary_elements {
            boundary_element.apply_boundary_condition(flow_field, time_step);
        }
    }
}

#[derive(Clone)]
pub struct BoundaryElement {
    pub element: Rc<RefCell<Element>>,   // Reference to the element in the mesh
    pub paired_element: Option<Rc<RefCell<Element>>>, // For periodic boundaries
    pub boundary_type: BoundaryType,     // Type of boundary condition applied
}

impl BoundaryElement {
    /// Apply boundary condition based on boundary type.
    pub fn apply_boundary_condition(&self, flow_field: &mut FlowField, time_step: f64) {
        match self.boundary_type {
            BoundaryType::Inflow => self.apply_inflow(flow_field),
            BoundaryType::Outflow => self.apply_outflow(flow_field),
            BoundaryType::NoSlip => self.apply_no_slip(),
            BoundaryType::FreeSurface => self.apply_free_surface(flow_field, time_step),
            BoundaryType::Periodic => self.apply_periodic(),
            BoundaryType::Reflective => self.apply_reflective(),
            BoundaryType::Open => self.apply_open_boundary(flow_field),  // Handle open boundary condition
        }
    }

    fn apply_inflow(&self, flow_field: &mut FlowField) {
        let mut element_ref = self.element.borrow_mut();
    
        // First, get the inflow velocity from the flow field (no mutation yet)
        let inflow_velocity = flow_field.get_inflow_velocity();  // Updated for Vector3
        
        // Store the mass in a temporary variable before modifying it
        let mass = element_ref.mass;
    
        // Now, apply the inflow conditions
        element_ref.velocity = inflow_velocity;
        element_ref.mass += flow_field.inflow_mass_rate();  // Add mass at inflow
        
        // Update momentum using the inflow velocity and the original mass
        element_ref.momentum += inflow_velocity * mass;  // Avoid borrowing element_ref again
    }

    fn apply_outflow(&self, flow_field: &mut FlowField) {
        let mut element_ref = self.element.borrow_mut();
    
        // Get the outflow velocity from the flow field (no mutation yet)
        let outflow_velocity = flow_field.get_outflow_velocity();  // Updated for Vector3
    
        // Store the mass in a temporary variable before modifying it
        let mass = element_ref.mass;
    
        // Now, apply the outflow conditions
        element_ref.velocity = outflow_velocity;
        element_ref.mass = (element_ref.mass - flow_field.outflow_mass_rate()).max(0.0);  // Ensure mass doesn't go negative
    
        // Update momentum using the outflow velocity and the original mass
        element_ref.momentum -= outflow_velocity * mass;  // Avoid borrowing element_ref again
    }

    fn apply_no_slip(&self) {
        let mut element_ref = self.element.borrow_mut();
        element_ref.velocity = Vector3::new(0.0, 0.0, 0.0);  // Set velocity to zero using Vector3 for no-slip condition
    }

    fn apply_free_surface(&self, flow_field: &mut FlowField, time_step: f64) {
        let mut element_ref = self.element.borrow_mut();
        let surface_flux = compute_surface_flux(&element_ref, flow_field);
        element_ref.pressure = flow_field.get_surface_pressure();
        element_ref.velocity.z += surface_flux / element_ref.area;  // Update vertical velocity component (z) using Vector3
        element_ref.height += element_ref.velocity.z * time_step;  // Update height based on vertical velocity
    }

    fn apply_periodic(&self) {
        if let Some(ref paired_element) = self.paired_element {
            let mut element_ref = self.element.borrow_mut();
            let mut paired_element_ref = paired_element.borrow_mut();

            // Exchange properties between the element and its paired element for periodic boundary conditions
            std::mem::swap(&mut element_ref.pressure, &mut paired_element_ref.pressure);
            std::mem::swap(&mut element_ref.velocity, &mut paired_element_ref.velocity);
            std::mem::swap(&mut element_ref.momentum, &mut paired_element_ref.momentum);
        }
    }

    fn apply_reflective(&self) {
        let mut element_ref = self.element.borrow_mut();
        element_ref.velocity = -element_ref.velocity;  // Reflect velocity vector in all directions
    }

    fn apply_open_boundary(&self, flow_field: &mut FlowField) {
        let mut element_ref = self.element.borrow_mut();
    
        // Store values in temporary variables before mutating the element
        let outflow_velocity = flow_field.get_outflow_velocity();  // Updated for Vector3
        let outflow_mass_rate = flow_field.outflow_mass_rate();
        let current_mass = element_ref.mass;
    
        // Apply the open boundary conditions
        element_ref.velocity = outflow_velocity;
        element_ref.mass = (element_ref.mass - outflow_mass_rate).max(0.0);  // Ensure mass doesn't go negative
        
        // Update momentum using the velocity and the original mass
        element_ref.momentum -= outflow_velocity * current_mass;  // Avoid borrowing element_ref again
    }
}

/// Compute the surface flux for free surface boundary conditions.
fn compute_surface_flux(element: &Element, flow_field: &FlowField) -> f64 {
    let pressure_diff = element.pressure - flow_field.get_surface_pressure();
    let density = element.compute_density();
    (2.0 * pressure_diff / density).sqrt()  // Compute flux using Bernoulli’s equation
}
