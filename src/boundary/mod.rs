use std::rc::Rc;
use std::cell::RefCell;

pub mod flow;
pub mod free_surface;
pub mod no_slip;
pub mod periodic;


pub use flow::{FlowBoundary, Inflow, Outflow};
pub use free_surface::FreeSurfaceBoundary;
pub use no_slip::NoSlipBoundary;
pub use periodic::PeriodicBoundary;
use crate::solver::FlowField;
use crate::domain::{Mesh, Element};

#[derive(PartialEq)] 
#[derive(Clone)]
pub enum BoundaryType {
    Inflow,
    Outflow,
    NoSlip,
    FreeSurface,
    Periodic,
    Reflective,
}

pub struct Boundary {
    pub boundary_elements: Vec<BoundaryElement>,
    pub condition_type: BoundaryType,
}

impl Boundary {
    pub fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        match self.condition_type {
            BoundaryType::NoSlip => self.apply_no_slip(mesh),
            BoundaryType::FreeSurface => self.apply_free_surface(flow_field),
            BoundaryType::Periodic => self.apply_periodic(&mut mesh.elements),
            BoundaryType::Reflective => self.apply_reflective(mesh),
            BoundaryType::Inflow => self.apply_inflow(flow_field, time_step),
            BoundaryType::Outflow => self.apply_outflow(flow_field, time_step),
        }
    }

    fn apply_inflow(&self, flow_field: &mut FlowField, time_step: f64) {
        // Iterate over inflow boundary elements and apply inflow boundary condition
        for boundary_element in &self.boundary_elements {
            boundary_element.apply_boundary_condition(flow_field, time_step);
        }
    }

    fn apply_outflow(&self, flow_field: &mut FlowField, time_step: f64) {
        // Iterate over outflow boundary elements and apply outflow boundary condition
        for boundary_element in &self.boundary_elements {
            boundary_element.apply_boundary_condition(flow_field, time_step);
        }
    }

    fn apply_no_slip(&self, _mesh: &mut Mesh) {
        // Set velocity to zero at boundary elements
        for boundary_element in &self.boundary_elements {
            let mut element_ref = boundary_element.element.borrow_mut();
            element_ref.velocity = (0.0, 0.0, 0.0);  // Apply no-slip condition
        }
    }

    fn apply_free_surface(&self, flow_field: &mut FlowField) {
        // Adjust pressure or height at free surface elements
        for boundary_element in &self.boundary_elements {
            let mut element_ref = boundary_element.element.borrow_mut();
            element_ref.pressure = flow_field.get_surface_pressure();
        }
    }

    pub fn apply_periodic(&self, elements: &mut [Element]) {
        if self.condition_type == BoundaryType::Periodic {
            let len = elements.len();
            
            // Use split_at_mut() to safely borrow first and last elements
            let (first_slice, last_slice) = elements.split_at_mut(1);
            let first_element = &mut first_slice[0];  // First element
            let last_element = &mut last_slice[len - 2];  // Last element
            
            // Periodic boundary: link the first and last element
            let temp_pressure = first_element.pressure;
            first_element.pressure = last_element.pressure;
            last_element.pressure = temp_pressure;

            let temp_velocity = first_element.velocity;
            first_element.velocity = last_element.velocity;
            last_element.velocity = temp_velocity;

            let temp_momentum = first_element.momentum;
            first_element.momentum = last_element.momentum;
            last_element.momentum = temp_momentum;
        }
    }

    fn apply_reflective(&self, _mesh: &mut Mesh) {
        // Reflect velocity or other properties at boundary elements
        for boundary_element in &self.boundary_elements {
            let mut element_ref = boundary_element.element.borrow_mut();
            element_ref.velocity = (
                -element_ref.velocity.0,
                -element_ref.velocity.1,
                -element_ref.velocity.2,
            );
        }
    }
}

#[derive(Clone)]
pub struct BoundaryElement {
    pub element: Rc<RefCell<Element>>,   // Reference to the element in the mesh
    pub boundary_type: BoundaryType,     // Type of boundary condition applied
}

impl BoundaryElement {
    pub fn apply_boundary_condition(&self, flow_field: &mut FlowField, time_step: f64) {
        match self.boundary_type {
            BoundaryType::Inflow => {
                let mut element_ref = self.element.borrow_mut();
                let inflow_velocity = flow_field.get_inflow_velocity();
                element_ref.velocity = inflow_velocity;
                element_ref.mass += flow_field.inflow_mass_rate();  // Add mass at inflow
            }

            BoundaryType::Outflow => {
                let mut element_ref = self.element.borrow_mut();
                let outflow_velocity = flow_field.get_outflow_velocity();
                element_ref.velocity = outflow_velocity;
                element_ref.mass = (element_ref.mass - flow_field.outflow_mass_rate()).max(0.0);  // Ensure mass doesn't go negative
            }

            BoundaryType::NoSlip => {
                let mut element_ref = self.element.borrow_mut();
                element_ref.velocity = (0.0, 0.0, 0.0);  // Set velocity to zero
            }

            BoundaryType::FreeSurface => {
                let mut element_ref = self.element.borrow_mut();
                let surface_flux = compute_surface_flux(&element_ref, flow_field);  // Ensure function is defined
                element_ref.pressure = flow_field.get_surface_pressure();
                element_ref.velocity.2 += surface_flux / element_ref.area;  // Ensure area exists in `Element`
                element_ref.height += element_ref.velocity.2 * time_step;  // Use the vertical velocity component
            }

            BoundaryType::Periodic => {
                // Implement periodic boundary condition logic
            }

            BoundaryType::Reflective => {
                // Implement reflective boundary condition logic
            }
        }
    }
}

fn compute_surface_flux(element: &Element, flow_field: &FlowField) -> f64 {
    let pressure_diff = element.pressure - flow_field.get_surface_pressure();
    let density = element.compute_density();  // Ensure density is computed properly
    (2.0 * pressure_diff / density).sqrt()  // Example using Bernoulli's equation
}