use crate::boundary::BoundaryCondition;
use crate::domain::{Element, FlowField, Mesh};
use nalgebra::Vector3;

/// Represents an open boundary condition that can handle both inflow and outflow behavior.
pub struct OpenBoundaryCondition {
    pub boundary_map: Vec<(usize, usize)>,  // Custom boundary mapping for inflow/outflow behavior
}

impl OpenBoundaryCondition {
    /// Constructor for creating an OpenBoundary with an optional custom mapping.
    pub fn new(boundary_map: Option<Vec<(usize, usize)>>) -> Self {
        let boundary_map = boundary_map.unwrap_or_else(|| vec![]);
        Self { boundary_map }
    }

    /// Adds a custom inflow/outflow mapping (source_index -> target_index).
    pub fn add_boundary_mapping(&mut self, source_index: usize, target_index: usize) {
        self.boundary_map.push((source_index, target_index));
    }

    /// This method should now query the `BoundaryManager` for inflow and outflow elements.
    fn apply_boundary_conditions(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        // Get inflow elements and outflow elements first
        let inflow_elements = flow_field.boundary_manager.get_inflow_elements(mesh);
        let outflow_elements = flow_field.boundary_manager.get_outflow_elements(mesh);

        // Apply inflow logic
        for inflow_element_id in inflow_elements {
            // Extract inflow velocity and mass rate before mutably borrowing flow_field
            let inflow_velocity = flow_field.get_inflow_velocity(mesh).unwrap_or(Vector3::zeros());
            let mass_rate = flow_field.get_inflow_mass_rate(mesh) * time_step;
            
            // Extract pressure separately before the mutable borrow occurs
            let pressure = flow_field.get_pressure(inflow_element_id).unwrap_or(0.0);

            if let Some(target_element) = flow_field.elements.get_mut(inflow_element_id as usize) {
                // Apply inflow logic without accessing flow_field again
                self.apply_inflow(target_element, inflow_velocity, mass_rate, pressure);
            }
        }

        // Apply outflow logic
        for outflow_element_id in outflow_elements {
            // Extract outflow velocity and mass rate before mutably borrowing flow_field
            let outflow_velocity = flow_field.get_outflow_velocity(mesh).unwrap_or(Vector3::zeros());
            let mass_rate = flow_field.get_outflow_mass_rate(mesh) * time_step;

            // Extract pressure separately before the mutable borrow occurs
            let pressure = flow_field.get_pressure(outflow_element_id).unwrap_or(0.0);

            if let Some(target_element) = flow_field.elements.get_mut(outflow_element_id as usize) {
                // Apply outflow logic without accessing flow_field again
                self.apply_outflow(target_element, outflow_velocity, mass_rate, pressure);
            }
        }
    }

    // Adjust inflow logic to avoid mutable borrow issues
    fn apply_inflow(
        &self,
        target_element: &mut Element,
        inflow_velocity: Vector3<f64>,
        mass_addition: f64,
        pressure: f64
    ) {
        target_element.mass += mass_addition;
        target_element.momentum += inflow_velocity * mass_addition;
        target_element.pressure = pressure;
    }

    // Adjust outflow logic to avoid mutable borrow issues
    fn apply_outflow(
        &self,
        target_element: &mut Element,
        outflow_velocity: Vector3<f64>,
        mass_removal: f64,
        pressure: f64
    ) {
        target_element.mass = (target_element.mass - mass_removal).max(0.0);
        target_element.momentum -= outflow_velocity * mass_removal;
        target_element.momentum = target_element.momentum.map(|val| val.max(0.0));
        target_element.pressure = pressure;
    }

    /// Identify inflow boundary elements in the mesh
    fn get_inflow_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        mesh.elements.iter()
            .filter(|e| is_inflow(e))
            .map(|e| e.id)
            .collect()
    }

    /// Identify outflow boundary elements in the mesh
    fn get_outflow_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        mesh.elements.iter()
            .filter(|e| is_outflow(e))
            .map(|e| e.id)
            .collect()
    }
}

/// Determines whether an element is at an inflow boundary.
fn is_inflow(element: &Element) -> bool {
    element.velocity.x > 0.0  // Example: inflow if x-velocity is positive
}

/// Determines whether an element is at an outflow boundary.
fn is_outflow(element: &Element) -> bool {
    element.velocity.x < 0.0  // Example: outflow if x-velocity is negative
}

impl BoundaryCondition for OpenBoundaryCondition {
    fn update(&mut self, _time: f64) {
        // No specific update logic needed for open boundary conditions
    }

    fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        self.apply_boundary_conditions(mesh, flow_field, time_step);
    }

    fn velocity(&self) -> Option<Vector3<f64>> {
        None  // Open boundary does not specify a single velocity
    }

    fn mass_rate(&self) -> Option<f64> {
        None  // Open boundary does not specify a single mass rate
    }

    /// Retrieve the inflow and outflow boundary elements from the flow field
    fn get_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        let inflow_elements = self.get_inflow_boundary_elements(mesh);
        let outflow_elements = self.get_outflow_boundary_elements(mesh);

        // Combine both inflow and outflow elements
        [inflow_elements, outflow_elements].concat()
    }
}
