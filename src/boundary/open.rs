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

    /// Apply open boundary conditions to the flow field.
    /// This includes inflow or outflow conditions. Inflow adds mass, momentum, and pressure,
    /// while outflow removes them.
    fn apply_boundary_conditions(
        &self,
        _mesh: &mut Mesh,
        flow_field: &mut FlowField,
        time_step: f64,
    ) {
        for &(source_index, target_index) in &self.boundary_map {
            if let (Some(source_element), Some(target_element)) = (
                flow_field.elements.get(source_index),
                flow_field.elements.get_mut(target_index),
            ) {
                if is_inflow(&source_element) {
                    self.apply_inflow(source_element, target_element, flow_field, time_step);
                } else if is_outflow(&source_element) {
                    self.apply_outflow(source_element, target_element, flow_field, time_step);
                }
            } else {
                eprintln!(
                    "Invalid open boundary mapping from index {} to index {}",
                    source_index, target_index
                );
            }
        }
    }

    /// Apply inflow boundary conditions. Inflow adds mass, momentum, and adjusts velocity.
    fn apply_inflow(
        &self,
        _source_element: &Element,
        target_element: &mut Element,
        flow_field: &FlowField,
        time_step: f64,
    ) {
        let inflow_velocity = flow_field.get_inflow_velocity().unwrap_or(Vector3::zeros());

        // Add mass to the target element
        let mass_addition = flow_field.get_inflow_mass_rate() * time_step;
        target_element.mass += mass_addition;

        // Add momentum based on inflow velocity
        target_element.momentum += inflow_velocity * mass_addition;

        // Adjust pressure to account for inflow
        target_element.pressure = flow_field.get_pressure(target_element.id).unwrap_or(0.0);
    }

    /// Apply outflow boundary conditions. Outflow removes mass, momentum, and adjusts velocity.
    fn apply_outflow(
        &self,
        _source_element: &Element,
        target_element: &mut Element,
        flow_field: &FlowField,
        time_step: f64,
    ) {
        // Remove mass from the target element, ensuring non-negative mass
        let mass_removal = flow_field.get_outflow_mass_rate() * time_step;
        target_element.mass = (target_element.mass - mass_removal).max(0.0);

        // Reduce momentum based on outflow velocity
        let outflow_velocity = flow_field.get_outflow_velocity().unwrap_or(Vector3::zeros());
        target_element.momentum -= outflow_velocity * mass_removal;

        // Ensure momentum doesn't go negative
        target_element.momentum = target_element.momentum.max(Vector3::zeros());

        // Adjust pressure to account for outflow
        target_element.pressure = flow_field.get_pressure(target_element.id).unwrap_or(0.0);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryManager;
    use crate::domain::{Element, FlowField, Mesh};
    use nalgebra::Vector3;

    #[test]
    fn test_open_boundary_inflow() {
        let elements = vec![
            Element {
                pressure: 100.0,
                velocity: Vector3::new(1.0, 0.0, 0.0),  // Positive velocity indicates inflow
                momentum: Vector3::new(10.0, 0.0, 0.0),
                mass: 5.0,
                ..Default::default()
            },
            Element {
                pressure: 50.0,
                velocity: Vector3::zeros(),
                momentum: Vector3::new(5.0, 1.0, 0.0),
                mass: 3.0,
                ..Default::default()
            },
        ];

        let boundary_manager = BoundaryManager::new();
        let mut flow_field = FlowField::new(elements.clone(), boundary_manager);
        let mut open_boundary = OpenBoundaryCondition::new(None);
        open_boundary.add_boundary_mapping(0, 1);

        // Apply boundary with inflow
        open_boundary.apply(&mut Mesh::default(), &mut flow_field, 1.0);

        // Check that inflow has added mass and momentum
        let target_element = &flow_field.elements[1];
        assert!(target_element.mass > 3.0, "Mass should increase");
        assert!(target_element.momentum.x > 5.0, "Momentum should increase");
        assert!(target_element.pressure > 50.0, "Pressure should increase");
    }

    #[test]
    fn test_open_boundary_outflow() {
        let elements = vec![
            Element {
                pressure: 100.0,
                velocity: Vector3::new(-1.0, 0.0, 0.0),  // Negative velocity indicates outflow
                momentum: Vector3::new(10.0, 0.0, 0.0),
                mass: 5.0,
                ..Default::default()
            },
            Element {
                pressure: 50.0,
                velocity: Vector3::zeros(),
                momentum: Vector3::new(5.0, 1.0, 0.0),
                mass: 3.0,
                ..Default::default()
            },
        ];

        let boundary_manager = BoundaryManager::new();
        let mut flow_field = FlowField::new(elements.clone(), boundary_manager);
        let mut open_boundary = OpenBoundaryCondition::new(None);
        open_boundary.add_boundary_mapping(0, 1);

        // Apply boundary with outflow
        open_boundary.apply(&mut Mesh::default(), &mut flow_field, 1.0);

        // Check that outflow has reduced mass and momentum
        let target_element = &flow_field.elements[1];
        assert!(target_element.mass < 3.0, "Mass should decrease");
        assert!(target_element.momentum.x < 5.0, "Momentum should decrease");
    }
}
