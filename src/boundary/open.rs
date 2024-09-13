use crate::domain::Element;
use crate::domain::FlowField;
use nalgebra::Vector3;

pub struct OpenBoundary {
    pub elements: Vec<Element>,  // Elements at the open boundary
    pub boundary_map: Vec<(usize, usize)>,  // Custom boundary mapping for inflow/outflow behavior
}

impl OpenBoundary {
    /// Constructor for creating an OpenBoundary with an optional custom mapping.
    pub fn new(elements: Vec<Element>, boundary_map: Option<Vec<(usize, usize)>>) -> Self {
        let boundary_map = boundary_map.unwrap_or_else(|| vec![]);
        Self {
            elements,
            boundary_map,
        }
    }

    /// Apply open boundary conditions to the elements. This includes inflow or outflow conditions.
    /// Inflow adds mass, momentum, and pressure, while outflow removes them.
    pub fn apply_boundary(&self, elements: &mut Vec<Element>, flow_field: &mut FlowField, time_step: f64) {
        for &(source_index, target_index) in &self.boundary_map {
            if let (Some(source_element), Some(target_element)) = (
                elements.get(source_index).cloned(),
                elements.get_mut(target_index),
            ) {
                if is_inflow(&source_element) {
                    self.apply_inflow(&source_element, target_element, flow_field, time_step);
                } else if is_outflow(&source_element) {
                    self.apply_outflow(&source_element, target_element, flow_field, time_step);
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
    fn apply_inflow(&self, _source_element: &Element, target_element: &mut Element, flow_field: &mut FlowField, time_step: f64) {
        let inflow_velocity = flow_field.get_inflow_velocity();
    
        // Add mass to the target element
        let mass_addition = flow_field.inflow_mass_rate() * time_step;
        target_element.mass += mass_addition;
        
        // Add momentum based on inflow velocity (scalar multiplication with a Vector3)
        target_element.momentum += inflow_velocity * mass_addition;
    
        // Adjust pressure to account for inflow
        target_element.pressure = flow_field.get_surface_pressure();
    }
    
    fn apply_outflow(&self, _source_element: &Element, target_element: &mut Element, flow_field: &mut FlowField, time_step: f64) {
        // Remove mass from the target element, ensuring non-negative mass
        let mass_removal = flow_field.outflow_mass_rate() * time_step;
        target_element.mass = (target_element.mass - mass_removal).max(0.0);
    
        // Reduce momentum based on outflow velocity (element-wise subtraction)
        let outflow_velocity = flow_field.get_outflow_velocity();
        
        // Element-wise max to ensure no negative momentum values
        target_element.momentum = Vector3::new(
            (target_element.momentum.x - outflow_velocity.x * mass_removal).max(0.0),
            (target_element.momentum.y - outflow_velocity.y * mass_removal).max(0.0),
            (target_element.momentum.z - outflow_velocity.z * mass_removal).max(0.0),
        );
    
        // Adjust pressure to account for outflow
        target_element.pressure = flow_field.get_surface_pressure();
    }

    /// Add custom inflow/outflow mapping (source_index -> target_index).
    pub fn add_boundary_mapping(&mut self, source_index: usize, target_index: usize) {
        if source_index < self.elements.len() && target_index < self.elements.len() {
            self.boundary_map.push((source_index, target_index));
        } else {
            eprintln!(
                "Cannot add mapping: Invalid indices {} -> {}",
                source_index, target_index
            );
        }
    }
}

/// Determines whether an element is at an inflow boundary.
/// For now, this is a placeholder logic. You may add domain-specific checks.
fn is_inflow(element: &Element) -> bool {
    element.velocity.x > 0.0  // Example: inflow if x-velocity is positive
}

/// Determines whether an element is at an outflow boundary.
/// For now, this is a placeholder logic. You may add domain-specific checks.
fn is_outflow(element: &Element) -> bool {
    element.velocity.x < 0.0  // Example: outflow if x-velocity is negative
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_open_boundary_inflow() {
        let mut elements = vec![
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

        let mut flow_field = FlowField::default();
        let open_boundary = OpenBoundary::new(elements.clone(), None);
        
        open_boundary.apply_boundary(&mut elements, &mut flow_field, 1.0);

        // Check that inflow has added mass and momentum
        assert!(elements[1].mass > 3.0);
        assert!(elements[1].momentum.x > 5.0);
        assert!(elements[1].pressure > 50.0);  // Pressure should have been adjusted
    }

    #[test]
    fn test_open_boundary_outflow() {
        let mut elements = vec![
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

        let mut flow_field = FlowField::default();
        let open_boundary = OpenBoundary::new(elements.clone(), None);

        open_boundary.apply_boundary(&mut elements, &mut flow_field, 1.0);

        // Check that outflow has reduced mass and momentum
        assert!(elements[1].mass < 3.0);
        assert!(elements[1].momentum.x < 5.0);
    }
}
