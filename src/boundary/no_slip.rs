// src/boundary/no_slip.rs

use crate::boundary::{BoundaryCondition, BoundaryType};
use crate::domain::{FlowField, Mesh};
use nalgebra::Vector3;

pub struct NoSlipBoundaryCondition {
    pub boundary_elements: Vec<u32>, // IDs of boundary elements
}

impl NoSlipBoundaryCondition {
    pub fn new() -> Self {
        Self {
            boundary_elements: Vec::new(),
        }
    }

    /// Adds a boundary element to the no-slip condition.
    pub fn add_boundary_element(&mut self, element_id: u32) {
        self.boundary_elements.push(element_id);
    }
}

impl BoundaryCondition for NoSlipBoundaryCondition {
    fn update(&mut self, _time: f64) {
        // No-slip condition typically does not update with time
    }

    fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, _time_step: f64) {
        for &element_id in &self.boundary_elements {
            if let Some(element) = flow_field.elements.iter_mut().find(|e| e.id == element_id) {
                // Apply the no-slip condition: set velocity and momentum to zero
                element.velocity = Vector3::zeros();
                element.momentum = Vector3::zeros();
            }
        }
    }

    fn velocity(&self) -> Option<Vector3<f64>> {
        Some(Vector3::zeros())  // No-slip implies zero velocity
    }

    fn mass_rate(&self) -> Option<f64> {
        None  // No-slip condition does not affect mass rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Element;
    use nalgebra::Vector3;

    #[test]
    fn test_no_slip_boundary() {
        let mut element = Element {
            id: 1,
            velocity: Vector3::new(5.0, 2.0, 1.0),
            momentum: Vector3::new(10.0, 4.0, 2.0),
            ..Default::default()
        };

        let mut flow_field = FlowField::new(vec![element.clone()]);
        let mut no_slip_boundary = NoSlipBoundaryCondition::new();
        no_slip_boundary.add_boundary_element(1);

        no_slip_boundary.apply(&mut Mesh::default(), &mut flow_field, 0.0);

        let updated_element = &flow_field.elements[0];
        assert_eq!(updated_element.velocity, Vector3::zeros());
        assert_eq!(updated_element.momentum, Vector3::zeros());
    }
}
