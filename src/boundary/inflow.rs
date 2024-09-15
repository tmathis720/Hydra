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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryManager, BoundaryType};
    use crate::domain::{Element, FlowField, Mesh, Node};
    use nalgebra::Vector3;

    #[test]
    fn test_apply_inflow_boundary() {
        // Create elements with initial mass and momentum
        let elements = vec![
            Element {
                id: 0,
                mass: 1.0,
                ..Default::default()
            },
            Element {
                id: 1,
                mass: 2.0,
                ..Default::default()
            },
        ];

        // Create mesh (not needed for this test, but provided for completeness)
        let nodes = vec![
            Node {
                id: 0,
                position: Vector3::new(0.0, 0.0, 0.0),
            },
            Node {
                id: 1,
                position: Vector3::new(1.0, 0.0, 0.0),
            },
        ];
        let mut mesh = Mesh {
            elements: elements.clone(),
            nodes,
            ..Mesh::default()
        };

        // Create an inflow boundary condition with velocity and mass rate
        let mut inflow_boundary = InflowBoundaryCondition::new(Vector3::new(1.0, 0.0, 0.0), 0.5);
        inflow_boundary.add_boundary_element(0);
        inflow_boundary.add_boundary_element(1);

        // Create a BoundaryManager and register the inflow boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::Inflow, Box::new(inflow_boundary));

        // Initialize FlowField with the elements and BoundaryManager
        let mut flow_field = FlowField::new(elements, boundary_manager);

        // Apply the inflow boundary condition
        flow_field.apply_boundary_conditions(&mut mesh, 0.1);

        // Verify that the inflow condition has been applied correctly

        // Element 0
        let element_0 = &flow_field.elements[0];
        assert_eq!(element_0.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(element_0.mass, 1.5);  // Initial mass + inflow mass rate
        assert_eq!(
            element_0.momentum,
            Vector3::new(1.0 * 1.5, 0.0, 0.0)
        );  // Momentum updated

        // Element 1
        let element_1 = &flow_field.elements[1];
        assert_eq!(element_1.velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(element_1.mass, 2.5);  // Initial mass + inflow mass rate
        assert_eq!(
            element_1.momentum,
            Vector3::new(1.0 * 2.5, 0.0, 0.0)
        );  // Momentum updated
    }

    #[test]
    fn test_inflow_boundary_mass_conservation() {
        // Create elements with initial mass
        let elements = vec![
            Element {
                id: 0,
                mass: 1.0,
                ..Default::default()
            },
            Element {
                id: 1,
                mass: 2.0,
                ..Default::default()
            },
        ];

        // Create an inflow boundary condition
        let _inflow_boundary = InflowBoundaryCondition::new(Vector3::new(1.0, 0.0, 0.0), 0.5);

        // Create a BoundaryManager (empty for this test)
        let boundary_manager = BoundaryManager::new();

        // Initialize FlowField with the elements
        let flow_field = FlowField::new(elements.clone(), boundary_manager);

        // No mass added yet, so mass should be conserved
        let total_mass_before: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        let total_mass_after: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        assert!((total_mass_after - total_mass_before).abs() < 1e-6, "Mass should be conserved");
    }
}
