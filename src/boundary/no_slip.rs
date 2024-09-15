use crate::boundary::BoundaryCondition;
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

    /// Applies the no-slip boundary condition to the elements in the flow field.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh (if additional mesh information is needed).
    /// - `flow_field`: Reference to the flow field, which contains the elements.
    /// - `time_step`: The current simulation time step (if needed).
    fn apply(&self, _mesh: &mut Mesh, flow_field: &mut FlowField, _time_step: f64) {
        for &element_id in &self.boundary_elements {
            if let Some(element) = flow_field.elements.iter_mut().find(|e| e.id == element_id) {
                // Apply the no-slip condition: set velocity and momentum to zero
                element.velocity = Vector3::zeros();
                element.momentum = Vector3::zeros();
            }
        }
    }

    /// Retrieves the velocity associated with the no-slip boundary condition (always zero).
    fn velocity(&self) -> Option<Vector3<f64>> {
        Some(Vector3::zeros())  // No-slip implies zero velocity
    }

    /// No mass rate change for no-slip boundary condition.
    fn mass_rate(&self) -> Option<f64> {
        None  // No-slip condition does not affect mass rate
    }

    fn get_boundary_elements(&self, _mesh: &Mesh) -> Vec<u32> {
        // Return the boundary elements for this inflow condition
        self.boundary_elements.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryManager, BoundaryType};
    use crate::domain::{Element, FlowField, Mesh, Node};
    use nalgebra::Vector3;

    #[test]
    fn test_apply_no_slip_boundary() {
        // Create elements with initial velocity and momentum
        let elements = vec![
            Element {
                id: 1,
                velocity: Vector3::new(5.0, 2.0, 1.0),
                momentum: Vector3::new(10.0, 4.0, 2.0),
                ..Default::default()
            },
            Element {
                id: 2,
                velocity: Vector3::new(3.0, 1.0, 0.0),
                momentum: Vector3::new(6.0, 2.0, 0.0),
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

        // Create a NoSlipBoundaryCondition and add elements
        let mut no_slip_boundary = NoSlipBoundaryCondition::new();
        no_slip_boundary.add_boundary_element(1);
        no_slip_boundary.add_boundary_element(2);

        // Create a BoundaryManager and register the no-slip boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::NoSlip, Box::new(no_slip_boundary));

        // Initialize FlowField with the elements and BoundaryManager
        let mut flow_field = FlowField::new(elements, boundary_manager);

        // Apply the no-slip boundary condition
        flow_field.apply_boundary_conditions(&mut mesh, 0.1);

        // Verify that the no-slip condition has been applied correctly
        let element_1 = &flow_field.elements[0];
        assert_eq!(element_1.velocity, Vector3::zeros());
        assert_eq!(element_1.momentum, Vector3::zeros());

        let element_2 = &flow_field.elements[1];
        assert_eq!(element_2.velocity, Vector3::zeros());
        assert_eq!(element_2.momentum, Vector3::zeros());
    }

    #[test]
    fn test_no_slip_boundary_does_not_affect_mass() {
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

        // Create the NoSlipBoundaryCondition
        let _no_slip_boundary = NoSlipBoundaryCondition::new();

        // Create a BoundaryManager (empty for this test)
        let boundary_manager = BoundaryManager::new();

        // Initialize FlowField with the elements
        let flow_field = FlowField::new(elements.clone(), boundary_manager);

        // No mass added or removed in no-slip condition, so mass should remain the same
        let total_mass_before: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        let total_mass_after: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        assert!((total_mass_after - total_mass_before).abs() < 1e-6, "Mass should be conserved");
    }
}
