use crate::boundary::BoundaryCondition;
use crate::domain::{FlowField, Mesh};
use nalgebra::Vector3;

pub struct NoSlipBoundaryCondition {
    pub boundary_faces: Vec<u32>, // IDs of boundary faces
}

impl NoSlipBoundaryCondition {
    pub fn new() -> Self {
        Self {
            boundary_faces: Vec::new(),
        }
    }

    /// Adds a boundary face to the no-slip condition.
    pub fn add_boundary_face(&mut self, face_id: u32) {
        self.boundary_faces.push(face_id);
    }
}

impl BoundaryCondition for NoSlipBoundaryCondition {
    fn update(&mut self, _time: f64) {
        // No-slip condition typically does not update with time
    }

    /// Applies the no-slip boundary condition to the faces in the flow field.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh to identify the boundary faces.
    /// - `flow_field`: Reference to the flow field, which contains the elements.
    /// - `time_step`: The current simulation time step (if needed).
    fn apply(&self, mesh: &mut Mesh, _flow_field: &mut FlowField, _time_step: f64) {
        for &face_id in &self.boundary_faces {
            if let Some(face) = mesh.faces.iter_mut().find(|f| f.id == face_id) {
                // Apply the no-slip condition: set velocity to zero for the face's velocity
                face.velocity = Vector3::zeros();
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
        // Return the boundary faces for this no-slip condition
        self.boundary_faces.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryManager, BoundaryType};
    use crate::domain::{Mesh, Face, Node, FlowField};
    use nalgebra::Vector3;

    /// Helper function to create a mock mesh with faces
    fn create_mock_mesh() -> Mesh {
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(5.0, 2.0, 1.0), 1.0, Vector3::new(1.0, 0.0, 0.0), None),
            Face::new(2, vec![2, 3], Vector3::new(3.0, 1.0, 0.0), 1.0, Vector3::new(1.0, 0.0, 0.0), None),
        ];

        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 1.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 1.0, 0.0)),
            Node::new(2, Vector3::new(0.0, 2.0, 0.0)),
            Node::new(3, Vector3::new(1.0, 2.0, 0.0)),
        ];

        Mesh {
            faces,
            nodes,
            elements: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        }
    }

    #[test]
    fn test_apply_no_slip_boundary_on_faces() {
        // Create mock mesh with faces
        let mut mesh = create_mock_mesh();

        // Create a NoSlipBoundaryCondition and add faces
        let mut no_slip_boundary = NoSlipBoundaryCondition::new();
        no_slip_boundary.add_boundary_face(1);
        no_slip_boundary.add_boundary_face(2);

        // Create a BoundaryManager and register the no-slip boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::NoSlip, Box::new(no_slip_boundary));

        // Create a FlowField (not really used in this test)
        let mut flow_field = FlowField::new(vec![], boundary_manager);

        // Apply the no-slip boundary condition
        flow_field.apply_boundary_conditions(&mut mesh, 0.1);

        // Verify that the no-slip condition has been applied correctly
        let face_1 = &mesh.faces[0];
        assert_eq!(face_1.velocity, Vector3::zeros(), "Face 1 velocity should be zero due to no-slip condition.");

        let face_2 = &mesh.faces[1];
        assert_eq!(face_2.velocity, Vector3::zeros(), "Face 2 velocity should be zero due to no-slip condition.");
    }

    #[test]
    fn test_no_slip_boundary_does_not_affect_elements() {
        // Create mock mesh with faces, but no effect on elements
        let _mesh = create_mock_mesh();

        // Create the NoSlipBoundaryCondition
        let mut no_slip_boundary = NoSlipBoundaryCondition::new();
        no_slip_boundary.add_boundary_face(1);
        no_slip_boundary.add_boundary_face(2);

        // Create a BoundaryManager and register the no-slip boundary condition
        let boundary_manager = BoundaryManager::new();

        // Initialize FlowField with the boundary manager (empty elements)
        let flow_field = FlowField::new(vec![], boundary_manager);

        // No elements should be affected, only faces.
        assert!(flow_field.elements.is_empty(), "No elements should be affected by no-slip boundary condition.");
    }
}