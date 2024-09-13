use crate::domain::{Element, Face, Mesh};
use nalgebra::Vector3;

pub struct NoSlipBoundary;

impl NoSlipBoundary {
    /// Apply no-slip boundary condition: set velocity and momentum to zero
    pub fn apply_boundary(&self, element: &mut Element, _dt: f64) {
        // Zero out both momentum and velocity to enforce the no-slip condition
        element.momentum = Vector3::zeros();
        element.velocity = Vector3::zeros();  // Ensure velocity is explicitly set to zero
    }

    /// Apply no-slip boundary condition to all nodes on a face
    /// This method sets velocity to zero for the elements associated with the face's nodes
    pub fn apply_face_boundary(&self, face: &mut Face, elements: &mut [Element], mesh: &Mesh, _dt: f64) {
        for node_index in &face.nodes {
            // Access the corresponding element for each node on the face
            if let Some(element) = mesh.get_element_connected_to_node(*node_index) {
                let element = &mut elements[element];
                // Set the velocity and momentum to zero for the no-slip condition
                element.velocity = Vector3::zeros();
                element.momentum = Vector3::zeros();
            }
        }
    }

    /// Determine if a face corresponds to a no-slip boundary based on the domain geometry
    /// This logic checks if any node of the face is on the domain boundary (e.g., x=0, y=0)
    pub fn is_no_slip_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes[0]];
        let node_2 = &mesh.nodes[face.nodes[1]];

        // Check if either node is at the boundary of the domain
        self.is_boundary_node(node_1.position, mesh) || self.is_boundary_node(node_2.position, mesh)
    }

    /// Helper function to determine if a node is at the boundary of the domain
    fn is_boundary_node(&self, node_position: Vector3<f64>, mesh: &Mesh) -> bool {
        // Check if the node is at the boundary (x = 0, x = domain width, y = 0, or y = domain height)
        let domain_width = mesh.domain_width();
        let domain_height = mesh.domain_height();
        node_position.x == 0.0 || node_position.x == domain_width || node_position.y == 0.0 || node_position.y == domain_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Node, Mesh, Face};
    use nalgebra::Vector3;

    #[test]
    fn test_no_slip_boundary_application() {
        // Test the no-slip boundary condition on an element
        let mut element = Element {
            momentum: Vector3::new(10.0, 10.0, 5.0),
            velocity: Vector3::new(5.0, 5.0, 2.0),
            ..Default::default()  // Assuming a default initializer for other fields
        };

        let no_slip_boundary = NoSlipBoundary {};
        no_slip_boundary.apply_boundary(&mut element, 1.0);

        // Assert that momentum and velocity are both zero after applying the no-slip condition
        assert_eq!(element.momentum, Vector3::zeros());
        assert_eq!(element.velocity, Vector3::zeros());
    }

    #[test]
    fn test_no_slip_face_boundary_application() {
        // Test the no-slip boundary condition applied to a face
        let mut elements = vec![
            Element {
                momentum: Vector3::new(10.0, 10.0, 5.0),
                velocity: Vector3::new(5.0, 5.0, 2.0),
                ..Default::default()
            },
            Element {
                momentum: Vector3::new(8.0, 7.0, 3.0),
                velocity: Vector3::new(4.0, 4.0, 1.0),
                ..Default::default()
            },
        ];

        let mut face = Face {
            nodes: vec![0, 1],
            ..Default::default()  // Assuming default implementation for other fields
        };

        let mesh = Mesh {
            elements: elements.clone(),
            ..Default::default()
        };

        let no_slip_boundary = NoSlipBoundary {};
        no_slip_boundary.apply_face_boundary(&mut face, &mut elements, &mesh, 1.0);

        // Assert that the momentum and velocity of the elements associated with the face are zero
        for element in elements.iter() {
            assert_eq!(element.momentum, Vector3::zeros());
            assert_eq!(element.velocity, Vector3::zeros());
        }
    }

    #[test]
    fn test_no_slip_face_detection() {
        let face = Face {
            nodes: vec![0, 1],
            ..Default::default()  // Assuming default implementation for other fields
        };

        let mesh = Mesh {
            nodes: vec![
                Node {
                    id: 0,
                    position: Vector3::new(0.0, 0.0, 0.0),
                },
                Node {
                    id: 1,
                    position: Vector3::new(0.0, 1.0, 0.0),
                },
            ],
            ..Default::default()
        };

        let no_slip_boundary = NoSlipBoundary {};
        let is_no_slip = no_slip_boundary.is_no_slip_face(&face, &mesh);

        // Assert that the face is correctly identified as a no-slip boundary
        assert!(is_no_slip);
    }
}
