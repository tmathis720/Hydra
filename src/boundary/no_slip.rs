use crate::domain::{Element, Face};

pub struct NoSlipBoundary;

impl NoSlipBoundary {
    /// Apply no-slip boundary condition (velocity should be zero, so momentum is zero)
    pub fn apply_boundary(&self, element: &mut Element, _dt: f64) {
        // Zero out both momentum and velocity to ensure that the element obeys the no-slip condition
        element.momentum = 0.0;
        element.velocity = (0.0, 0.0, 0.0); // Ensure velocity is also explicitly set to zero
    }

    /// Apply no-slip boundary condition to faces by setting velocity at each boundary node to zero
    /// Handles nodes individually to ensure that no-slip is applied correctly across the entire face
    pub fn apply_face_boundary(&self, face: &mut Face, elements: &mut [Element], _dt: f64) {
        for node_index in &[face.nodes.0, face.nodes.1] {
            // Access the corresponding element for each node on the face
            let element = &mut elements[*node_index as usize];
            
            // Set the velocity and momentum to zero for the no-slip condition
            element.velocity = (0.0, 0.0, 0.0);
            element.momentum = 0.0;
        }
    }

    /// Determine if a face corresponds to a no-slip boundary based on geometry or other criteria
    /// In this example, we check if any node is on the edge of the domain (boundary condition)
    pub fn is_no_slip_face(&self, face: &Face, domain_width: f64, domain_height: f64) -> bool {
        let node_1 = face.nodes.0;
        let node_2 = face.nodes.1;

        // Logic for determining if either node is on the domain boundary (x = 0, y = 0, etc.)
        self.is_boundary_node(node_1, domain_width, domain_height) || self.is_boundary_node(node_2, domain_width, domain_height)
    }

    /// Helper function to determine if a node is at the domain boundary (no-slip condition)
    fn is_boundary_node(&self, node_index: usize, domain_width: f64, domain_height: f64) -> bool {
        // Assuming node positions are available in a global mesh or domain data structure
        let node_position = get_node_position(node_index); // Retrieve node position (mock function)
        
        // Check if the node is at the boundary (x = 0, y = 0, or at the edges of the domain)
        node_position.0 == 0.0 || node_position.0 == domain_width || node_position.1 == 0.0 || node_position.1 == domain_height
    }
}

/// Mock function to retrieve a node's position in the domain (to be replaced by actual logic)
/// In real implementation, this would query the mesh or domain structure for the node's position
fn get_node_position(node_index: usize) -> (f64, f64) {
    // Mock positions for the purpose of this example
    // Replace with actual data lookup from the mesh or domain structure
    match node_index {
        0 => (0.0, 0.0),
        1 => (0.0, 1.0),
        _ => (1.0, 1.0),
    }
}

#[test]
fn test_no_slip_boundary_application() {
    let mut element = Element {
        momentum: 10.0,
        velocity: (5.0, 5.0, 0.0),
        ..Default::default() // Assuming a default initializer for the rest
    };

    let no_slip_boundary = NoSlipBoundary {};
    no_slip_boundary.apply_boundary(&mut element, 1.0);

    // Assert that momentum and velocity are both zero after applying no-slip
    assert_eq!(element.momentum, 0.0);
    assert_eq!(element.velocity, (0.0, 0.0, 0.0));
}
