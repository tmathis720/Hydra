use nalgebra::Vector3;

pub struct Face {
    pub id: u32,                   // Unique identifier for the face
    pub nodes: Vec<usize>,         // Nodes that define the face (typically two for 2D, three for 3D)
    pub velocity: Vector3<f64>,    // 3D velocity (u, v, w) on the face
    pub area: f64,                 // Face area or length in 2D
    pub normal: Vector3<f64>,      // Normal vector to the face in 3D
}

impl Face {
    /// Constructor for a new face
    pub fn new(id: u32, nodes: Vec<usize>, velocity: Vector3<f64>, area: f64, normal: Vector3<f64>) -> Self {
        Self { id, nodes, velocity, area, normal }
    }

    /// Calculate the flux through the face based on the velocity, area, and normal vector
    /// Flux = (velocity dot normal) * area (where normal is assumed to point outward)
    pub fn calculate_flux(&self) -> f64 {
        let normal_velocity_component = self.velocity.dot(&self.normal);  // Velocity in the direction of the normal
        normal_velocity_component * self.area
    }

    /// Set velocity of the face to a new value
    pub fn set_velocity(&mut self, new_velocity: Vector3<f64>) {
        self.velocity = new_velocity;
    }

    /// Get the average position of the face's nodes (mocked for now, depends on node positions)
    /// In practice, this would require access to the node positions from the mesh or domain
    pub fn average_position(&self, node_positions: &[(f64, f64, f64)]) -> (f64, f64, f64) {
        let num_nodes = self.nodes.len();
        if num_nodes >= 2 {
            let sum_pos = self
                .nodes
                .iter()
                .map(|&node_idx| node_positions[node_idx])
                .fold((0.0, 0.0, 0.0), |acc, pos| {
                    (acc.0 + pos.0, acc.1 + pos.1, acc.2 + pos.2)
                });
            (
                sum_pos.0 / num_nodes as f64,
                sum_pos.1 / num_nodes as f64,
                sum_pos.2 / num_nodes as f64,
            )
        } else {
            (0.0, 0.0, 0.0) // Handle invalid cases gracefully
        }
    }

    /// Check if the face is on the boundary based on the node positions and domain dimensions.
    /// This function is generalized for 3D and checks for boundaries on all axes.
    pub fn is_boundary_face(&self, domain_bounds: &[(f64, f64)], node_positions: &[(f64, f64, f64)]) -> bool {
        let (x_min, x_max) = domain_bounds[0];
        let (y_min, y_max) = domain_bounds[1];
        let (z_min, z_max) = domain_bounds[2];
        self.nodes.iter().any(|&node_idx| {
            let (x, y, z) = node_positions[node_idx];
            x == x_min || x == x_max || y == y_min || y == y_max || z == z_min || z == z_max
        })
    }
}

impl Default for Face {
    fn default() -> Self {
        Face {
            id: 0,
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Default to zero velocity
            area: 0.0,
            nodes: vec![0, 0],  // Default to two nodes
            normal: Vector3::new(0.0, 0.0, 0.0),  // Default normal vector
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_flux_calculation() {
        let face = Face::new(
            1,
            vec![0, 1],
            Vector3::new(2.0, 0.0, 0.0),  // 2D velocity in the x-direction
            10.0,
            Vector3::new(1.0, 0.0, 0.0),  // Normal vector in the x-direction
        );
        let flux = face.calculate_flux();

        // Check that flux is calculated correctly (area * velocity dot normal)
        assert_eq!(flux, 20.0);
    }

    #[test]
    fn test_average_position() {
        let node_positions = vec![(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
        let face = Face::new(1, vec![0, 1], Vector3::new(0.0, 0.0, 0.0), 1.0, Vector3::new(1.0, 0.0, 0.0));

        let avg_pos = face.average_position(&node_positions);

        // Check that the average position of the face's nodes is correct
        assert_eq!(avg_pos, (1.0, 0.0, 0.0));
    }

    #[test]
    fn test_boundary_face_detection() {
        let node_positions = vec![(0.0, 0.0, 0.0), (10.0, 5.0, 0.0)];
        let domain_bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 10.0)];
        let face = Face::new(1, vec![0, 1], Vector3::new(0.0, 0.0, 0.0), 1.0, Vector3::new(1.0, 0.0, 0.0));

        // Check if the face is a boundary face based on node positions
        let is_boundary = face.is_boundary_face(&domain_bounds, &node_positions);

        assert!(is_boundary);  // The face contains a node at (0,0), so it is a boundary face
    }
}
