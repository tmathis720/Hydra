// src/domain/face.rs

use nalgebra::Vector3;
use crate::boundary::BoundaryType;

#[derive(Debug, Clone)]
pub struct Face {
    pub id: u32,                   // Unique identifier for the face
    pub nodes: Vec<usize>,         // Indices of nodes that define the face
    pub velocity: Vector3<f64>,    // 3D velocity (u, v, w) on the face
    pub area: f64,                 // Face area or length in 2D
    pub normal: Vector3<f64>,      // Normal vector to the face in 3D
    pub boundary_type: Option<BoundaryType>,  // Type of boundary condition
    pub is_boundary: bool,         // Precomputed flag indicating if the face is on the boundary
}

impl Face {
    /// Constructor for a new face
    pub fn new(
        id: u32,
        nodes: Vec<usize>,
        velocity: Vector3<f64>,
        area: f64,
        normal: Vector3<f64>,
        boundary_type: Option<BoundaryType>,
    ) -> Self {
        Self {
            id,
            nodes,
            velocity,
            area,
            normal,
            boundary_type,
            is_boundary: false,  // Initialize as false; to be set during mesh initialization
        }
    }

    /// Calculate the flux through the face based on the velocity, area, and normal vector
    /// Flux = (velocity dot normal) * area (where normal is assumed to point outward)
    pub fn calculate_flux(&self) -> f64 {
        let normal_velocity_component = self.velocity.dot(&self.normal);
        normal_velocity_component * self.area
    }

    /// Set velocity of the face to a new value
    pub fn set_velocity(&mut self, new_velocity: Vector3<f64>) {
        self.velocity = new_velocity;
    }

    /// Determine if the face is on the boundary based on node positions and domain bounds
    /// This function should be called during mesh initialization
    pub fn determine_if_boundary_face(
        &mut self,
        domain_bounds: &[(f64, f64)],         // [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        node_positions: &[Vector3<f64>],      // Positions of nodes
        tolerance: f64,                       // Tolerance for floating-point comparisons
    ) {
        let (x_min, x_max) = domain_bounds[0];
        let (y_min, y_max) = domain_bounds[1];
        let (z_min, z_max) = domain_bounds[2];

        self.is_boundary = self.nodes.iter().any(|&node_idx| {
            let position = node_positions[node_idx];
            (position.x - x_min).abs() < tolerance
                || (position.x - x_max).abs() < tolerance
                || (position.y - y_min).abs() < tolerance
                || (position.y - y_max).abs() < tolerance
                || (position.z - z_min).abs() < tolerance
                || (position.z - z_max).abs() < tolerance
        });
    }

    /// Set the boundary type of the face
    pub fn set_boundary_type(&mut self, boundary_type: Option<BoundaryType>) {
        self.boundary_type = boundary_type;
    }

    /// Check if the face is on the boundary
    pub fn is_boundary_face(&self) -> bool {
        self.is_boundary
    }

    /// Set the face as a boundary face
    pub fn set_boundary(&mut self, is_boundary: bool) {
        self.is_boundary = is_boundary;
    }
}

impl Default for Face {
    fn default() -> Self {
        Face {
            id: 0,
            nodes: vec![],
            velocity: Vector3::zeros(),
            area: 0.0,
            normal: Vector3::zeros(),
            boundary_type: None,
            is_boundary: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_calculate_flux() {
        let face = Face::new(
            1,
            vec![0, 1],
            Vector3::new(2.0, 0.0, 0.0),
            10.0,
            Vector3::new(1.0, 0.0, 0.0),
            None,
        );
        let flux = face.calculate_flux();
        assert_eq!(flux, 20.0);
    }

    #[test]
    fn test_determine_if_boundary_face() {
        let node_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 5.0, 0.0),
        ];
        let domain_bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 0.0)];
        let tolerance = 1e-6;

        let mut face = Face::new(
            1,
            vec![0, 1],
            Vector3::zeros(),
            1.0,
            Vector3::zeros(),
            None,
        );

        face.determine_if_boundary_face(&domain_bounds, &node_positions, tolerance);

        assert!(face.is_boundary_face());
    }

    #[test]
    fn test_non_boundary_face() {
        let node_positions = vec![
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(9.0, 4.0, 0.0),
        ];
        let domain_bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 0.0)];
        let tolerance = 1e-6;

        let mut face = Face::new(
            2,
            vec![0, 1],
            Vector3::zeros(),
            1.0,
            Vector3::zeros(),
            None,
        );

        face.determine_if_boundary_face(&domain_bounds, &node_positions, tolerance);

        assert!(!face.is_boundary_face());
    }
}
