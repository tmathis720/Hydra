use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u32,                  // Unique identifier for the node
    pub position: Vector3<f64>,   // 3D coordinates (x, y, z) using nalgebra for vector math
}

impl Node {
    /// Calculate the Euclidean distance between this node and another node in 3D space.
    pub fn distance(&self, other: &Node) -> f64 {
        (self.position - other.position).norm()  // Use nalgebra's norm to compute the distance
    }

    /// Interpolate a scalar value from neighboring elements based on given weights.
    /// `element_values` contains the values at neighboring elements, and `weights` are the interpolation weights.
    /// Both slices must have the same length.
    pub fn interpolate_scalar(&self, element_values: &[f64], weights: &[f64]) -> f64 {
        assert_eq!(
            element_values.len(),
            weights.len(),
            "Mismatched lengths between element values and weights"
        );

        element_values
            .iter()
            .zip(weights.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    }

    /// Translate the node by a given displacement in 3D space.
    /// This modifies the node's position by adding the displacement to its current coordinates.
    pub fn translate(&mut self, displacement: Vector3<f64>) {
        self.position += displacement;
    }

    /// Check if this node is at the boundary of the domain (based on domain dimensions).
    pub fn is_boundary_node(&self, domain_min: Vector3<f64>, domain_max: Vector3<f64>) -> bool {
        let (x, y, z) = (self.position.x, self.position.y, self.position.z);
        let (min_x, min_y, min_z) = (domain_min.x, domain_min.y, domain_min.z);
        let (max_x, max_y, max_z) = (domain_max.x, domain_max.y, domain_max.z);

        (x == min_x || x == max_x) || (y == min_y || y == max_y) || (z == min_z || z == max_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_distance() {
        let node1 = Node { id: 1, position: Vector3::new(0.0, 0.0, 0.0) };
        let node2 = Node { id: 2, position: Vector3::new(3.0, 4.0, 0.0) };

        let dist = node1.distance(&node2);
        assert_eq!(dist, 5.0);  // Classic 3-4-5 triangle distance
    }

    #[test]
    fn test_interpolate_scalar() {
        let node = Node { id: 1, position: Vector3::new(1.0, 1.0, 1.0) };

        let element_values = vec![10.0, 20.0, 30.0];
        let weights = vec![0.2, 0.3, 0.5];

        let interpolated_value = node.interpolate_scalar(&element_values, &weights);
        assert!((interpolated_value - 23.0).abs() < 1e-6);
    }

    #[test]
    fn test_translate() {
        let mut node = Node { id: 1, position: Vector3::new(1.0, 1.0, 1.0) };
        node.translate(Vector3::new(2.0, 3.0, 4.0));
        assert_eq!(node.position, Vector3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn test_is_boundary_node() {
        let node = Node { id: 1, position: Vector3::new(0.0, 5.0, 10.0) };
        let domain_min = Vector3::new(0.0, 0.0, 0.0);
        let domain_max = Vector3::new(10.0, 10.0, 10.0);

        assert!(node.is_boundary_node(domain_min, domain_max));

        let internal_node = Node { id: 2, position: Vector3::new(5.0, 5.0, 5.0) };
        assert!(!internal_node.is_boundary_node(domain_min, domain_max));
    }
}
