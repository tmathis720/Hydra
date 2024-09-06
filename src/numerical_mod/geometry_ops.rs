use crate::mesh_mod::mesh_ops::Node;

/// Computes the area of a triangle given three node references.
pub fn compute_triangle_area(node1: &Node, node2: &Node, node3: &Node) -> f64 {
    0.5 * ((node1.x * (node2.y - node3.y))
         + (node2.x * (node3.y - node1.y))
         + (node3.x * (node1.y - node2.y))).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        let node1 = Node { id: 1, x: 0.0, y: 0.0, z: 0.0 };
        let node2 = Node { id: 2, x: 1.0, y: 0.0, z: 0.0 };
        let node3 = Node { id: 3, x: 0.0, y: 1.0, z: 0.0 };

        let area = compute_triangle_area(&node1, &node2, &node3);
        assert!((area - 0.5).abs() < 1e-6, "Area mismatch");
    }
}
