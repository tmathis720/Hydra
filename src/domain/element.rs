use crate::domain::node::Node;

pub struct Element {
    pub id: u32,
    pub nodes: Vec<usize>, // Update to use usize as index
    pub faces: Vec<u32>,
    pub pressure: f64,
    pub neighbor_ref: usize,
    pub mass: f64,
    pub element_type: u32,
    pub momentum: f64,
}

impl Element {
    pub fn has_node(&self, node_id: u32) -> bool {
        self.faces.contains(&node_id) // Example check for whether element has a specific node
    }
    // Calculate the area based on the positions of nodes
    pub fn area(&self, nodes: &[Node]) -> f64 {
        // Assuming 2D triangular elements for now
        let positions: Vec<(f64, f64, f64)> = self.nodes.iter().map(|&i| nodes[i].position).collect();
        if positions.len() == 3 {
            let (x1, y1, _z1) = positions[0];
            let (x2, y2, _z2) = positions[1];
            let (x3, y3, _z3) = positions[2];
            0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs()
        } else {
            0.0 // Handle other types of elements later
        }
    }
}

impl Default for Element {
    fn default() -> Self {
        Element {
            id: 0,
            nodes: vec![],
            faces: vec![],
            pressure: 0.0,
            neighbor_ref: 0,
            mass: 0.0,
            element_type: 0,
            momentum: 0.0,
        }
    }
}
