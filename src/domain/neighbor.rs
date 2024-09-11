use crate::domain::Element;

pub struct Neighbor {
    pub element_id: usize,
    pub neighbors: Vec<usize>,
}

impl Neighbor {
    // Add a new neighbor for an element
    pub fn add_neighbor(&mut self, neighbor_id: usize) {
        if !self.neighbors.contains(&neighbor_id) {
            self.neighbors.push(neighbor_id);
        }
    }
    
    // Initialize neighbors for all elements based on shared nodes
    pub fn assign_neighbors(elements: &[Element]) -> Vec<Neighbor> {
        use std::collections::HashMap;
        
        let mut node_to_elements: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut neighbors = Vec::new();

        // Create neighbors for each element
        for element in elements {
            neighbors.push(Neighbor {
                element_id: element.id as usize,
                neighbors: Vec::new(),
            });
        }

        // Build the node-to-elements map
        for element in elements {
            for &node_id in &element.nodes {
                node_to_elements.entry(node_id).or_default().push(element.id as usize);
            }
        }

        // Now assign neighbors for each element
        for element in elements {
            let neighbor_ref = element.id as usize;
            if let Some(neighbor) = neighbors.iter_mut().find(|n| n.element_id == neighbor_ref) {
                for &node_id in &element.nodes {
                    if let Some(elements_sharing_node) = node_to_elements.get(&node_id) {
                        for &neighbor_id in elements_sharing_node {
                            if neighbor_id != element.id as usize {
                                neighbor.add_neighbor(neighbor_id);
                            }
                        }
                    }
                }
            }
        }

        neighbors
    }
}
