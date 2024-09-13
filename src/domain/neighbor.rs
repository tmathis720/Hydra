use crate::domain::Element;
use std::collections::HashMap;

pub struct Neighbor {
    pub element_id: usize,      // The ID of the element
    pub neighbors: Vec<usize>,  // List of neighboring element IDs
}

impl Neighbor {
    /// Add a new neighbor for an element, avoiding duplicates.
    pub fn add_neighbor(&mut self, neighbor_id: usize) {
        if !self.neighbors.contains(&neighbor_id) {
            self.neighbors.push(neighbor_id);
        }
    }

    /// Assign neighbors for all elements based on shared nodes.
    /// Elements that share nodes are considered neighbors.
    pub fn assign_neighbors(elements: &[Element]) -> Vec<Neighbor> {
        let mut node_to_elements: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut neighbors: Vec<Neighbor> = elements
            .iter()
            .map(|element| Neighbor {
                element_id: element.id as usize,
                neighbors: Vec::new(),
            })
            .collect();

        // Map each node to the elements that share it
        for element in elements {
            for &node_id in &element.nodes {
                node_to_elements.entry(node_id).or_default().push(element.id as usize);
            }
        }

        // Assign neighbors to each element based on shared nodes
        for element in elements {
            let element_id = element.id as usize;
            if let Some(neighbor) = neighbors.iter_mut().find(|n| n.element_id == element_id) {
                for &node_id in &element.nodes {
                    if let Some(shared_elements) = node_to_elements.get(&node_id) {
                        for &neighbor_id in shared_elements {
                            if neighbor_id != element_id {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Element;

    #[test]
    fn test_neighbor_assignment() {
        // Mock elements for testing (nodes are shared between some elements)
        let elements = vec![
            Element {
                id: 1,
                nodes: vec![0, 1, 2],
                ..Default::default()
            },
            Element {
                id: 2,
                nodes: vec![2, 3, 4],
                ..Default::default()
            },
            Element {
                id: 3,
                nodes: vec![1, 2, 5],
                ..Default::default()
            },
        ];

        let neighbors = Neighbor::assign_neighbors(&elements);

        // Check that the correct neighbors are assigned
        assert_eq!(neighbors.len(), 3);

        // Element 1 should have elements 2 and 3 as neighbors
        let element_1_neighbors = &neighbors[0].neighbors;
        assert!(element_1_neighbors.contains(&2));
        assert!(element_1_neighbors.contains(&3));

        // Element 2 should have elements 1 and 3 as neighbors
        let element_2_neighbors = &neighbors[1].neighbors;
        assert!(element_2_neighbors.contains(&1));
        assert!(element_2_neighbors.contains(&3));

        // Element 3 should have element 1 as a neighbor
        let element_3_neighbors = &neighbors[2].neighbors;
        assert!(element_3_neighbors.contains(&1));
    }
}
