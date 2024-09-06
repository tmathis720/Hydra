/// Mesh parsing and neighbor relationships
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::mesh_mod::element_ops::Element;
use crate::mesh_mod::neighbor_ops::build_neighbors;



use crate::numerical_mod::geometry_ops::compute_triangle_area;
// Define a struct to represent nodes
pub struct Node {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64, // Although we use 2D, Gmsh includes z-coordinates.
}



// Define a struct to represent the mesh
pub struct Mesh {
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
}

// Example: Create an Element Instance


impl Mesh {
    // Initialize an empty mesh
    pub fn new() -> Self {
        Mesh {
            nodes: Vec::new(),
            elements: Vec::new(),
        }
    }

    // Add node to the mesh
    pub fn add_node(&mut self, id: usize, x: f64, y: f64, z: f64) {
        self.nodes.push(Node { id, x, y, z });
    }

    // Add element (triangle) to the mesh
    pub fn add_element(&mut self, id: usize, node_ids: [usize; 3], tags: Vec<usize>) {
        let area = self.compute_element_area(&node_ids);
        // initialize state to a default value for now
        let mut state = 0.0;
        let mut flux = 0.0;

        self.elements.push(Element { 
            id, 
            node_ids, 
            tags,
            area,
            neighbors: Vec::new(), 
            state,
            flux,
        });
    }

    // Compute area of a triangle element given its node ids
    fn compute_element_area(&self, node_ids: &[usize; 3]) -> f64 {
        let node1 = &self.nodes[node_ids[0] - 1];
        let node2 = &self.nodes[node_ids[1] - 1];
        let node3 = &self.nodes[node_ids[2] - 1];

        // Calculate the area of the triangle using the determinant formula
        compute_triangle_area(node1, node2, node3)
    }

    // Load mesh from a Gmsh ASCII v2 file
    pub fn load_from_gmsh(file_path: &str) -> Result<Self, String> {
        let file = File::open(file_path).map_err(|_| "Failed to open file")?;
        let reader = BufReader::new(file);
        
        let mut mesh = Mesh::new();
        let mut parsing_nodes = false;
        let mut parsing_elements = false;
        let mut node_count = 0;
        let mut element_count = 0;

        for line in reader.lines() {
            let line = line.unwrap();

            if line.contains("$Nodes") {
                parsing_nodes = true;
                continue;
            }

            if parsing_nodes {
                if node_count == 0 {
                    // Read the number of nodes
                    node_count = line.parse::<usize>().expect("Invalid number of nodes");
                    println!("Number of nodes: {}", node_count);
                    continue;  // Move to the next line where node data starts
                }

                // Parse node data
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 4 {
                    let id = parts[0].parse::<usize>().expect("Invalid node ID");
                    let x = parts[1].parse::<f64>().expect("Invalid x-coordinate");
                    let y = parts[2].parse::<f64>().expect("Invalid y-coordinate");
                    let z = parts[3].parse::<f64>().expect("Invalid z-coordinate");
                    mesh.add_node(id, x, y, z);
                    node_count -= 1;
                }

                // Finish node parsing
                if node_count == 0 {
                    parsing_nodes = false;
                }
                continue;
            }

            if line.contains("$Elements") {
                parsing_elements = true;
                continue;
            }

            if parsing_elements {
                if element_count == 0 {
                    // Read the number of elements
                    element_count = line.parse::<usize>().expect("Invalid number of elements");
                    println!("Number of elements: {}", element_count);
                    continue;  // Move to the next line where element data starts
                }

                // Parse element data
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() > 4 {
                    let id = parts[0].parse::<usize>().expect("Invalid element ID");
                    let element_type = parts[1].parse::<usize>().expect("Invalid element type");

                    // Only handle triangular elements (type 2)
                    if element_type == 2 {
                        let num_tags = parts[2].parse::<usize>().expect("Invalid number of tags");
                        let mut tags = Vec::with_capacity(num_tags);
                        for i in 0..num_tags {
                            tags.push(parts[3 + i].parse::<usize>().expect("Invalid tag"));
                        }
                        let node_1 = parts[3 + num_tags].parse::<usize>().expect("Invalid node ID 1");
                        let node_2 = parts[4 + num_tags].parse::<usize>().expect("Invalid node ID 2");
                        let node_3 = parts[5 + num_tags].parse::<usize>().expect("Invalid node ID 3");

                        mesh.add_element(id, [node_1, node_2, node_3], tags);
                    }

                    element_count -= 1;
                }

                // Finish element parsing
                if element_count == 0 {
                    parsing_elements = false;
                }
            }
        }

        Ok(mesh)
    }

    pub fn build_neighbors(&mut self) {
        build_neighbors(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_node_creation() {
        let mut mesh = Mesh::new();
        mesh.add_node(1, 0.0, 0.0, 0.0);
        assert_eq!(mesh.nodes.len(), 1);
        assert_eq!(mesh.nodes[0].id, 1);
    }

    #[test]
    fn test_mesh_element_creation() {
        let mut mesh = Mesh::new();
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.elements[0].node_ids, [1, 2, 3]);
    }

    #[test]
    fn test_neighbor_computation() {
        let mut mesh = Mesh::new();
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        mesh.add_element(2, [2, 4, 3], vec![99, 2]);
        mesh.build_neighbors();
        assert_eq!(mesh.elements[0].neighbors, vec![2]);
    }

    #[test]

    fn test_load_gmsh_mesh() {
        // Create a mock mesh with nodes and elements
        let mut mesh = Mesh::new();

        // Add mock nodes
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);
        mesh.add_node(5, 0.5, 0.5, 0.0);
        mesh.add_node(6, 0.5, 1.5, 0.0);

        // Add mock elements (triangles)
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        mesh.add_element(2, [2, 4, 3], vec![99, 2]);

        // Check the number of nodes and elements
        assert_eq!(mesh.nodes.len(), 6, "Expected 6 nodes");
        assert_eq!(mesh.elements.len(), 2, "Expected 2 elements");

        // Validate a node's coordinates
        let node = &mesh.nodes[0];
        assert_eq!(node.id, 1);
        assert_eq!(node.x, 0.0);
        assert_eq!(node.y, 0.0);
        assert_eq!(node.z, 0.0);

        // Validate an element's node IDs and tags
        let element = &mesh.elements[0];
        assert_eq!(element.node_ids, [1, 2, 3]);
        assert_eq!(element.tags, vec![99, 2]);
    }

    #[test]

    fn test_element_area_and_neighbors() {
        let mut mesh = Mesh::new();

        // Add nodes (a simple right triangle mesh)
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);

        // Add elements (two triangles forming a square)
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        mesh.add_element(2, [2, 4, 3], vec![99, 2]);

        // Compute element areas
        assert!((mesh.elements[0].area - 0.5).abs() < 1e-6, "Area mismatch for element 1");
        assert!((mesh.elements[1].area - 0.5).abs() < 1e-6, "Area mismatch for element 2");

        // Build neighbor relationships
        mesh.build_neighbors();
        assert_eq!(mesh.elements[0].neighbors, vec![2], "Neighbors mismatch for element 1");
        assert_eq!(mesh.elements[1].neighbors, vec![1], "Neighbors mismatch for element 2");
    }
}