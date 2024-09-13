use crate::domain::{Element, Face, Node, Neighbor};
use crate::input::gmsh::GmshParser;
use std::io;

#[derive(Default)]
pub struct Mesh {
    pub elements: Vec<Element>,         // List of all elements in the mesh
    pub nodes: Vec<Node>,               // List of all nodes in the mesh
    pub faces: Vec<Face>,               // List of all faces in the mesh
    pub neighbors: Vec<Neighbor>,       // List of neighboring relationships
    pub face_element_relations: Vec<FaceElementRelation>, // (Face ID, Left Element ID, Right Element ID)
}

// Face-Element Relationship Table
pub struct FaceElementRelation {
    pub face_id: u32,           // ID of the face
    pub left_element_id: u32,   // ID of the element on the left side of the face
    pub right_element_id: u32,  // ID of the element on the right side of the face
}

impl Mesh {
    /// Constructor for creating a new mesh with elements, nodes, and faces
    pub fn new(
        elements: Vec<Element>,
        nodes: Vec<Node>,
        faces: Vec<Face>,
        face_element_relations: Vec<FaceElementRelation>,
    ) -> Self {
        let neighbors = Neighbor::assign_neighbors(&elements); // Automatically assign neighbors based on elements
        Mesh {
            elements,
            nodes,
            faces,
            neighbors,
            face_element_relations,
        }
    }

    /// Load mesh data from a Gmsh file
    pub fn load_from_gmsh(file_path: &str) -> Result<Mesh, io::Error> {
        let (nodes, elements, faces) = GmshParser::load_mesh(file_path)?;

        // Build face-element relationships based on shared nodes
        let mut face_element_relations = Vec::new();
        for face in &faces {
            let left_element_id = Mesh::find_left_element_id(&elements, face)
                .unwrap_or_else(|| panic!("Left element not found for face {}", face.id));
            let right_element_id = Mesh::find_right_element_id(&elements, face)
                .unwrap_or_else(|| panic!("Right element not found for face {}", face.id));

            face_element_relations.push(FaceElementRelation {
                face_id: face.id,
                left_element_id,
                right_element_id,
            });
        }

        Ok(Mesh::new(elements, nodes, faces, face_element_relations))
    }

    /// Get an element by its ID (returns None if not found)
    pub fn get_element_by_id(&self, id: u32) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// Get a mutable reference to a face by its ID (returns None if not found)
    pub fn get_face_by_id_mut(&mut self, id: u32) -> Option<&mut Face> {
        self.faces.iter_mut().find(|f| f.id == id)
    }

    /// Find the left element of a face (first matching element based on shared nodes)
    pub fn find_left_element_id(elements: &[Element], face: &Face) -> Option<u32> {
        elements
            .iter()
            .find(|element| Mesh::element_shares_face_nodes(element, face))
            .map(|element| element.id)
    }

    /// Find the right element of a face (second matching element based on shared nodes)
    pub fn find_right_element_id(elements: &[Element], face: &Face) -> Option<u32> {
        elements
            .iter()
            .find(|element| Mesh::element_shares_face_nodes(element, face))
            .map(|element| element.id)
    }

    /// Helper function to check if an element shares nodes with a face
    fn element_shares_face_nodes(element: &Element, face: &Face) -> bool {
        let face_node_1 = face.nodes[0];
        let face_node_2 = face.nodes[1];
        element.nodes.contains(&face_node_1) && element.nodes.contains(&face_node_2)
    }

    /// Return the width of the domain based on node positions
    pub fn domain_width(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;  // Handle case where there are no nodes in the mesh
        }

        let min_x = self.nodes.iter().map(|node| node.position.x).fold(f64::INFINITY, f64::min);
        let max_x = self.nodes.iter().map(|node| node.position.x).fold(f64::NEG_INFINITY, f64::max);
        max_x - min_x
    }

    /// Return the height of the domain based on node positions
    pub fn domain_height(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let min_y = self.nodes.iter().map(|node| node.position.y).fold(f64::INFINITY, f64::min);
        let max_y = self.nodes.iter().map(|node| node.position.y).fold(f64::NEG_INFINITY, f64::max);
        max_y - min_y
    }

    /// Return the depth of the domain based on node positions (for 3D support)
    pub fn domain_depth(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let min_z = self.nodes.iter().map(|node| node.position.z).fold(f64::INFINITY, f64::min);
        let max_z = self.nodes.iter().map(|node| node.position.z).fold(f64::NEG_INFINITY, f64::max);
        max_z - min_z
    }

    /// Get the two elements connected to a face (returns two Option<&Element> values).
    ///
    /// The first is the element on the left, and the second is the element on the right.
    /// Returns None if the element is not found.
    /// Retrieves mutable references to the elements connected to a face.
    /// This returns a tuple of mutable references to the left and right elements, if they exist.
    pub fn get_connected_elements(&mut self, face: &Face) -> (Option<&mut Element>, Option<&mut Element>) {
        // First, find the face-element relationship without borrowing `self` mutably
        if let Some(relation) = self.face_element_relations.iter().find(|rel| rel.face_id == face.id) {
            let left_element_index = self.elements.iter().position(|e| e.id == relation.left_element_id);
            let right_element_index = self.elements.iter().position(|e| e.id == relation.right_element_id);

            // Now borrow `self` mutably only after we have the indices
            match (left_element_index, right_element_index) {
                (Some(left_idx), Some(right_idx)) => {
                    let (left, right) = self.elements.split_at_mut(left_idx.max(right_idx));
                    if left_idx < right_idx {
                        (Some(&mut left[left_idx]), Some(&mut right[0]))
                    } else {
                        (Some(&mut right[0]), Some(&mut left[right_idx]))
                    }
                }
                (Some(left_idx), None) => (Some(&mut self.elements[left_idx]), None),
                (None, Some(right_idx)) => (None, Some(&mut self.elements[right_idx])),
                (None, None) => (None, None),
            }
        } else {
            (None, None)  // If no relationship is found, return None for both
        }
    }

    /// Get the first element connected to a given node index (returns None if not found)
    pub fn get_element_connected_to_node(&self, node_index: usize) -> Option<usize> {
        self.elements
            .iter()
            .position(|element| element.nodes.contains(&node_index))
    }

    /// Get all faces connected to an element by its ID
    pub fn get_faces_connected_to_element(&self, element_id: u32) -> Vec<&Face> {
        self.face_element_relations
            .iter()
            .filter_map(|relation| {
                if relation.left_element_id == element_id || relation.right_element_id == element_id {
                    self.faces.iter().find(|face| face.id == relation.face_id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the neighboring elements of a given element by its ID
    pub fn get_neighbors_of_element(&self, element_id: u32) -> Vec<&Element> {
        self.face_element_relations
            .iter()
            .filter_map(|relation| {
                if relation.left_element_id == element_id {
                    self.get_element_by_id(relation.right_element_id)
                } else if relation.right_element_id == element_id {
                    self.get_element_by_id(relation.left_element_id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the boundary faces of the mesh based on boundary criteria (now handles 3D boundaries)
    pub fn get_boundary_faces(&self, domain_bounds: &[(f64, f64)]) -> Vec<&Face> {
        let node_positions: Vec<(f64, f64, f64)> = self
            .nodes
            .iter()
            .map(|n| (n.position.x, n.position.y, n.position.z))
            .collect();

        self.faces
            .iter()
            .filter(|face| face.is_boundary_face(domain_bounds, &node_positions))
            .collect()
    }
}
