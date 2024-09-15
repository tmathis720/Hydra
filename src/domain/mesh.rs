// src/domain/mesh.rs

use crate::domain::{Element, Face, Node};
use crate::input::gmsh::GmshParser;
use nalgebra::Vector3;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::io;

/// Custom error type for the Mesh module
#[derive(Debug)]
pub enum MeshError {
    IoError(io::Error),
    ElementNotFound(String),
    FaceNotFound(String),
    Other(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::IoError(e) => write!(f, "IO error: {}", e),
            MeshError::ElementNotFound(msg) => write!(f, "Element not found: {}", msg),
            MeshError::FaceNotFound(msg) => write!(f, "Face not found: {}", msg),
            MeshError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for MeshError {}

impl From<io::Error> for MeshError {
    fn from(error: io::Error) -> Self {
        MeshError::IoError(error)
    }
}

/// Represents the relationship between a face and the elements that share it.
pub struct FaceElementRelation {
    pub face_id: u32,               // ID of the face
    pub connected_elements: Vec<u32>, // IDs of elements sharing the face
}

#[derive(Default)]
pub struct Mesh {
    pub elements: Vec<Element>,         // List of all elements in the mesh
    pub nodes: Vec<Node>,               // List of all nodes in the mesh
    pub faces: Vec<Face>,               // List of all faces in the mesh
    pub neighbors: HashMap<u32, Vec<u32>>, // Map from element ID to neighboring element IDs
    pub face_element_relations: Vec<FaceElementRelation>, // Face to elements mapping
}

/// Represents a unique key for a face based on its node indices.
/// Node indices are sorted to ensure consistency.
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
struct FaceKey(Vec<usize>);

impl FaceKey {
    fn new(mut nodes: Vec<usize>) -> Self {
        nodes.sort_unstable();
        FaceKey(nodes)
    }
}

impl Mesh {
    /// Constructor for creating a new mesh with elements, nodes, and faces.
    /// Automatically builds face-element relationships and assigns neighbors.
    pub fn new(
        elements: Vec<Element>,
        nodes: Vec<Node>,
        faces: Vec<Face>,
    ) -> Result<Self, MeshError> {
        let (face_element_relations, face_id_map) = Mesh::build_face_element_relations(&elements)?;

        let neighbors = Mesh::assign_neighbors(&elements, &face_element_relations);

        Ok(Mesh {
            elements,
            nodes,
            faces,
            neighbors,
            face_element_relations,
        })
    }

    /// Load mesh data from a Gmsh file.
    pub fn load_from_gmsh(file_path: &str) -> Result<Mesh, MeshError> {
        let (nodes, elements, faces) = GmshParser::load_mesh(file_path)?;

        Mesh::new(elements, nodes, faces)
    }

    /// Builds the face-element relationships based on shared faces.
    /// Returns a tuple containing the relationships and a map from FaceKey to face ID.
    fn build_face_element_relations(
        elements: &[Element],
    ) -> Result<(Vec<FaceElementRelation>, HashMap<FaceKey, u32>), MeshError> {
        let mut face_to_elements: HashMap<FaceKey, Vec<u32>> = HashMap::new();

        // Generate faces for each element
        for element in elements {
            let element_faces = element.generate_faces();

            for face_nodes in element_faces {
                let face_key = FaceKey::new(face_nodes);
                face_to_elements
                    .entry(face_key)
                    .or_insert_with(Vec::new)
                    .push(element.id);
            }
        }

        // Assign face IDs and create face-element relations
        let mut face_element_relations = Vec::new();
        let mut face_id_counter = 0;
        let mut face_id_map = HashMap::new();

        for (face_key, element_ids) in face_to_elements {
            face_id_map.insert(face_key.clone(), face_id_counter);
            face_element_relations.push(FaceElementRelation {
                face_id: face_id_counter,
                connected_elements: element_ids,
            });
            face_id_counter += 1;
        }

        Ok((face_element_relations, face_id_map))
    }

    /// Assigns neighbors to each element based on shared faces.
    fn assign_neighbors(
        elements: &[Element],
        face_element_relations: &[FaceElementRelation],
    ) -> HashMap<u32, Vec<u32>> {
        let mut neighbors = HashMap::new();

        for relation in face_element_relations {
            if relation.connected_elements.len() > 1 {
                for &element_id in &relation.connected_elements {
                    let neighbor_ids = relation
                        .connected_elements
                        .iter()
                        .cloned()
                        .filter(|&id| id != element_id)
                        .collect::<Vec<_>>();
                    neighbors
                        .entry(element_id)
                        .or_insert_with(Vec::new)
                        .extend(neighbor_ids);
                }
            }
        }

        // Remove duplicate neighbor entries
        for neighbor_list in neighbors.values_mut() {
            neighbor_list.sort_unstable();
            neighbor_list.dedup();
        }

        neighbors
    }

    /// Get an element by its ID (returns None if not found).
    pub fn get_element_by_id(&self, id: u32) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// Get a mutable reference to an element by its ID.
    pub fn get_element_by_id_mut(&mut self, id: u32) -> Option<&mut Element> {
        self.elements.iter_mut().find(|e| e.id == id)
    }

    /// Get a mutable reference to a face by its ID.
    pub fn get_face_by_id_mut(&mut self, id: u32) -> Option<&mut Face> {
        self.faces.iter_mut().find(|f| f.id == id)
    }

    /// Get the elements connected to a face by its ID.
    pub fn get_elements_connected_to_face(&self, face_id: u32) -> Option<&[u32]> {
        self.face_element_relations
            .iter()
            .find(|rel| rel.face_id == face_id)
            .map(|rel| rel.connected_elements.as_slice())
    }

    /// Get all faces connected to an element by its ID.
    pub fn get_faces_connected_to_element(&self, element_id: u32) -> Vec<u32> {
        self.face_element_relations
            .iter()
            .filter(|relation| relation.connected_elements.contains(&element_id))
            .map(|relation| relation.face_id)
            .collect()
    }

    /// Return the neighboring elements of a given element by its ID.
    pub fn get_neighbors_of_element(&self, element_id: u32) -> Vec<u32> {
        self.neighbors
            .get(&element_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Return the width of the domain based on node positions.
    pub fn domain_width(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let min_x = self
            .nodes
            .iter()
            .map(|node| node.position.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = self
            .nodes
            .iter()
            .map(|node| node.position.x)
            .fold(f64::NEG_INFINITY, f64::max);
        max_x - min_x
    }

    /// Return the height of the domain based on node positions.
    pub fn domain_height(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let min_y = self
            .nodes
            .iter()
            .map(|node| node.position.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = self
            .nodes
            .iter()
            .map(|node| node.position.y)
            .fold(f64::NEG_INFINITY, f64::max);
        max_y - min_y
    }

    /// Return the depth of the domain based on node positions (for 3D support).
    pub fn domain_depth(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let min_z = self
            .nodes
            .iter()
            .map(|node| node.position.z)
            .fold(f64::INFINITY, f64::min);
        let max_z = self
            .nodes
            .iter()
            .map(|node| node.position.z)
            .fold(f64::NEG_INFINITY, f64::max);
        max_z - min_z
    }

    /// Get the boundary faces of the mesh based on boundary criteria.
    pub fn get_boundary_faces(&self) -> Vec<&Face> {
        self.faces
            .iter()
            .filter(|face| face.is_boundary)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Element, Face, Node};
    use nalgebra::Vector3;

    #[test]
    fn test_face_key_equality() {
        let face_key1 = FaceKey::new(vec![1, 2]);
        let face_key2 = FaceKey::new(vec![2, 1]);
        let face_key3 = FaceKey::new(vec![1, 3]);

        assert_eq!(face_key1, face_key2, "FaceKey should be equal regardless of node order");
        assert_ne!(face_key1, face_key3, "Different nodes should result in different FaceKeys");
    }

    #[test]
    fn test_build_face_element_relations() {
        // Create nodes
        let _nodes = vec![
            Node { id: 0, position: Vector3::new(0.0, 0.0, 0.0) },
            Node { id: 1, position: Vector3::new(1.0, 0.0, 0.0) },
            Node { id: 2, position: Vector3::new(1.0, 1.0, 0.0) },
            Node { id: 3, position: Vector3::new(0.0, 1.0, 0.0) },
        ];

        // Create elements (two adjacent triangles sharing an edge)
        let elements = vec![
            Element::new(0, vec![0, 1, 2], vec![], 0),
            Element::new(1, vec![0, 2, 3], vec![], 0),
        ];

        // Implement generate_faces in Element struct if not already done
        // For testing purposes, let's assume it's implemented as shown earlier

        let (face_element_relations, face_id_map) =
            Mesh::build_face_element_relations(&elements).expect("Failed to build face-element relations");

        // There should be 5 faces in total (3 edges per triangle, 1 shared edge)
        assert_eq!(face_element_relations.len(), 5, "Should have 5 unique faces");

        // Check that the shared edge connects both elements
        let shared_face_key = FaceKey::new(vec![2, 0]); // Edge between nodes 0 and 2
        let shared_face_id = face_id_map.get(&shared_face_key).expect("Shared face not found");

        let shared_face_relation = face_element_relations
            .iter()
            .find(|rel| rel.face_id == *shared_face_id)
            .expect("Shared face relation not found");

        assert_eq!(
            shared_face_relation.connected_elements.len(),
            2,
            "Shared face should connect two elements"
        );
    }

    #[test]
    fn test_assign_neighbors() {
        // Create elements
        let elements = vec![
            Element::new(0, vec![0, 1, 2], vec![], 0),
            Element::new(1, vec![0, 2, 3], vec![], 0),
            Element::new(2, vec![2, 3, 4], vec![], 0),
        ];

        // Build face-element relations
        let (face_element_relations, _face_id_map) =
            Mesh::build_face_element_relations(&elements).expect("Failed to build face-element relations");

        // Assign neighbors
        let neighbors = Mesh::assign_neighbors(&elements, &face_element_relations);

        // Element 0 should have element 1 as a neighbor
        assert_eq!(
            neighbors.get(&0).cloned().unwrap_or_else(Vec::new),
            vec![1],
            "Element 0 should have Element 1 as neighbor"
        );

        // Element 1 should have elements 0 and 2 as neighbors
        let mut element1_neighbors = neighbors.get(&1).cloned().unwrap_or_else(Vec::new);
        element1_neighbors.sort_unstable();
        assert_eq!(
            element1_neighbors,
            vec![0, 2],
            "Element 1 should have Elements 0 and 2 as neighbors"
        );

        // Element 2 should have element 1 as a neighbor
        assert_eq!(
            neighbors.get(&2).cloned().unwrap_or_else(Vec::new),
            vec![1],
            "Element 2 should have Element 1 as neighbor"
        );
    }

    #[test]
    fn test_get_elements_connected_to_face() {
        // Create elements
        let elements = vec![
            Element::new(0, vec![0, 1, 2], vec![], 0),
            Element::new(1, vec![0, 2, 3], vec![], 0),
        ];

        // Build face-element relations
        let (face_element_relations, face_id_map) =
            Mesh::build_face_element_relations(&elements).expect("Failed to build face-element relations");

        // Create a mesh instance (faces are empty for this test)
        let mesh = Mesh {
            elements,
            nodes: vec![],
            faces: vec![],
            neighbors: HashMap::new(),
            face_element_relations,
        };

        // Get face ID for the shared face between elements 0 and 1
        let shared_face_key = FaceKey::new(vec![0, 2]);
        let shared_face_id = face_id_map.get(&shared_face_key).expect("Shared face not found");

        let connected_elements = mesh
            .get_elements_connected_to_face(*shared_face_id)
            .expect("Connected elements not found");

        // The shared face should connect elements 0 and 1
        let mut connected_elements_sorted = connected_elements.to_vec();
        connected_elements_sorted.sort_unstable();
        assert_eq!(connected_elements_sorted, vec![0, 1], "Shared face should connect elements 0 and 1");
    }

    #[test]
    fn test_get_faces_connected_to_element() {
        // Create elements
        let elements = vec![
            Element::new(0, vec![0, 1, 2], vec![], 0),
            Element::new(1, vec![0, 2, 3], vec![], 0),
        ];

        // Build face-element relations
        let (face_element_relations, _face_id_map) =
            Mesh::build_face_element_relations(&elements).expect("Failed to build face-element relations");

        // Create a mesh instance (faces are empty for this test)
        let mesh = Mesh {
            elements,
            nodes: vec![],
            faces: vec![],
            neighbors: HashMap::new(),
            face_element_relations,
        };

        let faces_connected_to_element0 = mesh.get_faces_connected_to_element(0);
        // Element 0 has 3 faces
        assert_eq!(faces_connected_to_element0.len(), 3, "Element 0 should have 3 faces");
    }

    #[test]
    fn test_domain_dimensions() {
        // Create nodes
        let nodes = vec![
            Node { id: 0, position: Vector3::new(0.0, 0.0, 0.0) },
            Node { id: 1, position: Vector3::new(2.0, 0.0, 0.0) },
            Node { id: 2, position: Vector3::new(2.0, 1.0, 0.0) },
            Node { id: 3, position: Vector3::new(0.0, 1.0, 0.0) },
        ];

        let mesh = Mesh {
            elements: vec![],
            nodes,
            faces: vec![],
            neighbors: HashMap::new(),
            face_element_relations: vec![],
        };

        assert_eq!(mesh.domain_width(), 2.0, "Domain width should be 2.0");
        assert_eq!(mesh.domain_height(), 1.0, "Domain height should be 1.0");
        assert_eq!(mesh.domain_depth(), 0.0, "Domain depth should be 0.0");
    }

    #[test]
    fn test_get_boundary_faces() {
        // Create faces with boundary flags
        let faces = vec![
            Face {
                id: 0,
                nodes: vec![0, 1],
                is_boundary: true,
                ..Default::default()
            },
            Face {
                id: 1,
                nodes: vec![1, 2],
                is_boundary: false,
                ..Default::default()
            },
            Face {
                id: 2,
                nodes: vec![2, 3],
                is_boundary: true,
                ..Default::default()
            },
        ];

        let mesh = Mesh {
            elements: vec![],
            nodes: vec![],
            faces,
            neighbors: HashMap::new(),
            face_element_relations: vec![],
        };

        let boundary_faces = mesh.get_boundary_faces();
        assert_eq!(boundary_faces.len(), 2, "There should be 2 boundary faces");
        let boundary_face_ids: Vec<u32> = boundary_faces.iter().map(|f| f.id).collect();
        assert_eq!(boundary_face_ids, vec![0, 2], "Boundary face IDs should be 0 and 2");
    }
}
