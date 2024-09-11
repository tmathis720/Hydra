use crate::domain::Element;
use crate::domain::Face;
use crate::domain::Node;
use crate::input::gmsh::GmshParser;
use crate::domain::Neighbor;
use std::io;

#[derive(Default)]
pub struct Mesh {
    pub elements: Vec<Element>, 
    pub nodes: Vec<Node>,        
    pub faces: Vec<Face>, 
    pub neighbors: Vec<Neighbor>,
    pub face_element_relations: Vec<FaceElementRelation>, // (Face ID, Left Element ID, Right Element ID)
}

// Face-Element Relationship Table
pub struct FaceElementRelation {
    pub face_id: u32,
    pub left_element_id: u32,
    pub right_element_id: u32,
}

impl Mesh {
    pub fn new(elements: Vec<Element>, nodes: Vec<Node>, faces: Vec<Face>, face_element_relations: Vec<FaceElementRelation>) -> Self {
        let neighbors = Neighbor::assign_neighbors(&elements);
        Mesh { elements, nodes, faces, neighbors, face_element_relations }
    }

    // Load mesh from Gmsh parser
    pub fn load_from_gmsh(file_path: &str) -> Result<Mesh, io::Error> {
        let (nodes, elements, faces) = GmshParser::load_mesh(file_path)?;

        let mut face_element_relations = Vec::new();
        for face in &faces {
            let left_element_id = Mesh::find_left_element_id(&elements, face);
            let right_element_id = Mesh::find_right_element_id(&elements, face);
            face_element_relations.push(FaceElementRelation {
                face_id: face.id,
                left_element_id,
                right_element_id,
            });
        }

        Ok(Mesh::new(elements, nodes, faces, face_element_relations))
    }

    // Fetch an element by its ID
    pub fn get_element_by_id(&self, id: u32) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    // Fetch a mutable face by its ID
    pub fn get_face_by_id_mut(&mut self, id: u32) -> Option<&mut Face> {
        self.faces.iter_mut().find(|f| f.id == id)
    }

    // Find the left element of a face (first matching element based on shared nodes)
    pub fn find_left_element_id(elements: &Vec<Element>, face: &Face) -> u32 {
        for element in elements {
            if Mesh::element_shares_face_nodes(element, face) {
                return element.id;
            }
        }
        panic!("Left element not found for face {}", face.id);
    }

    // Find the right element of a face (second matching element based on shared nodes)
    pub fn find_right_element_id(elements: &Vec<Element>, face: &Face) -> u32 {
        for element in elements {
            if Mesh::element_shares_face_nodes(element, face) {
                return element.id;
            }
        }
        panic!("Right element not found for face {}", face.id);
    }

    // Helper function to check if an element shares nodes with a face
    fn element_shares_face_nodes(element: &Element, face: &Face) -> bool {
        let face_node_1 = face.nodes.0;
        let face_node_2 = face.nodes.1;
        element.nodes.contains(&face_node_1) && element.nodes.contains(&face_node_2)
    }
}


