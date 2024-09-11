// Import the necessary modules and types
use hydra::input::gmsh::GmshParser;
use hydra::domain::mesh::{Mesh, FaceElementRelation};
use hydra::domain::element::Element;
use hydra::domain::face::Face;

fn main() {
    // Load the mesh from a Gmsh file
    let (nodes, elements, faces) = GmshParser::load_mesh("C:/rust_projects/HYDRA/inputs/test.msh2")
        .expect("Failed to load mesh");

    // Define face-element relations (this may need to be loaded or computed)
    let face_element_relations = vec![
        FaceElementRelation {
            face_id: 0,
            left_element_id: 0,
            right_element_id: 1,
        },
        FaceElementRelation {
            face_id: 1,
            left_element_id: 1,
            right_element_id: 2,
        },
    ];

    // Create the mesh
    let mut mesh = Mesh::new(elements, nodes, faces, face_element_relations);

    // Example of verbose output during simulation or processing
    println!("Mesh successfully created with {} elements and {} faces.", mesh.elements.len(), mesh.faces.len());

    // Example: Iterate over the elements and faces to display their properties
    for element in &mesh.elements {
        println!(
            "Element ID: {}, Pressure: {}, Number of Neighbors: {}",
            element.id,
            element.pressure,
            element.neighbor_ref,
        );
    }

    for face in &mesh.faces {
        println!(
            "Face ID: {}, Velocity: {:?}, Area: {}",
            face.id,
            face.velocity,
            face.area
        );
    }

    // Further logic for the simulation goes here...
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_face_element_relation() {
        let element_1 = Element {
            id: 0,
            pressure: 1.0,
            ..Default::default()
        };

        let element_2 = Element {
            id: 1,
            pressure: 2.0,
            ..Default::default()
        };

        let face = Face {
            id: 0,
            velocity: (0.0, 0.0),
            ..Default::default()
        };

        let face_relation = FaceElementRelation {
            face_id: face.id,
            left_element_id: element_1.id,
            right_element_id: element_2.id,
        };

        // Verbose test output
        println!("Testing FaceElementRelation with Face ID: {}", face_relation.face_id);
        println!(
            "Linking Left Element ID: {} with Right Element ID: {}",
            face_relation.left_element_id, face_relation.right_element_id
        );

        // Assert that the face relation correctly links the elements
        assert_eq!(
            face_relation.left_element_id, element_1.id,
            "Left element ID does not match."
        );
        assert_eq!(
            face_relation.right_element_id, element_2.id,
            "Right element ID does not match."
        );
    }

    #[test]
    fn test_mesh_creation() {
        // Create a dummy mesh with elements, nodes, and faces
        let elements = vec![
            Element {
                id: 0,
                pressure: 1.0,
                ..Default::default()
            },
            Element {
                id: 1,
                pressure: 2.0,
                ..Default::default()
            },
        ];

        let nodes = vec![]; // Fill with dummy nodes
        let faces = vec![
            Face {
                id: 0,
                velocity: (0.0, 0.0),
                ..Default::default()
            },
        ];

        let face_element_relations = vec![
            FaceElementRelation {
                face_id: 0,
                left_element_id: 0,
                right_element_id: 1,
            },
        ];

        let mesh = Mesh::new(elements, nodes, faces, face_element_relations);

        // Verbose test output
        println!("Testing Mesh creation:");
        println!("Mesh has {} elements and {} faces.", mesh.elements.len(), mesh.faces.len());

        assert_eq!(mesh.elements.len(), 2, "Mesh should have 2 elements.");
        assert_eq!(mesh.faces.len(), 1, "Mesh should have 1 face.");
    }
}
