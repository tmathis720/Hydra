use crate::domain::mesh::Mesh;
use crate::domain::face::Face;

pub struct FreeSurfaceBoundary;

impl FreeSurfaceBoundary {
    // Apply free surface boundary condition to nodes on the top boundary
    pub fn apply(&self, mesh: &mut Mesh) {
        // Collect indices of faces that satisfy the free surface condition
        let free_surface_faces: Vec<usize> = mesh.faces.iter().enumerate()
            .filter(|(_, face)| self.is_free_surface_face(face, mesh)) // Immutable borrow
            .map(|(i, _)| i)
            .collect();

        // Apply the boundary condition to the collected faces
        for &i in &free_surface_faces {
            mesh.faces[i].velocity.1 = mesh.faces[i].velocity.1; // Apply free surface condition
        }
    }

    pub fn is_free_surface_face(&self, face: &Face, mesh: &Mesh) -> bool {
        // Check if the face is part of the free surface
        let node_1 = &mesh.nodes[face.nodes.0 as usize];
        let node_2 = &mesh.nodes[face.nodes.1 as usize];
        node_1.position.1 == 1.0 || node_2.position.1 == 1.0 // Example condition for free surface
    }
}
