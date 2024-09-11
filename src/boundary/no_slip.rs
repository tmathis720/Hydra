use crate::domain::mesh::Mesh;
use crate::domain::face::Face;

pub struct NoSlipBoundary;

impl NoSlipBoundary {
    // Apply no-slip boundary condition on the given boundary nodes
    pub fn apply(&self, mesh: &mut Mesh) {
        for face in &mut mesh.faces {
            if self.is_no_slip_face(face) {
                face.velocity = (0.0, 0.0); // Set no-slip boundary condition on faces
            }
        }
    }

    // Function to determine if a node is on a boundary (can be expanded)
    pub fn is_no_slip_face(&self, face: &Face) -> bool {
        face.nodes.0 == 0 || face.nodes.1 == 0 // Adjust logic as needed
    }
}