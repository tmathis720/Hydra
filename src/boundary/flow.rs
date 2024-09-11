use crate::domain::mesh::Mesh;
use crate::domain::element::Element;
use crate::domain::face::Face;

pub struct FlowBoundary {
    pub inflow_velocity: f64, // The velocity at which fluid enters or leaves
}

impl FlowBoundary {
    // Apply flow boundary condition to nodes on an inflow/outflow boundary
    pub fn apply(&self, mesh: &mut Mesh) {
        let boundary_faces: Vec<usize> = mesh.faces.iter().enumerate()
            .filter(|(_, face)| self.is_flow_boundary_face(face, mesh))
            .map(|(i, _)| i)
            .collect();

        for &i in &boundary_faces {
            mesh.faces[i].velocity.0 = self.inflow_velocity; // Apply inflow velocity
        }
    }

    pub fn is_flow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes.0 as usize];
        let node_2 = &mesh.nodes[face.nodes.1 as usize];
        node_1.position.0 == 0.0 || node_2.position.0 == 0.0 // Example condition for inflow boundary
    }
}

pub struct Inflow {
    pub rate: f64, // The rate at which mass/momentum is added to the system
}

impl Inflow {
    /// Apply the inflow boundary condition by adding mass and momentum
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass += self.rate * dt;
        element.momentum += self.rate * element.pressure * dt;
    }
}

pub struct Outflow {
    pub rate: f64, // The rate at which mass/momentum is removed from the system
}

impl Outflow {
    /// Apply the outflow boundary condition by removing mass and momentum
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass = (element.mass - self.rate * dt).max(0.0); // Prevent negative mass
        element.momentum = (element.momentum - self.rate * element.pressure * dt).max(0.0); // Prevent negative momentum
    }
}
