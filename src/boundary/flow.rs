use crate::domain::{Mesh, Element, Face};
use crate::domain::FlowField;
use crate::boundary::BoundaryType;
use nalgebra::Vector3;  // Use nalgebra's Vector3 for 3D operations

const MASS_CONSERVATION_THRESHOLD: f64 = 1e-6;  // Threshold for mass conservation check

/// FlowBoundary structure applies flow conditions such as inflow and outflow
pub struct FlowBoundary {
    pub inflow_velocity: Vector3<f64>,  // 3D inflow velocity
    pub outflow_velocity: Vector3<f64>, // 3D outflow velocity (if required)
}

impl FlowBoundary {
    /// Apply the flow boundary condition to inflow and outflow boundary faces
    pub fn apply(&self, mesh: &mut Mesh) {
        let inflow_faces = self.get_boundary_faces(mesh, BoundaryType::Inflow);
        let outflow_faces = self.get_boundary_faces(mesh, BoundaryType::Outflow);

        // Apply inflow conditions
        for face_index in inflow_faces {
            let face = &mut mesh.faces[face_index];
            face.velocity = self.inflow_velocity;  // Apply 3D inflow velocity
            self.apply_inflow_to_elements(face.id, mesh);
        }

        // Apply outflow conditions
        for face_index in outflow_faces {
            let face = &mut mesh.faces[face_index];
            face.velocity = self.outflow_velocity;  // Apply 3D outflow velocity
            self.apply_outflow_to_elements(face.id, mesh);
        }
    }

    /// Apply inflow condition to connected elements of a face
    fn apply_inflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.left_element_id) {
                    left_element.velocity = self.inflow_velocity;
                    left_element.momentum += self.inflow_velocity * left_element.mass;
                }
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.right_element_id) {
                    right_element.velocity = self.inflow_velocity;
                    right_element.momentum += self.inflow_velocity * right_element.mass;
                }
                break;
            }
        }
    }

    /// Apply outflow condition to connected elements of a face
    fn apply_outflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.left_element_id) {
                    left_element.velocity = self.outflow_velocity;
                    left_element.momentum -= self.outflow_velocity * left_element.mass;
                }
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.right_element_id) {
                    right_element.velocity = self.outflow_velocity;
                    right_element.momentum -= self.outflow_velocity * right_element.mass;
                }
                break;
            }
        }
    }

    /// Helper function to get inflow/outflow boundary faces
    fn get_boundary_faces(&self, mesh: &Mesh, boundary_type: BoundaryType) -> Vec<usize> {
        mesh.faces.iter().enumerate()
            .filter_map(|(i, face)| {
                if (boundary_type == BoundaryType::Inflow && self.is_inflow_boundary_face(face, mesh)) ||
                   (boundary_type == BoundaryType::Outflow && self.is_outflow_boundary_face(face, mesh)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if the face is at the inflow boundary
    fn is_inflow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes[0]];
        let node_2 = &mesh.nodes[face.nodes[1]];
        node_1.position.x == 0.0 || node_2.position.x == 0.0  // Simple check for inflow at x = 0
    }

    /// Check if the face is at the outflow boundary
    fn is_outflow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes[0]];
        let node_2 = &mesh.nodes[face.nodes[1]];
        node_1.position.x == mesh.domain_width() || node_2.position.x == mesh.domain_width()  // Check for outflow at the domain's right edge
    }

    /// Check for mass conservation by comparing initial and current total mass
    pub fn check_mass_conservation(&self, flow_field: &FlowField) -> bool {
        let total_mass: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        let mass_difference = total_mass - flow_field.initial_mass;
        mass_difference.abs() < MASS_CONSERVATION_THRESHOLD
    }
}

/// Struct to represent inflow boundary conditions
pub struct Inflow {
    pub rate: f64,  // Rate of mass/momentum added to the system
}

impl Inflow {
    /// Apply inflow by adding mass and momentum to the element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass += self.rate * dt;  // Add mass
        element.momentum += Vector3::new(self.rate * element.pressure * dt, 0.0, 0.0);  // Add momentum (assumed in x-direction)
    }
}

/// Struct to represent outflow boundary conditions
pub struct Outflow {
    pub rate: f64,  // Rate of mass/momentum removed from the system
}

impl Outflow {
    /// Apply outflow by removing mass and momentum from the element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass = (element.mass - self.rate * dt).max(0.0);  // Ensure mass does not go negative
        element.momentum.x = (element.momentum.x - self.rate * element.pressure * dt).max(0.0);  // Prevent negative momentum
    }
}
