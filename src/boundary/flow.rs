use crate::domain::{Mesh, Element, Face};
use crate::solver::FlowField;
use crate::boundary::BoundaryType;

const MASS_CONSERVATION_THRESHOLD: f64 = 1e-6;  // Example threshold value
// FlowBoundary structure that applies flow conditions for both inflow and outflow
pub struct FlowBoundary {
    pub inflow_velocity: f64,  // Velocity at the inflow boundary
    pub outflow_velocity: f64, // Velocity at the outflow boundary (if required)
}

impl FlowBoundary {
    /// Apply the flow boundary condition to faces at the inflow and outflow boundaries.
    pub fn apply(&self, mesh: &mut Mesh) {
        // Identify the boundary faces based on inflow/outflow conditions
        let inflow_faces = self.get_boundary_faces(mesh, BoundaryType::Inflow);
        let outflow_faces = self.get_boundary_faces(mesh, BoundaryType::Outflow);
    
        // Apply inflow velocity to inflow boundary faces
        for face_index in inflow_faces {
            let inflow_velocity = self.inflow_velocity;
    
            // Borrow the face to modify its velocity
            let face_id;
            {
                let face = &mut mesh.faces[face_index];
                face.velocity.0 = inflow_velocity;  // Apply inflow velocity to the face
                face_id = face.id; // Extract necessary information from the face
            }
            
            // Now, we can safely apply inflow to connected elements using the face ID
            self.apply_inflow_to_elements(face_id, mesh); // Apply inflow to connected elements
        }
    
        // Apply outflow velocity to outflow boundary faces
        for face_index in outflow_faces {
            let outflow_velocity = self.outflow_velocity;
    
            // Borrow the face to modify its velocity
            let face_id;
            {
                let face = &mut mesh.faces[face_index];
                face.velocity.0 = outflow_velocity;  // Apply outflow velocity to the face
                face_id = face.id; // Extract necessary information from the face
            }
    
            // Now, we can safely apply outflow to connected elements using the face ID
            self.apply_outflow_to_elements(face_id, mesh); // Apply outflow to connected elements
        }
    }

    /// Helper function to apply inflow to the connected elements of a face by face ID
    fn apply_inflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        // Find the elements connected to this face using the face-element relationship table
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                // Apply inflow to the left element (if it exists in the mesh)
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.left_element_id) {
                    left_element.velocity.0 = self.inflow_velocity;  // Apply inflow velocity to the left element
                }

                // Apply inflow to the right element (if it exists in the mesh)
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.right_element_id) {
                    right_element.velocity.0 = self.inflow_velocity;  // Apply inflow velocity to the right element
                }

                // Break after finding the matching face relation
                break;
            }
        }
    }

    /// Helper function to apply outflow to the connected elements of a face by face ID
    fn apply_outflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        // Find the elements connected to this face using the face-element relationship table
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                // Apply outflow to the left element (if it exists in the mesh)
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.left_element_id) {
                    left_element.velocity.0 = self.outflow_velocity;  // Apply outflow velocity to the left element
                }

                // Apply outflow to the right element (if it exists in the mesh)
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.right_element_id) {
                    right_element.velocity.0 = self.outflow_velocity;  // Apply outflow velocity to the right element
                }

                // Break after finding the matching face relation
                break;
            }
        }
    }

    /// Helper function to get boundary faces of a specific type (inflow or outflow)
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

    /// Determine if a face is an inflow boundary
    pub fn is_inflow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes.0 as usize];
        let node_2 = &mesh.nodes[face.nodes.1 as usize];
        // Custom condition to identify inflow (based on node positions or other factors)
        node_1.position.0 == 0.0 || node_2.position.0 == 0.0
    }

    /// Determine if a face is an outflow boundary
    pub fn is_outflow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes.0 as usize];
        let node_2 = &mesh.nodes[face.nodes.1 as usize];
        // Custom condition to identify outflow (e.g., position at domain exit)
        node_1.position.0 == mesh.domain_width() || node_2.position.0 == mesh.domain_width()
    }

    /// Check mass conservation to ensure that inflow and outflow conditions are consistent
    pub fn check_mass_conservation(&self, flow_field: &FlowField) -> bool {
        let total_mass: f64 = flow_field.elements.iter().map(|e| e.calculate_mass()).sum();
        let mass_difference = total_mass - flow_field.initial_mass;
        mass_difference.abs() < MASS_CONSERVATION_THRESHOLD
    }
}

pub struct Inflow {
    pub rate: f64, // Rate at which mass and momentum are added to the system
}

impl Inflow {
    /// Apply the inflow boundary condition by adding mass and momentum to an element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass += self.rate * dt;  // Add mass over time
        element.momentum += self.rate * element.pressure * dt;  // Add momentum
    }
}

pub struct Outflow {
    pub rate: f64, // Rate at which mass and momentum are removed from the system
}

impl Outflow {
    /// Apply the outflow boundary condition by removing mass and momentum from an element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass = (element.mass - self.rate * dt).max(0.0);  // Prevent negative mass
        element.momentum = (element.momentum - self.rate * element.pressure * dt).max(0.0);  // Prevent negative momentum
    }
}
