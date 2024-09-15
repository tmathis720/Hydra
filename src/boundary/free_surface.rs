use crate::domain::{Mesh, Element, Face};
use crate::domain::FlowField;  // Correct module import for FlowField
use crate::boundary::BoundaryType;
use nalgebra::Vector3;

const MASS_CONSERVATION_THRESHOLD: f64 = 1e-6;  // Threshold for mass conservation check

/// FreeSurfaceBoundary structure applies flow conditions like free surface elevations and velocities
pub struct FreeSurfaceBoundaryCondition {
    pub inflow_velocity: Vector3<f64>,  // 3D inflow velocity
    pub outflow_velocity: Vector3<f64>, // 3D outflow velocity (if required)
    pub surface_elevation: f64,         // Free surface elevation at boundary
}

impl FreeSurfaceBoundaryCondition {
    /// Apply free surface boundary conditions to inflow and outflow boundary faces
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

    /// Apply inflow condition to elements connected to a face
    fn apply_inflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.connected_elements[0]) {
                    left_element.velocity = self.inflow_velocity;
                    left_element.momentum += self.inflow_velocity * left_element.mass;
                    left_element.height = self.surface_elevation;  // Update free surface elevation
                }
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.connected_elements[1]) {
                    right_element.velocity = self.inflow_velocity;
                    right_element.momentum += self.inflow_velocity * right_element.mass;
                    right_element.height = self.surface_elevation;  // Update free surface elevation
                }
                break;
            }
        }
    }

    /// Apply outflow condition to elements connected to a face
    fn apply_outflow_to_elements(&self, face_id: u32, mesh: &mut Mesh) {
        for relation in &mesh.face_element_relations {
            if relation.face_id == face_id {
                if let Some(left_element) = mesh.elements.iter_mut().find(|e| e.id == relation.connected_elements[0]) {
                    left_element.velocity = self.outflow_velocity;
                    left_element.momentum -= self.outflow_velocity * left_element.mass;
                    left_element.height = self.surface_elevation;
                }
                if let Some(right_element) = mesh.elements.iter_mut().find(|e| e.id == relation.connected_elements[1]) {
                    right_element.velocity = self.outflow_velocity;
                    right_element.momentum -= self.outflow_velocity * right_element.mass;
                    right_element.height = self.surface_elevation;
                }
                break;
            }
        }
    }

    /// Helper function to retrieve inflow or outflow boundary faces
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
        node_1.position.x == 0.0 || node_2.position.x == 0.0  // Check inflow at x = 0
    }

    /// Check if the face is at the outflow boundary
    fn is_outflow_boundary_face(&self, face: &Face, mesh: &Mesh) -> bool {
        let node_1 = &mesh.nodes[face.nodes[0]];
        let node_2 = &mesh.nodes[face.nodes[1]];
        node_1.position.x == mesh.domain_width() || node_2.position.x == mesh.domain_width()  // Check outflow at domain's right edge
    }

    /// Check for mass conservation by comparing the initial and current total mass
    pub fn check_mass_conservation(&self, flow_field: &FlowField) -> bool {
        let total_mass: f64 = flow_field.elements.iter().map(|e| e.mass).sum();
        let mass_difference = total_mass - flow_field.initial_mass;
        mass_difference.abs() < MASS_CONSERVATION_THRESHOLD
    }
}

/// Struct to represent free surface inflow boundary conditions
pub struct FreeSurfaceInflow {
    pub rate: f64,  // Rate of mass/momentum added to the system
}

impl FreeSurfaceInflow {
    /// Apply free surface inflow by adding mass and momentum to the element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass += self.rate * dt;  // Add mass
        element.momentum += Vector3::new(self.rate * element.pressure * dt, 0.0, 0.0);  // Add momentum (assumed in x-direction)
        // Optionally, modify surface elevation if necessary for free surface
    }
}

/// Struct to represent free surface outflow boundary conditions
pub struct FreeSurfaceOutflow {
    pub rate: f64,  // Rate of mass/momentum removed from the system
}

impl FreeSurfaceOutflow {
    /// Apply free surface outflow by removing mass and momentum from the element
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        element.mass = (element.mass - self.rate * dt).max(0.0);  // Ensure mass does not go negative
        element.momentum.x = (element.momentum.x - self.rate * element.pressure * dt).max(0.0);  // Prevent negative momentum
        // Optionally, adjust surface elevation at the outflow boundary
    }
}

// Unit tests for FreeSurfaceBoundaryCondition

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryManager;
    use crate::domain::{Element, Face, FaceElementRelation, FlowField, Mesh, Node};
    use nalgebra::Vector3;

    #[test]
    fn test_apply_free_surface_boundary() {
        // Create nodes for the mesh
        let nodes = vec![
            Node {
                id: 0,
                position: Vector3::new(0.0, 0.0, 0.0), // Inflow boundary at x = 0.0
            },
            Node {
                id: 1,
                position: Vector3::new(1.0, 0.0, 0.0),
            },
            Node {
                id: 2,
                position: Vector3::new(2.0, 0.0, 0.0), // Outflow boundary at x = 2.0
            },
        ];

        // Create elements connected to the nodes
        let elements = vec![
            Element {
                id: 0,
                nodes: vec![0, 1],
                mass: 1.0,
                ..Default::default()
            },
            Element {
                id: 1,
                nodes: vec![1, 2],
                mass: 1.0,
                ..Default::default()
            },
        ];

        // Create faces between nodes
        let faces = vec![
            Face {
                id: 0,
                nodes: vec![0, 1],
                area: 1.0,
                ..Default::default()
            },
            Face {
                id: 1,
                nodes: vec![1, 2],
                area: 1.0,
                ..Default::default()
            },
        ];

        // Define face-element relationships
        let face_element_relations = vec![
            FaceElementRelation {
                face_id: 0,
                connected_elements: vec![0, 1],
            },
            FaceElementRelation {
                face_id: 1,
                connected_elements: vec![1, 2],
            },
        ];

        // Create the mesh
        let mut mesh = Mesh {
            elements: elements.clone(),
            nodes,
            faces,
            face_element_relations,
            ..Mesh::default()
        };

        // Create the FreeSurfaceBoundary instance
        let free_surface_boundary = FreeSurfaceBoundaryCondition {
            inflow_velocity: Vector3::new(1.0, 0.0, 0.0),  // Inflow in x-direction
            outflow_velocity: Vector3::new(1.0, 0.0, 0.0), // Same velocity at outflow for this test
            surface_elevation: 2.0,                        // Set surface elevation to 2.0
        };

        // Apply the boundary conditions
        free_surface_boundary.apply(&mut mesh);

        // Assertions to verify that the boundary conditions were applied correctly

        // Check inflow face
        let inflow_face = &mesh.faces[0];
        assert_eq!(inflow_face.velocity, free_surface_boundary.inflow_velocity);

        // Check outflow face
        let outflow_face = &mesh.faces[1];
        assert_eq!(outflow_face.velocity, free_surface_boundary.outflow_velocity);

        // Check that elements connected to inflow face have updated properties
        let inflow_element = &mesh.elements[0];
        assert_eq!(inflow_element.velocity, free_surface_boundary.inflow_velocity);
        assert_eq!(inflow_element.height, free_surface_boundary.surface_elevation);
        assert_eq!(
            inflow_element.momentum,
            free_surface_boundary.inflow_velocity * inflow_element.mass
        );

        // Check that elements connected to outflow face have updated properties
        let outflow_element = &mesh.elements[1];
        assert_eq!(outflow_element.velocity, free_surface_boundary.outflow_velocity);
        assert_eq!(outflow_element.height, free_surface_boundary.surface_elevation);
        // Since outflow reduces momentum, we expect it to decrease
        assert_eq!(
            outflow_element.momentum,
            -free_surface_boundary.outflow_velocity * outflow_element.mass
        );
    }

    #[test]
    fn test_mass_conservation() {
        // Create elements with initial mass
        let elements = vec![
            Element {
                id: 0,
                mass: 1.0,
                ..Default::default()
            },
            Element {
                id: 1,
                mass: 1.0,
                ..Default::default()
            },
        ];

        // Create a BoundaryManager (empty for this test)
        let boundary_manager = BoundaryManager::new();

        // Initialize flow field with the elements and boundary manager
        let flow_field = FlowField::new(elements.clone(), boundary_manager);

        // Create the FreeSurfaceBoundary instance
        let free_surface_boundary = FreeSurfaceBoundaryCondition {
            inflow_velocity: Vector3::new(1.0, 0.0, 0.0),
            outflow_velocity: Vector3::new(1.0, 0.0, 0.0),
            surface_elevation: 2.0,
        };

        // Since no mass is added or removed in this simplified test, mass should be conserved
        let mass_conserved = free_surface_boundary.check_mass_conservation(&flow_field);
        assert!(mass_conserved, "Mass should be conserved");
    }
}

