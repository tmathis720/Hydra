use crate::domain::{Mesh, Face, FlowField};
use crate::boundary::{BoundaryCondition, BoundaryType};
use nalgebra::Vector3;
use std::cell::RefCell;

/// FreeSurfaceBoundaryCondition applies flow conditions like free surface elevations and velocities.
pub struct FreeSurfaceBoundaryCondition {
    pub surface_elevation: RefCell<Vec<f64>>,  // eta values
    pub velocity_potential: RefCell<Vec<f64>>, // phi values
    pub shear_vector: RefCell<Vec<f64>>,       // tau values
    pub mass_flux: f64,               // mass flux, currently unused
}

impl FreeSurfaceBoundaryCondition {
    /// Creates a new instance of the free surface boundary condition.
    pub fn new(
        surface_elevation: Vec<f64>,
        velocity_potential: Vec<f64>,
        shear_vector: Vec<f64>,
        mass_flux: f64,
    ) -> Self {
        FreeSurfaceBoundaryCondition {
            surface_elevation: RefCell::new(surface_elevation),
            velocity_potential: RefCell::new(velocity_potential),
            shear_vector: RefCell::new(shear_vector),
            mass_flux,
        }
    }

    /// Adds boundary faces related to the free surface.
    pub fn add_boundary_faces(&self, mesh: &mut Mesh) {
        let surface_face_ids: Vec<u32> = mesh.faces
            .iter()
            .filter(|face| mesh.is_surface_face(face))
            .map(|face| face.id)
            .collect();

        for face_id in surface_face_ids {
            mesh.mark_face_as_boundary(face_id, BoundaryType::FreeSurface);
        }
    }

    /// Applies the kinematic boundary condition for the free surface.
    ///
    /// The kinematic condition is typically:
    /// d_eta/dt = w - u * d(eta)/dx
    ///
    /// This ensures that the vertical velocity of the surface matches the material velocity.
    pub fn apply_kinematic_condition(&self, mesh: &Mesh, flow_field: &FlowField) {
        for (i, face_id) in flow_field.boundary_manager.get_free_surface_faces(mesh).iter().enumerate() {
            if let Some(face) = mesh.get_face_by_id(*face_id) {
                let u = face.velocity.x;  // Horizontal velocity at the face
                let w = face.velocity.z;  // Vertical velocity at the face
    
                println!("Applying kinematic condition:");
                println!("u (horizontal velocity): {}", u);
                println!("w (vertical velocity): {}", w);
    
                if i < self.surface_elevation.borrow().len() - 1 {
                    if let Some(next_face_id) = flow_field.boundary_manager.get_free_surface_faces(mesh).get(i + 1) {
                        if let Some(next_face) = mesh.get_face_by_id(*next_face_id) {
                            let dx = self.compute_dx_between_faces(face, next_face, mesh);
                            let d_eta_dx = (self.surface_elevation.borrow()[i + 1] - self.surface_elevation.borrow()[i]) / dx;
    
                            println!("dx: {}", dx);
                            println!("d_eta_dx: {}", d_eta_dx);
    
                            self.surface_elevation.borrow_mut()[i] += w - u * d_eta_dx;
                            println!("Updated surface elevation: {}", self.surface_elevation.borrow()[i]);
                        }
                    }
                }
            }
        }
    }
    

    /// Helper function to compute the distance (dx) between the first nodes of two faces.
    fn compute_dx_between_faces(&self, face1: &Face, face2: &Face, mesh: &Mesh) -> f64 {
        let node1_idx = face1.nodes[0]; // First node of face1
        let node2_idx = face2.nodes[0]; // First node of face2

        let node_pos1 = mesh.nodes[node1_idx].position; // Get position of the first node of face1
        let node_pos2 = mesh.nodes[node2_idx].position; // Get position of the first node of face2

        // Compute the Euclidean distance between node1 and node2
        (node_pos2 - node_pos1).norm()
    }

    /// Applies the dynamic boundary condition for the free surface.
    pub fn apply_dynamic_condition(&self, mesh: &Mesh, flow_field: &FlowField, wind_speed: f64) {
        let g = 9.81; // Gravitational constant
        let rho_air = 1.225; // Air density (kg/m^3)
        let drag_coefficient = 0.002; // Example drag coefficient for wind shear
    
        for (i, face_id) in flow_field.boundary_manager.get_free_surface_faces(mesh).iter().enumerate() {
            if let Some(face) = mesh.get_face_by_id(*face_id) {
                let u = face.velocity.x;
                let v = face.velocity.y;
    
                println!("Face velocity x: {}", u);
                println!("Face velocity y: {}", v);
    
                let wind_shear_effect = rho_air * drag_coefficient * wind_speed.powi(2);
                println!("Wind shear effect: {}", wind_shear_effect);
    
                let expected_phi = -g * self.surface_elevation.borrow()[i] + 0.5 * (u.powi(2) + v.powi(2)) + wind_shear_effect;
    
                self.velocity_potential.borrow_mut()[i] -= expected_phi;
    
                println!("Updated velocity potential: {}", self.velocity_potential.borrow()[i]);
            }
        }
    }

    /// Applies shear forces (e.g., wind shear) on the surface.
    pub fn apply_surface_shear(&self, wind_speed: f64) {
        let rho_air = 1.225;  // Air density (kg/m^3)
        let drag_coefficient = 0.002;  // Example drag coefficient for wind
    
        let mut shear_vector = self.shear_vector.borrow_mut();
        
        for i in 0..shear_vector.len() {
            shear_vector[i] = rho_air * drag_coefficient * wind_speed.powi(2);
        }
    }
}

impl BoundaryCondition for FreeSurfaceBoundaryCondition {
    /// Updates the inflow boundary condition based on time or other factors.
    fn update(&mut self, _time: f64) {
        // Add any time-dependent updates here if needed.
    }

    /// Applies the free surface boundary condition to the surface faces in the flow field.
    fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, _time_step: f64) {
        self.apply_kinematic_condition(mesh, flow_field);
        
        let wind_speed = 10.0; // Placeholder wind speed value
        self.apply_dynamic_condition(mesh, flow_field, wind_speed);
        
        self.apply_surface_shear(wind_speed);
    }

    /// Retrieves the velocity associated with the free surface boundary condition.
    fn velocity(&self) -> Option<Vector3<f64>> {
        Some(Vector3::new(0.0, 0.0, 0.0)) // Placeholder: Replace with actual velocity calculation.
    }

    /// Retrieves the mass rate associated with the free surface boundary condition.
    fn mass_rate(&self) -> Option<f64> {
        Some(self.mass_flux)
    }

    /// Retrieves the boundary elements affected by the free surface boundary condition.
    fn get_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        mesh.get_free_surface_faces()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Mesh, Node, Face, FlowField};
    use crate::boundary::BoundaryManager;
    use nalgebra::Vector3;

    #[test]
    fn test_add_boundary_faces() {
        // Create mock faces and mesh
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(0.0, 0.0, 0.0), 1.0, Vector3::new(0.0, 1.0, 0.0), None),
            Face::new(2, vec![2, 3], Vector3::new(0.0, 0.0, 1.0), 1.0, Vector3::new(0.0, 0.0, 1.0), None),
        ];

        // Mock nodes to ensure the node indices make sense
        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 0.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 0.0, 0.0)),
            Node::new(2, Vector3::new(0.0, 1.0, 0.0)),
            Node::new(3, Vector3::new(1.0, 1.0, 0.0)),
        ];

        let mut mesh = Mesh {
            faces,
            nodes,
            elements: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        };

        // Create a free surface boundary condition
        let free_surface = FreeSurfaceBoundaryCondition::new(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0], 0.0);

        // Add boundary faces
        free_surface.add_boundary_faces(&mut mesh);

        // Ensure the faces have been marked correctly
        assert!(mesh.faces[1].is_boundary_face(), "Face should be marked as a boundary face.");
        assert_eq!(mesh.faces[1].boundary_type, Some(BoundaryType::FreeSurface));
    }

    #[test]
    fn test_kinematic_condition() {
        // Mock flow field and boundary condition
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(0.1, 0.0, 0.2), 1.0, Vector3::new(0.0, 1.0, 0.0), None),
        ];

        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 0.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 0.0, 0.0)),
        ];

        let boundary_manager = BoundaryManager::new();
        let mesh = Mesh::new(vec![], nodes, faces).unwrap();
        let flow_field = FlowField::new(vec![], boundary_manager);

        // Initialize surface elevation with a non-zero value
        let free_surface = FreeSurfaceBoundaryCondition::new(vec![0.1], vec![0.0], vec![0.0], 0.0);

        free_surface.apply_kinematic_condition(&mesh, &flow_field);

        {
            let surface_elevation = free_surface.surface_elevation.borrow();
            println!("Updated surface elevation: {}", surface_elevation[0]);

            assert!((surface_elevation[0] - 0.3).abs() < 1e-6, "Surface elevation should be updated by kinematic condition.");
        }
    }



    #[test]
    fn test_dynamic_condition() {
        // Mock flow field and boundary condition
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(0.1, 0.1, 0.2), 1.0, Vector3::new(0.0, 1.0, 0.0), None),
        ];
        let boundary_manager = BoundaryManager::new();
        let mesh = Mesh::new(vec![], vec![], faces).unwrap();
        let flow_field = FlowField::new(vec![], boundary_manager);

        // Initialize velocity potential with a non-zero value
        let free_surface = FreeSurfaceBoundaryCondition::new(vec![1.0], vec![0.5], vec![0.0], 0.0);

        free_surface.apply_dynamic_condition(&mesh, &flow_field, 5.0); // Mock wind speed of 5 m/s

        {
            // Calculate expected velocity potential
            let u: f64 = 0.1;
            let v: f64 = 0.1;
            let g: f64 = 9.81;
            let rho_air: f64 = 1.225;
            let drag_coefficient: f64 = 0.002;
            let wind_speed: f64 = 5.0;

            let wind_shear_effect = rho_air * drag_coefficient * wind_speed.powi(2);
            let expected_phi = -g * 1.0 - 0.5 * (u.powi(2) + v.powi(2)) + wind_shear_effect;

            let actual_phi = free_surface.velocity_potential.borrow()[0];

            println!("Expected phi: {}", expected_phi);
            println!("Actual phi: {}", actual_phi);

            assert!((actual_phi - expected_phi).abs() < 1e-6, "Velocity potential should be updated by dynamic condition.");
        }
    }



    #[test]
    fn test_apply_surface_shear() {
        let free_surface = FreeSurfaceBoundaryCondition::new(vec![0.0], vec![0.0], vec![0.0], 0.0);

        // Apply surface shear with mock wind speed
        free_surface.apply_surface_shear(10.0); // Wind speed of 10 m/s

        {
            // Borrow the shear vector immutably only after the mutation is complete
            let shear_vector = free_surface.shear_vector.borrow();
            let shear_value = shear_vector[0];
            let expected_shear = 1.225 * 0.002 * 10.0f64.powi(2);

            assert!((shear_value - expected_shear).abs() < 1e-6, "Shear stress should be updated by wind shear.");
        }
    }
}
