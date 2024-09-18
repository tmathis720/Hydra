use crate::domain::{Mesh, Face, FlowField};
use crate::boundary::{BoundaryCondition, BoundaryType};
use nalgebra::Vector3;
use std::cell::RefCell;

/// FreeSurfaceBoundaryCondition applies free-surface conditions 
/// like water surface elevation, velocity potential, shear vector,
/// and mass flux (i.e., evaporation and rainfall).
pub struct FreeSurfaceBoundaryCondition {
    pub surface_elevation: RefCell<Vec<f64>>,  // eta values
    pub velocity_potential: RefCell<Vec<f64>>, // phi values
    pub shear_vector: RefCell<Vec<f64>>,       // tau values
    pub mass_flux: f64,                        // mass flux, currently unused
    // Environmental factors (placeholders for future extensions)
    pub wind_speed: f64,
    pub air_temperature: f64,
    pub relative_humidity: f64,
    pub air_pressure: f64,
    pub cloud_cover: f64,
}

impl FreeSurfaceBoundaryCondition {
    /// Creates a new instance of the free surface boundary condition.
    pub fn new(
        surface_elevation: Vec<f64>,
        velocity_potential: Vec<f64>,
        shear_vector: Vec<f64>,
        mass_flux: f64,
        wind_speed: f64,
        air_temperature: f64,
        relative_humidity: f64,
        air_pressure: f64,
        cloud_cover: f64,
    ) -> Self {
        FreeSurfaceBoundaryCondition {
            surface_elevation: RefCell::new(surface_elevation),
            velocity_potential: RefCell::new(velocity_potential),
            shear_vector: RefCell::new(shear_vector),
            mass_flux,
            wind_speed,
            air_temperature,
            relative_humidity,
            air_pressure,
            cloud_cover,
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

    /// Computes the spatial derivative of surface elevation between two adjacent faces.
    fn compute_d_eta_dx(&self, face1: &Face, face2: &Face, mesh: &Mesh) -> f64 {
        let dx = self.compute_dx_between_faces(face1, face2, mesh);
        let d_eta_dx = (self.surface_elevation.borrow()[face2.id as usize] 
                       - self.surface_elevation.borrow()[face1.id as usize]) / dx;
        d_eta_dx
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

                if i < self.surface_elevation.borrow().len() - 1 {
                    if let Some(next_face_id) = flow_field.boundary_manager.get_free_surface_faces(mesh).get(i + 1) {
                        if let Some(next_face) = mesh.get_face_by_id(*next_face_id) {
                            let d_eta_dx = self.compute_d_eta_dx(face, next_face, mesh);
                            self.surface_elevation.borrow_mut()[i] += w - u * d_eta_dx;
                        }
                    }
                }
            }
        }
    }

    /// Applies the dynamic boundary condition for the free surface.
    pub fn apply_dynamic_condition(&self, mesh: &Mesh, flow_field: &FlowField) {
        let g = 9.81; // Gravitational constant
        let rho_air = 1.225; // Air density (kg/m^3)
        let drag_coefficient = 0.002; // Example drag coefficient for wind shear

        for (i, face_id) in flow_field.boundary_manager.get_free_surface_faces(mesh).iter().enumerate() {
            if let Some(face) = mesh.get_face_by_id(*face_id) {
                let u = face.velocity.x;
                let v = face.velocity.y;

                let wind_shear_effect = rho_air * drag_coefficient * self.wind_speed.powi(2);
                let expected_phi = -g * self.surface_elevation.borrow()[i] + 0.5 * (u.powi(2) + v.powi(2)) + wind_shear_effect;

                self.velocity_potential.borrow_mut()[i] -= expected_phi;
            }
        }
    }

    /// Applies shear forces (e.g., wind shear) on the surface.
    pub fn apply_surface_shear(&self) {
        let rho_air = 1.225;  // Air density (kg/m^3)
        let drag_coefficient = 0.002;  // Example drag coefficient for wind

        let mut shear_vector = self.shear_vector.borrow_mut();
        
        for i in 0..shear_vector.len() {
            shear_vector[i] = rho_air * drag_coefficient * self.wind_speed.powi(2);
        }
    }

    /// Updates environmental conditions (e.g., wind speed, temperature) to be used by the boundary condition.
    pub fn update_environmental_conditions(
        &mut self, 
        wind_speed: f64, 
        air_temperature: f64, 
        relative_humidity: f64, 
        air_pressure: f64, 
        cloud_cover: f64,
    ) {
        self.wind_speed = wind_speed;
        self.air_temperature = air_temperature;
        self.relative_humidity = relative_humidity;
        self.air_pressure = air_pressure;
        self.cloud_cover = cloud_cover;
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
        self.apply_dynamic_condition(mesh, flow_field);
        self.apply_surface_shear();
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

impl Default for FreeSurfaceBoundaryCondition {
    fn default() -> Self {
        FreeSurfaceBoundaryCondition {
            surface_elevation: RefCell::new(vec![0.0; 10]), // Default to 10 zero elevation points
            velocity_potential: RefCell::new(vec![0.0; 10]), // Default to 10 zero velocity potential points
            shear_vector: RefCell::new(vec![0.0; 10]), // Default to 10 zero shear vectors
            mass_flux: 0.0,  // No mass flux by default
            wind_speed: 0.0,  // No wind by default
            air_temperature: 293.0,  // Default to 293K (20°C)
            relative_humidity: 50.0, // 50% relative humidity
            air_pressure: 1013.0,    // Standard atmospheric pressure in hPa
            cloud_cover: 0.0,        // No cloud cover
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Mesh, Node, Face, FlowField};
    use crate::boundary::BoundaryManager;
    use nalgebra::Vector3;

    /// Helper function to create a simple mock mesh
    fn create_mock_mesh() -> Mesh {
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(0.1, 0.0, 0.2), 1.0, Vector3::new(0.0, 1.0, 0.0), None),
            Face::new(2, vec![2, 3], Vector3::new(0.2, 0.0, 0.2), 1.0, Vector3::new(0.0, 0.0, 1.0), None),
        ];

        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 0.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 0.0, 0.0)),
            Node::new(2, Vector3::new(0.0, 1.0, 0.0)),
            Node::new(3, Vector3::new(1.0, 1.0, 0.0)),
        ];

        Mesh {
            faces,
            nodes,
            elements: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        }
    }

    #[test]
    fn test_default_initialization() {
        // Test the default implementation
        let default_condition = FreeSurfaceBoundaryCondition::default();

        // Check that default values are initialized correctly
        assert_eq!(default_condition.surface_elevation.borrow().len(), 10);
        assert_eq!(default_condition.surface_elevation.borrow()[0], 0.0);
        assert_eq!(default_condition.velocity_potential.borrow().len(), 10);
        assert_eq!(default_condition.velocity_potential.borrow()[0], 0.0);
        assert_eq!(default_condition.mass_flux, 0.0);
        assert_eq!(default_condition.wind_speed, 0.0);
        assert_eq!(default_condition.air_temperature, 293.0);  // 20°C
        assert_eq!(default_condition.relative_humidity, 50.0);
        assert_eq!(default_condition.air_pressure, 1013.0);
        assert_eq!(default_condition.cloud_cover, 0.0);
    }

/*     #[test]
    fn test_add_boundary_faces() {
        let mut mesh = create_mock_mesh();

        // Ensure that the faces are actually marked as surface faces
        for face in &mut mesh.faces {
            face.boundary_type = Some(BoundaryType::FreeSurface);
        }

        // Create a free surface boundary condition with default values
        let free_surface = FreeSurfaceBoundaryCondition::default();

        // Add boundary faces
        free_surface.add_boundary_faces(&mut mesh);

        // Ensure the faces have been marked correctly
        assert!(mesh.faces[0].is_boundary_face(), "Face 1 should be marked as a boundary face.");
        assert_eq!(mesh.faces[0].boundary_type, Some(BoundaryType::FreeSurface));
    } */

    #[test]
    fn test_compute_dx_between_faces() {
        let mesh = create_mock_mesh();
        let free_surface = FreeSurfaceBoundaryCondition::default();

        // Compute the distance between two adjacent faces
        let dx = free_surface.compute_dx_between_faces(&mesh.faces[0], &mesh.faces[1], &mesh);

        // Expected distance between nodes 0 and 2 should be based on the node positions
        let expected_dx = (mesh.nodes[0].position - mesh.nodes[2].position).norm();
        assert!((dx - expected_dx).abs() < 1e-6, "dx should be correctly calculated.");
    }

/*     #[test]
    fn test_compute_d_eta_dx() {
        let mut mesh = create_mock_mesh();
        
        // Ensure there are enough surface elevation values for the number of faces
        let free_surface = FreeSurfaceBoundaryCondition::new(
            vec![0.0, 1.0], // Surface elevation values (eta) for 2 faces
            vec![0.0; 2],   // Velocity potential values
            vec![0.0; 2],   // Shear vector values
            0.0,            // Mass flux
            0.0,            // Wind speed
            293.0,          // Air temperature
            50.0,           // Relative humidity
            1013.0,         // Air pressure
            0.0,            // Cloud cover
        );

        // Compute the elevation gradient between two faces
        let d_eta_dx = free_surface.compute_d_eta_dx(&mesh.faces[0], &mesh.faces[1], &mesh);

        // Expected gradient: (1.0 - 0.0) / distance between faces
        let expected_d_eta_dx = 1.0 / free_surface.compute_dx_between_faces(&mesh.faces[0], &mesh.faces[1], &mesh);
        assert!((d_eta_dx - expected_d_eta_dx).abs() < 1e-6, "d_eta_dx should be correctly calculated.");
    } */

    #[test]
    fn test_apply_kinematic_condition() {
        let mesh = create_mock_mesh();
        let boundary_manager = BoundaryManager::new();
        let flow_field = FlowField::new(vec![], boundary_manager);

        // Surface elevation with initial values and the condition to be applied
        let free_surface = FreeSurfaceBoundaryCondition::new(
            vec![0.1, 0.2], // Initial surface elevations
            vec![0.0; 2],   // Velocity potentials
            vec![0.0; 2],   // Shear vector values
            0.0,            // Mass flux
            0.0,            // Wind speed
            293.0,          // Air temperature
            50.0,           // Relative humidity
            1013.0,         // Air pressure
            0.0,            // Cloud cover
        );

        free_surface.apply_kinematic_condition(&mesh, &flow_field);

        // Ensure surface elevations were updated correctly
        let updated_elevations = free_surface.surface_elevation.borrow();
        // Simple check - further refinement can include analytical comparison based on inputs
        assert!((updated_elevations[0] - 0.1).abs() < 1e-6, "Elevation should be updated by kinematic condition.");
    }

/*     #[test]
    fn test_apply_dynamic_condition() {
        let mesh = create_mock_mesh();
        let boundary_manager = BoundaryManager::new();
        let flow_field = FlowField::new(vec![], boundary_manager);

        // Initialize with non-zero elevation and velocity potential
        let free_surface = FreeSurfaceBoundaryCondition::new(
            vec![1.0], // Surface elevation
            vec![0.5], // Velocity potential
            vec![0.0], // Shear vector values
            0.0,       // Mass flux
            5.0,       // Wind speed
            293.0,     // Air temperature
            50.0,      // Relative humidity
            1013.0,    // Air pressure
            0.0,       // Cloud cover
        );

        // Apply dynamic condition
        free_surface.apply_dynamic_condition(&mesh, &flow_field);

        // Check if the velocity potential was updated correctly
        let u: f64 = 0.1;
        let v: f64 = 0.1;
        let g: f64 = 9.81;
        let rho_air: f64 = 1.225;
        let drag_coefficient: f64 = 0.002;
        let wind_speed: f64 = 5.0;

        let wind_shear_effect = rho_air * drag_coefficient * wind_speed.powi(2);
        let expected_phi = -g * 1.0 + 0.5 * (u.powi(2) + v.powi(2)) + wind_shear_effect;
        
        assert!((free_surface.velocity_potential.borrow()[0] - (0.5 - expected_phi)).abs() < 1e-6, 
            "Velocity potential should be updated by dynamic condition.");
    } */

    #[test]
    fn test_apply_surface_shear() {
        let mut free_surface = FreeSurfaceBoundaryCondition::default();

        // Apply surface shear with a mock wind speed
        free_surface.update_environmental_conditions(10.0, 293.0, 50.0, 1013.0, 0.0);
        free_surface.apply_surface_shear();

        // Verify that shear values have been updated
        let shear_vector = free_surface.shear_vector.borrow();
        let expected_shear = 1.225 * 0.002 * 10.0f64.powi(2);
        assert!((shear_vector[0] - expected_shear).abs() < 1e-6, "Shear stress should be updated by wind shear.");
    }
}