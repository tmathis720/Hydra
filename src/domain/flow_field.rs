use crate::domain::{Element, Mesh};
use crate::boundary::BoundaryManager;
use nalgebra::Vector3;

/// The `FlowField` struct encapsulates the elements within a simulation domain
/// and is responsible for handling the physical state of the simulation (e.g., mass conservation, 
/// boundary condition management).
pub struct FlowField {
    /// A collection of elements within the simulation domain.
    pub elements: Vec<Element>,
    /// The initial mass of the system, used for mass conservation checks.
    pub initial_mass: f64,
    /// The `BoundaryManager` is responsible for managing and applying boundary conditions.
    pub boundary_manager: BoundaryManager,
}

impl FlowField {
    /// Creates a new `FlowField` from a given set of elements and a boundary manager.
    ///
    /// This method computes the initial mass of the system by summing the masses of all elements.
    /// 
    /// # Parameters
    /// - `elements`: A vector of `Element` instances representing the flow domain.
    /// - `boundary_manager`: The boundary condition manager.
    ///
    /// # Returns
    /// A new `FlowField` instance.
    pub fn new(elements: Vec<Element>, boundary_manager: BoundaryManager) -> Self {
        let initial_mass: f64 = elements.iter().map(|e| e.mass).sum();
        Self {
            elements,
            initial_mass,
            boundary_manager,
        }
    }

    /// Updates the boundary conditions using the `BoundaryManager` at a specific simulation time.
    ///
    /// This function ensures boundary conditions are applied correctly without borrow conflicts.
    /// 
    /// # Parameters
    /// - `mesh`: A mutable reference to the mesh.
    /// - `time`: The current time step of the simulation.
    pub fn update_boundary_conditions(&mut self, mesh: &mut Mesh, time: f64) {
        let boundary_manager = std::mem::take(&mut self.boundary_manager);
        boundary_manager.apply(mesh, self, time);
        self.boundary_manager = boundary_manager; // Restore after use
    }

    /// Computes and returns the mass of a specific element.
    ///
    /// # Parameters
    /// - `element`: A reference to the `Element` whose mass is being calculated.
    ///
    /// # Returns
    /// The mass of the specified element as an `f64`.
    pub fn compute_mass(&self, element: &Element) -> f64 {
        element.mass
    }

    /// Computes and returns the density of a given element.
    ///
    /// The density is calculated by dividing the mass by the area of the element.
    /// If the area is zero, `None` is returned.
    ///
    /// # Parameters
    /// - `element`: A reference to the `Element` whose density is being calculated.
    ///
    /// # Returns
    /// The density as an `Option<f64>`, or `None` if the area is zero.
    pub fn compute_density(&self, element: &Element) -> Option<f64> {
        if element.area > 0.0 {
            Some(element.mass / element.area)
        } else {
            None
        }
    }

    /// Checks for mass conservation by comparing the current mass with the initial mass.
    ///
    /// This function calculates the total mass in the system and compares it to the initial mass.
    /// If the relative difference exceeds a tolerance of 1e-6, an error is returned.
    ///
    /// # Returns
    /// A `Result` that indicates if the mass conservation check passed or failed.
    pub fn check_mass_conservation(&self) -> Result<(), String> {
        let current_mass: f64 = self.elements.iter().map(|e| e.mass).sum();
        let relative_difference = ((current_mass - self.initial_mass) / self.initial_mass).abs();
        let tolerance = 1e-6;

        if relative_difference < tolerance {
            Ok(())
        } else {
            Err(format!(
                "Mass conservation failed: Initial mass = {}, Current mass = {}, Relative difference = {}",
                self.initial_mass, current_mass, relative_difference
            ))
        }
    }

    /// Retrieves the velocity vector of a specific element by its ID.
    ///
    /// # Parameters
    /// - `element_id`: The ID of the element.
    ///
    /// # Returns
    /// An `Option` containing the velocity vector, or `None` if the element is not found.
    pub fn get_velocity(&self, element_id: u32) -> Option<Vector3<f64>> {
        self.elements.iter().find(|e| e.id == element_id).map(|e| e.velocity)
    }

    /// Retrieves the pressure value of a specific element by its ID.
    ///
    /// # Parameters
    /// - `element_id`: The ID of the element.
    ///
    /// # Returns
    /// An `Option` containing the pressure value, or `None` if the element is not found.
    pub fn get_pressure(&self, element_id: u32) -> Option<f64> {
        self.elements.iter().find(|e| e.id == element_id).map(|e| e.pressure)
    }

    /// Computes the pressure gradient for a given element.
    ///
    /// This function calculates the pressure gradient using a central difference
    /// approximation between the given element and its neighboring elements.
    ///
    /// # Parameters
    /// - `element`: The reference to the element for which to compute the pressure gradient.
    ///
    /// # Returns
    /// A `Vector3<f64>` representing the pressure gradient at the element.
    pub fn compute_pressure_gradient(&self, element: &Element) -> Vector3<f64> {
        // Ensure the element has neighbors
        if element.neighbor_refs.is_empty() {
            return Vector3::zeros(); // No neighbors, no gradient
        }

        let mut gradient = Vector3::zeros();

        // Iterate over neighbor_refs and compute contributions to the pressure gradient
        for (i, &neighbor_id) in element.neighbor_refs.iter().enumerate() {
            if let Some(neighbor) = self.elements.iter().find(|e| e.id == neighbor_id.try_into().unwrap()) {
                let delta_pressure = neighbor.pressure - element.pressure;
                let distance = element.neighbor_distance[i];  // Use precomputed neighbor distance

                // Convert centroid coordinates into nalgebra vectors for vector subtraction
                let delta_position = Vector3::from_column_slice(&neighbor.centroid_coordinates)
                    - Vector3::from_column_slice(&element.centroid_coordinates);

                // Add the contribution to the pressure gradient
                gradient += (delta_pressure / distance) * delta_position.normalize();
            }
        }

        gradient
    }

    /// Applies boundary conditions using the `BoundaryManager` at a specific time step.
    ///
    /// This function wraps the boundary manager to ensure safe application without borrowing conflicts.
    ///
    /// # Parameters
    /// - `mesh`: A mutable reference to the mesh.
    /// - `time_step`: The current time step.
    pub fn apply_boundary_conditions(&mut self, mesh: &mut Mesh, time_step: f64) {
        let boundary_manager = std::mem::take(&mut self.boundary_manager);
        boundary_manager.apply(mesh, self, time_step);
        self.boundary_manager = boundary_manager; // Restore after use
    }

    /// Calculates the inflow mass rate based on the boundary conditions.
    pub fn get_inflow_mass_rate(&self, mesh: &mut Mesh) -> f64 {
        let inflow_elements = self.boundary_manager.get_inflow_elements(mesh);
        let velocity = self.get_inflow_velocity(mesh).unwrap_or(Vector3::zeros());
        let area = self.calculate_inflow_area(mesh, &inflow_elements);
        let density = self.calculate_inflow_density(&inflow_elements);
        density * velocity.magnitude() * area
    }

    /// Calculates the total inflow area by summing the areas of inflow elements.
    fn calculate_inflow_area(&self, mesh: &Mesh, inflow_elements: &[u32]) -> f64 {
        inflow_elements.iter().fold(0.0, |total_area, &element_id| {
            if let Some(element) = mesh.get_element_by_id(element_id) {
                total_area + element.area
            } else {
                total_area
            }
        })
    }

    /// Calculates the average inflow density based on the inflow elements.
    fn calculate_inflow_density(&self, inflow_elements: &[u32]) -> f64 {
        let total_density: f64 = inflow_elements.iter().fold(0.0, |total_density, &element_id| {
            if let Some(element) = self.elements.iter().find(|e| e.id == element_id) {
                total_density + element.compute_density().unwrap_or(0.0)
            } else {
                total_density
            }
        });
        total_density / inflow_elements.len() as f64
    }

    pub fn calculate_outflow_density(&self, outflow_elements: &[u32]) -> f64 {
        let total_density: f64 = outflow_elements.iter().fold(0.0, |total_density, &element_id| {
            if let Some(element) = self.elements.iter().find(|e| e.id == element_id) {
                total_density + element.compute_density().unwrap_or(0.0)
            } else {
                total_density
            }
        });

        // Average density across outflow elements
        total_density / outflow_elements.len() as f64
    }

    /// Retrieves the average inflow velocity at the inflow boundary.
    ///
    /// This function calculates the inflow velocity by averaging the velocities
    /// of all elements at the inflow boundary.
    ///
    /// # Parameters
    /// - `mesh`: A mutable reference to the `Mesh`.
    ///
    /// # Returns
    /// An `Option<Vector3<f64>>` containing the average inflow velocity, or `None` if no inflow elements are found.
    pub fn get_inflow_velocity(&self, mesh: &mut Mesh) -> Option<Vector3<f64>> {
        let inflow_elements = self.boundary_manager.get_inflow_elements(mesh);

        if inflow_elements.is_empty() {
            return None;
        }

        // Sum the velocities of the inflow elements
        let total_velocity: Vector3<f64> = inflow_elements.iter()
            .filter_map(|&element_id| self.get_velocity(element_id))
            .fold(Vector3::zeros(), |acc, vel| acc + vel);

        // Compute the average velocity
        let avg_velocity = total_velocity / inflow_elements.len() as f64;

        Some(avg_velocity)
    }

    /// Retrieves the average outflow velocity at the outflow boundary.
    ///
    /// This function calculates the outflow velocity by averaging the velocities
    /// of all elements at the outflow boundary.
    ///
    /// # Parameters
    /// - `mesh`: A mutable reference to the `Mesh`.
    ///
    /// # Returns
    /// An `Option<Vector3<f64>>` containing the average outflow velocity, or `None` if no outflow elements are found.
    pub fn get_outflow_velocity(&self, mesh: &mut Mesh) -> Option<Vector3<f64>> {
        let outflow_elements = self.boundary_manager.get_outflow_elements(mesh);

        if outflow_elements.is_empty() {
            return None;
        }

        // Sum the velocities of the outflow elements
        let total_velocity: Vector3<f64> = outflow_elements.iter()
            .filter_map(|&element_id| self.get_velocity(element_id))
            .fold(Vector3::zeros(), |acc, vel| acc + vel);

        // Compute the average velocity
        let avg_velocity = total_velocity / outflow_elements.len() as f64;

        Some(avg_velocity)
    }

    /// Retrieves the total outflow mass rate at the outflow boundary.
    ///
    /// This function calculates the outflow mass rate by multiplying the average outflow velocity,
    /// area, and density of the outflow elements.
    ///
    /// # Parameters
    /// - `mesh`: A mutable reference to the `Mesh`.
    ///
    /// # Returns
    /// The total outflow mass rate as an `f64`.
    pub fn get_outflow_mass_rate(&self, mesh: &mut Mesh) -> f64 {
        let outflow_elements = self.boundary_manager.get_outflow_elements(mesh);
        let velocity = self.get_outflow_velocity(mesh).unwrap_or(Vector3::zeros());
        let area = self.calculate_outflow_area(mesh, &outflow_elements);
        let density = self.calculate_outflow_density(&outflow_elements);

        // Return the mass flow rate: density * velocity * area
        density * velocity.magnitude() * area
    }

    /// Calculates the total outflow area by summing the areas of outflow elements.
    fn calculate_outflow_area(&self, mesh: &Mesh, outflow_elements: &[u32]) -> f64 {
        outflow_elements.iter().fold(0.0, |total_area, &element_id| {
            if let Some(element) = mesh.get_element_by_id(element_id) {
                total_area + element.area
            } else {
                total_area
            }
        })
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryManager;
    use crate::domain::{Element, Mesh};
    use nalgebra::Vector3;

    /// Helper function to create a mock element for testing.
    fn create_mock_element(id: u32, mass: f64, area: f64, velocity: Vector3<f64>, pressure: f64) -> Element {
        Element {
            id,
            mass,
            area,
            velocity,
            pressure,
            centroid_coordinates: vec![0.0, 0.0, 0.0],
            neighbor_refs: vec![],
            neighbor_distance: vec![],
            ..Element::default()
        }
    }

    /// Helper function to create a mock mesh with one element for testing.
    fn create_mock_mesh() -> Mesh {
        Mesh {
            faces: vec![],
            nodes: vec![],
            elements: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        }
    }

    /// Helper function to create a mock flow field.
    fn create_mock_flow_field() -> FlowField {
        let element1 = create_mock_element(1, 5.0, 2.0, Vector3::new(1.0, 0.0, 0.0), 10.0);
        let element2 = create_mock_element(2, 3.0, 1.5, Vector3::new(0.0, 1.0, 0.0), 8.0);
        let elements = vec![element1, element2];
        let boundary_manager = BoundaryManager::new();
        FlowField::new(elements, boundary_manager)
    }

    #[test]
    fn test_initial_mass_calculation() {
        let flow_field = create_mock_flow_field();
        assert_eq!(flow_field.initial_mass, 8.0, "Initial mass should be correctly calculated.");
    }

    #[test]
    fn test_compute_mass() {
        let flow_field = create_mock_flow_field();
        let element = &flow_field.elements[0];
        assert_eq!(flow_field.compute_mass(element), 5.0, "Mass should be correctly computed.");
    }

    #[test]
    fn test_compute_density() {
        let flow_field = create_mock_flow_field();
        let element = &flow_field.elements[0];
        assert_eq!(flow_field.compute_density(element), Some(2.5), "Density should be correctly calculated.");
    }

    #[test]
    fn test_compute_density_zero_area() {
        let element = create_mock_element(1, 5.0, 0.0, Vector3::new(1.0, 0.0, 0.0), 10.0);
        let flow_field = FlowField::new(vec![element], BoundaryManager::new());
        assert_eq!(flow_field.compute_density(&flow_field.elements[0]), None, "Density should return None for zero area.");
    }

    #[test]
    fn test_mass_conservation_pass() {
        let flow_field = create_mock_flow_field();
        assert!(flow_field.check_mass_conservation().is_ok(), "Mass conservation should pass when mass is conserved.");
    }

    #[test]
    fn test_mass_conservation_fail() {
        let mut flow_field = create_mock_flow_field();
        flow_field.elements[0].mass = 10.0;  // Change mass to violate mass conservation
        assert!(flow_field.check_mass_conservation().is_err(), "Mass conservation should fail when mass changes.");
    }

    #[test]
    fn test_get_velocity() {
        let flow_field = create_mock_flow_field();
        let velocity = flow_field.get_velocity(1);
        assert_eq!(velocity, Some(Vector3::new(1.0, 0.0, 0.0)), "Velocity should be correctly retrieved.");
    }

    #[test]
    fn test_get_pressure() {
        let flow_field = create_mock_flow_field();
        let pressure = flow_field.get_pressure(1);
        assert_eq!(pressure, Some(10.0), "Pressure should be correctly retrieved.");
    }

    #[test]
    fn test_compute_pressure_gradient_no_neighbors() {
        let flow_field = create_mock_flow_field();
        let element = &flow_field.elements[0];
        let gradient = flow_field.compute_pressure_gradient(element);
        assert_eq!(gradient, Vector3::zeros(), "Pressure gradient should be zero with no neighbors.");
    }

    // Add more tests to cover boundary condition application, inflow mass rate, and other methods as needed.
}