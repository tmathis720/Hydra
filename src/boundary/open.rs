use crate::boundary::BoundaryCondition;
use crate::domain::{Element, FlowField, Mesh};
use nalgebra::Vector3;

/// Represents an open boundary condition that can handle both inflow and outflow behavior.
pub struct OpenBoundaryCondition {
    pub boundary_map: Vec<(usize, usize)>,  // Custom boundary mapping for inflow/outflow behavior
}

impl OpenBoundaryCondition {
    /// Constructor for creating an OpenBoundary with an optional custom mapping.
    pub fn new(boundary_map: Option<Vec<(usize, usize)>>) -> Self {
        let boundary_map = boundary_map.unwrap_or_else(|| vec![]);
        Self { boundary_map }
    }

    /// Adds a custom inflow/outflow mapping (source_index -> target_index).
    pub fn add_boundary_mapping(&mut self, source_index: usize, target_index: usize) {
        self.boundary_map.push((source_index, target_index));
    }

    /// This method should now query the `BoundaryManager` for inflow and outflow elements.
    fn apply_boundary_conditions(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        // Get inflow elements and outflow elements first
        let inflow_elements = flow_field.boundary_manager.get_inflow_elements(mesh);
        let outflow_elements = flow_field.boundary_manager.get_outflow_elements(mesh);

        // Apply inflow logic
        for inflow_element_id in inflow_elements {
            // Extract inflow velocity and mass rate before mutably borrowing flow_field
            let inflow_velocity = flow_field.get_inflow_velocity(mesh).unwrap_or(Vector3::zeros());
            let mass_rate = flow_field.get_inflow_mass_rate(mesh) * time_step;
            
            // Extract pressure separately before the mutable borrow occurs
            let pressure = flow_field.get_pressure(inflow_element_id).unwrap_or(0.0);

            if let Some(target_element) = flow_field.elements.get_mut(inflow_element_id as usize) {
                // Apply inflow logic without accessing flow_field again
                self.apply_inflow(target_element, inflow_velocity, mass_rate, pressure);
            }
        }

        // Apply outflow logic
        for outflow_element_id in outflow_elements {
            // Extract outflow velocity and mass rate before mutably borrowing flow_field
            let outflow_velocity = flow_field.get_outflow_velocity(mesh).unwrap_or(Vector3::zeros());
            let mass_rate = flow_field.get_outflow_mass_rate(mesh) * time_step;

            // Extract pressure separately before the mutable borrow occurs
            let pressure = flow_field.get_pressure(outflow_element_id).unwrap_or(0.0);

            if let Some(target_element) = flow_field.elements.get_mut(outflow_element_id as usize) {
                // Apply outflow logic without accessing flow_field again
                self.apply_outflow(target_element, outflow_velocity, mass_rate, pressure);
            }
        }
    }

    // Adjust inflow logic to avoid mutable borrow issues
    fn apply_inflow(
        &self,
        target_element: &mut Element,
        inflow_velocity: Vector3<f64>,
        mass_addition: f64,
        pressure: f64
    ) {
        target_element.mass += mass_addition;
        target_element.momentum += inflow_velocity * mass_addition;
        target_element.pressure = pressure;
    }

    // Adjust outflow logic to avoid mutable borrow issues
    fn apply_outflow(
        &self,
        target_element: &mut Element,
        outflow_velocity: Vector3<f64>,
        mass_removal: f64,
        pressure: f64
    ) {
        target_element.mass = (target_element.mass - mass_removal).max(0.0);
        target_element.momentum -= outflow_velocity * mass_removal;
        target_element.momentum = target_element.momentum.map(|val| val.max(0.0));
        target_element.pressure = pressure;
    }

    /// Identify inflow boundary elements in the mesh
    fn get_inflow_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        mesh.elements.iter()
            .filter(|e| is_inflow(e))
            .map(|e| e.id)
            .collect()
    }

    /// Identify outflow boundary elements in the mesh
    fn get_outflow_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        mesh.elements.iter()
            .filter(|e| is_outflow(e))
            .map(|e| e.id)
            .collect()
    }
}

/// Determines whether an element is at an inflow boundary.
fn is_inflow(element: &Element) -> bool {
    element.velocity.x > 0.0  // Example: inflow if x-velocity is positive
}

/// Determines whether an element is at an outflow boundary.
fn is_outflow(element: &Element) -> bool {
    element.velocity.x < 0.0  // Example: outflow if x-velocity is negative
}

impl BoundaryCondition for OpenBoundaryCondition {
    fn update(&mut self, _time: f64) {
        // No specific update logic needed for open boundary conditions
    }

    fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        self.apply_boundary_conditions(mesh, flow_field, time_step);
    }

    fn velocity(&self) -> Option<Vector3<f64>> {
        None  // Open boundary does not specify a single velocity
    }

    fn mass_rate(&self) -> Option<f64> {
        None  // Open boundary does not specify a single mass rate
    }

    /// Retrieve the inflow and outflow boundary elements from the flow field
    fn get_boundary_elements(&self, mesh: &Mesh) -> Vec<u32> {
        let inflow_elements = self.get_inflow_boundary_elements(mesh);
        let outflow_elements = self.get_outflow_boundary_elements(mesh);

        // Combine both inflow and outflow elements
        [inflow_elements, outflow_elements].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryManager, BoundaryType};
    use crate::domain::{Element, FlowField, Mesh, Node};
    use nalgebra::Vector3;

    /// Helper function to create a mock mesh with elements.
    fn create_mock_mesh() -> Mesh {
        let elements = vec![
            Element {
                id: 1,
                velocity: Vector3::new(1.0, 0.0, 0.0),  // Inflow (positive x-velocity)
                mass: 5.0,
                pressure: 101.0,
                ..Default::default()
            },
            Element {
                id: 2,
                velocity: Vector3::new(-1.0, 0.0, 0.0), // Outflow (negative x-velocity)
                mass: 3.0,
                pressure: 100.0,
                ..Default::default()
            },
        ];

        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 1.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 1.0, 0.0)),
            Node::new(2, Vector3::new(0.0, 2.0, 0.0)),
            Node::new(3, Vector3::new(1.0, 2.0, 0.0)),
        ];

        Mesh {
            elements,
            nodes,
            faces: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        }
    }

    #[test]
    fn test_apply_open_boundary_conditions_inflow() {
        let mut mesh = create_mock_mesh();

        // Create an OpenBoundaryCondition with default settings
        let open_boundary = OpenBoundaryCondition::new(None);

        // Create a BoundaryManager and register the open boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::Open, Box::new(open_boundary));

        // Create the FlowField with elements and boundary manager
        let mut flow_field = FlowField::new(mesh.elements.clone(), boundary_manager);

        // Apply the boundary conditions for a time step
        let time_step = 0.1;
        flow_field.apply_boundary_conditions(&mut mesh, time_step);

        // Verify inflow element (ID 1)
        let element_1 = &flow_field.elements[0]; // ID 1 is at index 0
        assert!(element_1.mass > 5.0, "Inflow element mass should increase.");
        assert!(element_1.momentum.x > 1.0, "Inflow element momentum should increase.");
        assert_eq!(element_1.pressure, 101.0, "Inflow element pressure should remain unchanged.");
    }

    #[test]
    fn test_apply_open_boundary_conditions_outflow() {
        let mut mesh = create_mock_mesh();

        // Create an OpenBoundaryCondition with default settings
        let open_boundary = OpenBoundaryCondition::new(None);

        // Create a BoundaryManager and register the open boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::Open, Box::new(open_boundary));

        // Create the FlowField with elements and boundary manager
        let mut flow_field = FlowField::new(mesh.elements.clone(), boundary_manager);

        // Apply the boundary conditions for a time step
        let time_step = 0.1;
        flow_field.apply_boundary_conditions(&mut mesh, time_step);

        // Verify outflow element (ID 2)
        let element_2 = &flow_field.elements[1]; // ID 2 is at index 1
        assert!(element_2.mass < 3.0, "Outflow element mass should decrease.");
        assert!(element_2.momentum.x < -1.0, "Outflow element momentum should decrease.");
        assert_eq!(element_2.pressure, 100.0, "Outflow element pressure should remain unchanged.");
    }

    #[test]
    fn test_custom_boundary_mapping() {
        let mut mesh = create_mock_mesh();

        // Create a custom mapping for inflow and outflow
        let mut open_boundary = OpenBoundaryCondition::new(None);
        open_boundary.add_boundary_mapping(1, 2);  // Map inflow element 1 to outflow element 2

        // Create a BoundaryManager and register the open boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::Open, Box::new(open_boundary));

        // Create the FlowField with elements and boundary manager
        let mut flow_field = FlowField::new(mesh.elements.clone(), boundary_manager);

        // Apply the boundary conditions for a time step
        let time_step = 0.1;
        flow_field.apply_boundary_conditions(&mut mesh, time_step);

        // Check if custom boundary mapping applied correctly (e.g., inflow modifies outflow)
        let element_1 = &flow_field.elements[0]; // ID 1 inflow
        let element_2 = &flow_field.elements[1]; // ID 2 outflow

        assert!(element_1.mass > 5.0, "Inflow element mass should increase.");
        assert!(element_2.mass < 3.0, "Outflow element mass should decrease.");
    }

    #[test]
    fn test_no_change_if_no_inflow_or_outflow() {
        let mut mesh = create_mock_mesh();

        // Modify mesh so no elements qualify as inflow or outflow
        mesh.elements[0].velocity = Vector3::new(0.0, 0.0, 0.0); // Neither inflow nor outflow
        mesh.elements[1].velocity = Vector3::new(0.0, 0.0, 0.0); // Neither inflow nor outflow

        // Create an OpenBoundaryCondition with default settings
        let open_boundary = OpenBoundaryCondition::new(None);

        // Create a BoundaryManager and register the open boundary condition
        let mut boundary_manager = BoundaryManager::new();
        boundary_manager.register_boundary(BoundaryType::Open, Box::new(open_boundary));

        // Create the FlowField with elements and boundary manager
        let mut flow_field = FlowField::new(mesh.elements.clone(), boundary_manager);

        // Apply the boundary conditions for a time step
        let time_step = 0.1;
        flow_field.apply_boundary_conditions(&mut mesh, time_step);

        // Verify no changes are applied
        let element_1 = &flow_field.elements[0]; // ID 1
        let element_2 = &flow_field.elements[1]; // ID 2

        assert_eq!(element_1.mass, 5.0, "No change expected for element 1 mass.");
        assert_eq!(element_2.mass, 3.0, "No change expected for element 2 mass.");
    }
}
