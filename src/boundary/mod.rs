use std::collections::HashMap;

use crate::domain::{FlowField, Mesh};

pub mod types;
pub use types::{BoundaryType, BoundaryCondition};

// Import specific boundary condition implementations
pub mod inflow;
pub mod outflow;
pub mod no_slip;
pub mod periodic;
pub mod free_surface;
pub mod open;

pub use inflow::InflowBoundaryCondition;
pub use outflow::OutflowBoundaryCondition;
pub use no_slip::NoSlipBoundaryCondition;
pub use periodic::PeriodicBoundaryCondition;
pub use free_surface::FreeSurfaceBoundaryCondition;
pub use open::OpenBoundaryCondition;

#[derive(Default)]
/// Manages all boundary conditions in the simulation.
pub struct BoundaryManager {
    /// A hashmap storing boundary conditions, mapped to the BoundaryType.
    pub boundaries: HashMap<BoundaryType, Box<dyn BoundaryCondition>>,
}

impl BoundaryManager {
    /// Creates a new BoundaryManager.
    pub fn new() -> Self {
        Self {
            boundaries: HashMap::new(),
        }
    }

    /// Registers a new boundary condition with the manager.
    ///
    /// # Arguments
    /// - `boundary_type`: The type of the boundary (e.g., FreeSurface, Inflow, etc.)
    /// - `condition`: The boundary condition to register, implementing the `BoundaryCondition` trait.
    pub fn register_boundary(&mut self, boundary_type: BoundaryType, condition: Box<dyn BoundaryCondition>) {
        self.boundaries.insert(boundary_type, condition);
    }

    /// Applies all registered boundary conditions to the mesh and flow field.
    ///
    /// # Arguments
    /// - `mesh`: The computational mesh.
    /// - `flow_field`: The flow field representing velocities, pressures, etc.
    /// - `time_step`: The current simulation time step.
    pub fn apply(&self, mesh: &mut Mesh, flow_field: &mut FlowField, time_step: f64) {
        for condition in self.boundaries.values() {
            condition.apply(mesh, flow_field, time_step);
        }
    }

    /// Get a list of elements associated with inflow boundary conditions.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for inflow elements.
    ///
    /// Returns a vector of element IDs.
    pub fn get_inflow_elements(&self, mesh: &Mesh) -> Vec<u32> {
        self.get_boundary_elements(mesh, BoundaryType::Inflow)
    }

    /// Get a list of elements associated with outflow boundary conditions.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for outflow elements.
    ///
    /// Returns a vector of element IDs.
    pub fn get_outflow_elements(&self, mesh: &Mesh) -> Vec<u32> {
        self.get_boundary_elements(mesh, BoundaryType::Outflow)
    }

    /// Get a list of faces associated with the free surface boundary condition.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to search for free surface faces.
    ///
    /// Returns a vector of face IDs.
    pub fn get_free_surface_faces(&self, mesh: &Mesh) -> Vec<u32> {
        let free_surface_faces = self.get_boundary_elements(mesh, BoundaryType::FreeSurface);
        println!("Free surface faces retrieved: {:?}", free_surface_faces);
        free_surface_faces
    }

    /// Retrieves elements (or faces) for a specific boundary type.
    ///
    /// # Arguments
    /// - `mesh`: The mesh to query.
    /// - `boundary_type`: The type of boundary (e.g., Inflow, FreeSurface).
    ///
    /// Returns a vector of element or face IDs.
    fn get_boundary_elements(&self, mesh: &Mesh, boundary_type: BoundaryType) -> Vec<u32> {
        if let Some(boundary_condition) = self.boundaries.get(&boundary_type) {
            boundary_condition.get_boundary_elements(mesh)
        } else {
            Vec::new()  // Return an empty vector if no boundary of that type is registered.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Mesh, Node, Face, Element, FlowField};
    use crate::boundary::{BoundaryType, FreeSurfaceBoundaryCondition, InflowBoundaryCondition, OutflowBoundaryCondition};
    use nalgebra::Vector3;
    use std::collections::HashMap;

    /// Helper function to create a mock mesh for testing
    fn create_mock_mesh() -> Mesh {
        let faces = vec![
            Face::new(1, vec![0, 1], Vector3::new(0.1, 0.0, 0.2), 1.0, Vector3::new(0.0, 1.0, 0.0), Some(BoundaryType::FreeSurface)),
            Face::new(2, vec![2, 3], Vector3::new(0.2, 0.0, 0.2), 1.0, Vector3::new(0.0, 0.0, 1.0), Some(BoundaryType::FreeSurface)),
        ];

        let nodes = vec![
            Node::new(0, Vector3::new(0.0, 1.0, 0.0)),
            Node::new(1, Vector3::new(1.0, 1.0, 0.0)),
            Node::new(2, Vector3::new(0.0, 2.0, 0.0)),
            Node::new(3, Vector3::new(1.0, 2.0, 0.0)),
        ];

        Mesh {
            faces,
            nodes,
            elements: vec![],
            neighbors: Default::default(),
            face_element_relations: vec![],
        }
    }

    /// Helper function to create a mock element
    fn create_mock_element(id: u32, mass: f64, area: f64) -> Element {
        Element {
            id,
            mass,
            area,
            velocity: Vector3::new(1.0, 0.0, 0.0),
            pressure: 1000.0,
            centroid_coordinates: vec![0.0, 0.0, 0.0],
            neighbor_refs: vec![],
            neighbor_distance: vec![],
            ..Element::default()
        }
    }

    /// Helper function to create a mock flow field
    fn create_mock_flow_field() -> FlowField {
        let element1 = create_mock_element(1, 5.0, 2.0);
        let element2 = create_mock_element(2, 3.0, 1.5);
        let elements = vec![element1, element2];
        let boundary_manager = BoundaryManager::new();
        FlowField::new(elements, boundary_manager)
    }

    #[test]
    fn test_register_boundary() {
        let mut boundary_manager = BoundaryManager::new();
        let free_surface_boundary = Box::new(FreeSurfaceBoundaryCondition::default());
        
        boundary_manager.register_boundary(BoundaryType::FreeSurface, free_surface_boundary);

        assert!(boundary_manager.boundaries.contains_key(&BoundaryType::FreeSurface), "FreeSurface boundary should be registered.");
    }

    #[test]
    fn test_apply_boundary_conditions() {
        let mut boundary_manager = BoundaryManager::new();
        let free_surface_boundary = Box::new(FreeSurfaceBoundaryCondition::default());
        boundary_manager.register_boundary(BoundaryType::FreeSurface, free_surface_boundary);

        let mut flow_field = create_mock_flow_field();
        let mut mesh = create_mock_mesh();

        // Apply boundary conditions
        boundary_manager.apply(&mut mesh, &mut flow_field, 0.1);

        // The test checks that no panic occurs and conditions are applied.
        // Additional checks can be added based on boundary condition effects.
    }

    #[test]
    fn test_get_free_surface_faces() {
        let mut boundary_manager = BoundaryManager::new();
        let free_surface_boundary = Box::new(FreeSurfaceBoundaryCondition::default());
        boundary_manager.register_boundary(BoundaryType::FreeSurface, free_surface_boundary);

        let mesh = create_mock_mesh();
        let free_surface_faces = boundary_manager.get_free_surface_faces(&mesh);

        assert_eq!(free_surface_faces.len(), 2, "There should be 2 free surface faces.");
        assert_eq!(free_surface_faces, vec![1, 2], "Free surface faces should have IDs 1 and 2.");
    }

    #[test]
    fn test_get_inflow_elements() {
        let mut boundary_manager = BoundaryManager::new();
        let inflow_boundary = Box::new(InflowBoundaryCondition::default());
        boundary_manager.register_boundary(BoundaryType::Inflow, inflow_boundary);

        let mut mesh = Mesh {
            elements: vec![create_mock_element(1, 5.0, 2.0), create_mock_element(2, 3.0, 1.5)],
            ..Default::default()
        };

        // Assuming that the InflowBoundaryCondition marks some elements
        let inflow_elements = boundary_manager.get_inflow_elements(&mesh);

        // Since this is a mock setup, we're checking that no panic occurs. You can extend this test based on actual InflowBoundary logic.
        assert!(inflow_elements.is_empty(), "Inflow elements should be empty in this mock setup.");
    }

    #[test]
    fn test_get_outflow_elements() {
        let mut boundary_manager = BoundaryManager::new();
        let outflow_boundary = Box::new(OutflowBoundaryCondition::default());
        boundary_manager.register_boundary(BoundaryType::Outflow, outflow_boundary);

        let mut mesh = Mesh {
            elements: vec![create_mock_element(1, 5.0, 2.0), create_mock_element(2, 3.0, 1.5)],
            ..Default::default()
        };

        // Assuming that the OutflowBoundaryCondition marks some elements
        let outflow_elements = boundary_manager.get_outflow_elements(&mesh);

        // Since this is a mock setup, we're checking that no panic occurs. You can extend this test based on actual OutflowBoundary logic.
        assert!(outflow_elements.is_empty(), "Outflow elements should be empty in this mock setup.");
    }
}