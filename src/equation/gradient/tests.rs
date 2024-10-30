// src/equation/gradient/tests.rs

use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
use crate::equation::gradient::gradient_calc::Gradient;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use crate::FaceShape;

    use super::*;

    fn mock_face_normal(_mesh: &Mesh, _face: &MeshEntity, _cell: &MeshEntity) -> [f64; 3] {
        // Simulated normal based on test requirements
        [1.0, 0.0, 0.0]
    }

    fn mock_face_area(_mesh: &Mesh, _face: &MeshEntity) -> f64 {
        // Simulated area value
        1.0
    }

    fn mock_cell_volume(_mesh: &Mesh, _cell: &MeshEntity) -> f64 {
        // Simulated volume value
        1.0
    }

    #[test]
    fn test_gradient_simple_mesh() {
        let mut mesh = Mesh::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face = MeshEntity::Face(1);
    
        // Add vertices with coordinates and connect them to both the face and cells
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4); // Fourth vertex for the cell
    
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
    
        // Set vertex coordinates
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);
    
        // Set up face and relationships with vertices
        mesh.add_entity(face);
        mesh.add_relationship(face, vertex1);
        mesh.add_relationship(face, vertex2);
        mesh.add_relationship(face, vertex3);
    
        // Test if get_face_vertices returns the vertices for this face
        let face_vertices = mesh.get_face_vertices(&face);
        assert_eq!(face_vertices.len(), 3, "Expected 3 vertices for the face");
        eprintln!("Face vertices: {:?}", face_vertices);
    
        // Set up cells and relationships with vertices and face
        mesh.add_entity(cell1);
        mesh.add_entity(cell2);
        mesh.add_relationship(cell1, face);
        mesh.add_relationship(cell2, face);
        mesh.add_relationship(cell1, vertex1);
        mesh.add_relationship(cell1, vertex2);
        mesh.add_relationship(cell1, vertex3);
        mesh.add_relationship(cell1, vertex4);
        mesh.add_relationship(cell2, vertex1);
        mesh.add_relationship(cell2, vertex2);
        mesh.add_relationship(cell2, vertex3);
        mesh.add_relationship(cell2, vertex4);
    
        // Initialize Sections for geometry properties
        let mut face_normals = Section::<[f64; 3]>::new();
        face_normals.set_data(face, [1.0, 0.0, 0.0]);
    
        let mut face_areas = Section::<f64>::new();
        face_areas.set_data(face, 1.0);
    
        let mut cell_volumes = Section::<f64>::new();
        cell_volumes.set_data(cell1, 1.0);
        cell_volumes.set_data(cell2, 1.0);
    
        // Define face shape for gradient calculation
        let mut face_shapes = Section::<FaceShape>::new();
        face_shapes.set_data(face, FaceShape::Triangle);
    
        // Initialize field values
        let mut field = Section::<f64>::new();
        field.set_data(cell1, 1.0);
        field.set_data(cell2, 2.0);
    
        // Set up gradient computation
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);
    
        // Compute the gradient and check for success
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_ok(), "Gradient calculation failed with result: {:?}", result);
    
        // Check that the gradient for cell1 matches the expected result
        let grad_cell1 = gradient.restrict(&cell1).expect("Gradient not computed for cell1");
        let expected_grad = [1.0, 0.0, 0.0];
        for i in 0..3 {
            assert!((grad_cell1[i] - expected_grad[i]).abs() < 1e-6);
        }
    }
    


    #[test]
    fn test_gradient_dirichlet_boundary() {
        let mut mesh = Mesh::new();
        let cell = MeshEntity::Cell(1);
        let face = MeshEntity::Face(1);

        mesh.add_entity(cell);
        mesh.add_entity(face);
        mesh.add_relationship(cell, face);

        // Use mock functions for testing
        let _face_normal = mock_face_normal(&mesh, &face, &cell);
        let _face_area = mock_face_area(&mesh, &face);
        let _cell_volume = mock_cell_volume(&mesh, &cell);

        let field = Section::<f64>::new();
        field.set_data(cell, 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(face.clone(), BoundaryCondition::Dirichlet(2.0));

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_ok());

        let grad = gradient.restrict(&cell).expect("Gradient not computed");
        let expected_grad = [1.0, 0.0, 0.0];
        for i in 0..3 {
            assert!((grad[i] - expected_grad[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_functional_boundary() {
        let mut mesh = Mesh::new();
        let cell = MeshEntity::Cell(1);
        let face = MeshEntity::Face(1);

        mesh.add_entity(cell);
        mesh.add_entity(face);
        mesh.add_relationship(cell, face);

        // Use mock functions for testing
        let _face_normal = mock_face_normal(&mesh, &face, &cell);
        let _face_area = mock_face_area(&mesh, &face);
        let _cell_volume = mock_cell_volume(&mesh, &cell);

        let field = Section::<f64>::new();
        field.set_data(cell, 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            face.clone(),
            BoundaryCondition::DirichletFn(Arc::new(|time, _| 1.0 + time)),
        );

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 2.0);
        assert!(result.is_ok());

        let grad = gradient.restrict(&cell).expect("Gradient not computed");
        let expected_grad = [0.0, 2.0, 0.0];
        for i in 0..3 {
            assert!((grad[i] - expected_grad[i]).abs() < 1e-6);
        }
    }

    // Additional test cases can follow similar patterns
}
