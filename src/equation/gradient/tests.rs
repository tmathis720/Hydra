// src/equation/gradient/tests.rs



#[cfg(test)]
mod tests {
    use crate::equation::gradient::GradientCalculationMethod;
    use crate::domain::{mesh::Mesh, MeshEntity, Section};
    use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
    use crate::equation::gradient::Gradient;
    use std::sync::Arc;

    /// Creates a simple mesh used by all tests to ensure consistency.
    fn create_simple_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Create vertices
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);

        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);

        // Set vertex coordinates
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);

        // Create face
        let face = MeshEntity::Face(1);
        mesh.add_entity(face);

        // Create cells
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        mesh.add_entity(cell1);
        mesh.add_entity(cell2);

        // Build relationships
        // Cells to face
        mesh.add_relationship(cell1, face.clone());
        mesh.add_relationship(cell2, face.clone());
        // Cells to vertices
        for &cell in &[cell1, cell2] {
            mesh.add_relationship(cell, vertex1);
            mesh.add_relationship(cell, vertex2);
            mesh.add_relationship(cell, vertex3);
            mesh.add_relationship(cell, vertex4);
        }
        // Face to vertices
        mesh.add_relationship(face.clone(), vertex1);
        mesh.add_relationship(face.clone(), vertex2);
        mesh.add_relationship(face.clone(), vertex3);

        mesh
    }

    #[test]
    fn test_gradient_with_finite_volume_method() {
        let mesh = create_simple_mesh();
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);
        field.set_data(MeshEntity::Cell(2), 2.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_ok(), "Gradient calculation failed: {:?}", result);

        let grad_cell1 = gradient.restrict(&MeshEntity::Cell(1)).expect("Gradient not computed for cell1");
        let expected_grad = [0.0, 0.0, 3.0];
        for i in 0..3 {
            assert!((grad_cell1[i] - expected_grad[i]).abs() < 1e-6, "Mismatch in gradient component {}", i);
        }
    }

    #[test]
    fn test_gradient_with_dirichlet_boundary() {
        let mesh = create_simple_mesh();
        // Remove cell 2 to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Dirichlet(2.0));

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_ok(), "Gradient calculation failed: {:?}", result);

        let grad = gradient.restrict(&MeshEntity::Cell(1)).expect("Gradient not computed");
        let expected_grad = [0.0, 0.0, 3.0];
        for i in 0..3 {
            assert!((grad[i] - expected_grad[i]).abs() < 1e-6, "Mismatch in gradient component {}", i);
        }
    }

    #[test]
    fn test_gradient_with_neumann_boundary() {
        let mesh = create_simple_mesh();
        // Remove cell 2 to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Neumann(2.0));

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_ok(), "Gradient calculation failed: {:?}", result);

        let grad = gradient.restrict(&MeshEntity::Cell(1)).expect("Gradient not computed");
        let expected_grad = [0.0, 0.0, 6.0]; // Adjusted expected gradient
        for i in 0..3 {
            assert!((grad[i] - expected_grad[i]).abs() < 1e-6, "Mismatch in gradient component {}", i);
        }
    }

    #[test]
    fn test_gradient_with_dirichlet_function_boundary() {
        let mesh = create_simple_mesh();
        // Remove cell 2 to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::DirichletFn(Arc::new(|time, _| 1.0 + time)),
        );

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let time = 2.0;
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, time);
        assert!(result.is_ok(), "Gradient calculation failed: {:?}", result);

        let grad = gradient.restrict(&MeshEntity::Cell(1)).expect("Gradient not computed");
        let expected_grad = [0.0, 0.0, 6.0]; // Adjusted expected gradient based on time
        for i in 0..3 {
            assert!((grad[i] - expected_grad[i]).abs() < 1e-6, "Mismatch in gradient component {}", i);
        }
    }

    #[test]
    fn test_gradient_error_on_missing_data() {
        let mesh = Mesh::new();
        let cell = MeshEntity::Cell(1);
        mesh.add_entity(cell.clone());

        let field = Section::<f64>::new(); // Field data is missing for the cell
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_err(), "Expected error due to missing field values");
    }

    #[test]
    fn test_gradient_error_on_unimplemented_robin_condition() {
        let mesh = create_simple_mesh();
        // Remove cell 2 to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::Robin { alpha: 1.0, beta: 2.0 },
        );

        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler, GradientCalculationMethod::FiniteVolume);

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(result.is_err(), "Expected error due to unimplemented Robin condition");
    }

}
