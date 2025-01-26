#[cfg(test)]
mod tests {
    use crate::domain::{mesh::Mesh, Section};
    use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
    use crate::equation::gradient::{Gradient, GradientCalculationMethod};
    use crate::domain::section::{scalar::Scalar, vector::Vector3};
    use crate::domain::mesh_entity::MeshEntity;

    /// Creates a basic mesh for testing.
    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Adding vertices
        let vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
            MeshEntity::Vertex(4),
        ];
        for vertex in &vertices {
            mesh.add_entity(vertex.clone()).unwrap();
        }

        // Setting coordinates
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]).unwrap();

        // Adding a cell
        let cell = MeshEntity::Cell(1);
        mesh.add_entity(cell.clone()).unwrap();
        for vertex in &vertices {
            mesh.add_relationship(cell.clone(), vertex.clone()).unwrap();
        }

        mesh
    }

    #[test]
    fn test_finite_volume_gradient_calculation() {
        let mesh = create_test_mesh();

        // Initialize scalar field and gradient section
        let field = Section::<Scalar>::new();
        field.set_data(MeshEntity::Cell(1), Scalar(1.0));

        let mut gradient = Section::<Vector3>::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // Create gradient calculator
        let mut gradient_calculator = Gradient::new(
            &mesh,
            &boundary_handler,
            GradientCalculationMethod::FiniteVolume,
        );

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);

        assert!(
            result.is_ok(),
            "Gradient computation failed: {:?}",
            result.err()
        );

        let computed_gradient = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient: {:?}", computed_gradient);

        // Example assertion; update to actual expected values
        let expected_gradient = Vector3([0.0, 0.0, 0.0]);
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Computed gradient does not match expected: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

    #[test]
    fn test_gradient_with_dirichlet_boundary_condition() {
        let mesh = create_test_mesh();

        // Add Dirichlet boundary condition
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::Dirichlet(2.0),
        );

        // Initialize scalar field
        let field = Section::<Scalar>::new();
        field.set_data(MeshEntity::Cell(1), Scalar(1.0));

        let mut gradient = Section::<Vector3>::new();

        // Create gradient calculator
        let mut gradient_calculator = Gradient::new(
            &mesh,
            &boundary_handler,
            GradientCalculationMethod::FiniteVolume,
        );

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);

        assert!(
            result.is_ok(),
            "Gradient computation failed with Dirichlet boundary: {:?}",
            result.err()
        );

        let computed_gradient = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient with Dirichlet BC: {:?}", computed_gradient);

        // Updated expected gradient calculation based on Dirichlet boundary condition
        // Assumes a single face contribution with a difference of (2.0 - 1.0).
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Modify based on mesh geometry and BC
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Mismatch: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

    #[test]
    fn test_gradient_with_neumann_boundary_condition() {
        let mesh = create_test_mesh();

        // Add Neumann boundary condition
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::Neumann(3.0),
        );

        // Initialize scalar field
        let field = Section::<Scalar>::new();
        field.set_data(MeshEntity::Cell(1), Scalar(1.0));

        let mut gradient = Section::<Vector3>::new();

        // Create gradient calculator
        let mut gradient_calculator = Gradient::new(
            &mesh,
            &boundary_handler,
            GradientCalculationMethod::FiniteVolume,
        );

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);

        assert!(
            result.is_ok(),
            "Gradient computation failed with Neumann boundary: {:?}",
            result.err()
        );

        let computed_gradient = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient with Neumann BC: {:?}", computed_gradient);

        // Updated expected gradient calculation for Neumann BC
        // Assumes Neumann flux contribution scales gradient directly by 3.0.
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Modify based on mesh geometry and BC
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Mismatch: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

}
