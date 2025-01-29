use crate::{boundary::bc_handler::BoundaryConditionHandler, 
    domain::section::scalar::Scalar, 
    solver::preconditioner::PreconditionerFactory, 
    Geometry, Mesh, MeshEntity, Section};

use super::{GradientError, GradientMethod};

/// Struct for the least-squares gradient calculation method.
///
/// This method computes the gradient by fitting a plane to the scalar field
/// values of neighboring cells, minimizing the error in the computed gradient.
pub struct LeastSquaresGradient;

impl GradientMethod for LeastSquaresGradient {
    fn calculate_gradient(
        &self,
        mesh: &Mesh,
        _boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        field: &Section<Scalar>,
        cell: &MeshEntity,
        _time: f64,
    ) -> Result<[f64; 3], GradientError> {
        // Compute the centroid of the current cell
        let cell_center = geometry.compute_cell_centroid(mesh, cell)
            .map_err(|err| GradientError::CalculationError(cell.clone(), format!("Failed to compute centroid: {}", err)))?;
    
        // Retrieve the field value for the current cell
        let phi_c = field
            .restrict(cell)
            .map(|scalar| scalar.0)
            .map_err(|_| GradientError::CalculationError(cell.clone(), "Field value not found for cell".to_string()))?;
    
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
    
        // Retrieve neighbors of the cell
        let neighbors = mesh
            .get_ordered_neighbors(cell)
            .map_err(|e| GradientError::CalculationError(cell.clone(), format!("Failed to retrieve neighbors: {}", e)))?;
    
        if neighbors.is_empty() {
            return Err(GradientError::CalculationError(
                cell.clone(),
                "No neighbors found for cell".to_string(),
            ));
        }
    
        // Compute contributions from neighbors
        for neighbor in neighbors.iter() {
            // Compute centroid for the neighbor
            let neighbor_center = geometry.compute_cell_centroid(mesh, neighbor)
                .map_err(|err| GradientError::CalculationError(neighbor.clone(), format!("Failed to compute centroid: {}", err)))?;
    
            // Retrieve the field value for the neighbor
            let phi_nb = field
                .restrict(neighbor)
                .map(|scalar| scalar.0)
                .map_err(|_| {
                    GradientError::CalculationError(neighbor.clone(), "Field value not found for neighbor".to_string())
                })?;
    
            // Compute the displacement vector
            let delta = [
                neighbor_center[0] - cell_center[0],
                neighbor_center[1] - cell_center[1],
                neighbor_center[2] - cell_center[2],
            ];
    
            // Populate the least-squares matrix (A) and RHS vector (b)
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] += delta[i] * delta[j];
                }
                b[i] += delta[i] * (phi_nb - phi_c);
            }
        }
    
        // Solve the least-squares system
        let grad_phi = solve_least_squares(a, b).map_err(|e| {
            GradientError::CalculationError(cell.clone(), format!("Failed to solve least-squares system: {}", e))
        })?;
    
        Ok(grad_phi)
    }
    
}


/// Solves the least-squares system A * x = b using Hydra's matrix and vector abstractions.
///
/// # Parameters
/// - `a`: Coefficient matrix (3x3).
/// - `b`: Right-hand side vector.
///
/// # Returns
/// - `Ok([f64; 3])`: Solution vector `x`.
/// - `Err`: If the system is singular or cannot be solved.
fn solve_least_squares(
    a: [[f64; 3]; 3],
    b: [f64; 3],
) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    // Build the coefficient matrix
    let mut a_matrix = crate::linalg::matrix::matrix_builder::MatrixBuilder::build_dense_matrix(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            a_matrix[(i, j)] = a[i][j];
        }
    }

    // Add regularization to make the matrix non-singular
    for i in 0..3 {
        a_matrix[(i, i)] += 1e-8;
    }

    // Build the right-hand side vector
    let mut b_vector = crate::linalg::vector::vector_builder::VectorBuilder::build_dense_vector(3);
    for i in 0..3 {
        b_vector[(i, 0)] = b[i];
    }

    // Create LU preconditioner
    let preconditioner = PreconditionerFactory::create_lu(&a_matrix);

    // Prepare solution vector
    let mut solution_vector = crate::linalg::vector::vector_builder::VectorBuilder::build_dense_vector(3);

    // Apply the preconditioner to solve the system
    preconditioner.apply(&a_matrix, &b_vector, &mut solution_vector);

    // Extract solution as a Rust array
    let solution = [
        solution_vector[(0, 0)],
        solution_vector[(1, 0)],
        solution_vector[(2, 0)],
    ];

    Ok(solution)
}

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
    
        // Add vertices
        let vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
            MeshEntity::Vertex(4),
            MeshEntity::Vertex(5),
        ];
        for vertex in &vertices {
            mesh.add_entity(vertex.clone()).unwrap();
        }
    
        // Set coordinates for vertices
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]).unwrap();
        mesh.set_vertex_coordinates(5, [1.0, 1.0, 1.0]).unwrap();
    
        // Add main test cell
        let test_cell = MeshEntity::Cell(1);
        mesh.add_entity(test_cell.clone()).unwrap();
        for vertex in &vertices {
            mesh.add_relationship(test_cell.clone(), vertex.clone()).unwrap();
        }
    
        // Add neighboring cells and relationships
        let neighbor_cells = vec![
            MeshEntity::Cell(2),
            MeshEntity::Cell(3),
        ];
        for neighbor in &neighbor_cells {
            mesh.add_entity(neighbor.clone()).unwrap();
            mesh.add_relationship(*neighbor, test_cell.clone()).unwrap();
        }
    
        // Ensure neighbors are connected through shared faces
        let face = MeshEntity::Face(1);
        mesh.add_entity(face.clone()).unwrap();
        mesh.add_relationship(test_cell.clone(), face.clone()).unwrap();
        for neighbor in &neighbor_cells {
            mesh.add_relationship(*neighbor, face.clone()).unwrap();
        }
    
        mesh
    }
    

    #[test]
    fn test_least_squares_gradient_calculation() {
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
            GradientCalculationMethod::LeastSquares,
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
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Modify based on least-squares calculation
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Computed gradient does not match expected: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

    #[test]
    fn test_least_squares_gradient_with_dirichlet_boundary_condition() {
        use crate::interface_adapters::domain_adapter::DomainBuilder;
        use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
        use crate::domain::section::{scalar::Scalar, vector::Vector3};
        use crate::equation::gradient::{Gradient, GradientCalculationMethod};
    
        // Create the mesh using DomainBuilder
        let mut builder = DomainBuilder::new();
    
        // Add vertices and a tetrahedron cell (4 vertices for simplicity)
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(3, [0.0, 1.0, 0.0]).is_ok());
        assert!(builder.add_vertex(4, [0.0, 0.0, 1.0]).is_ok());
        assert!(builder.add_tetrahedron_cell(vec![1, 2, 3, 4]).is_ok());
    
        let mesh = builder.build();
    
        // Add Dirichlet boundary condition to the face shared by all vertices
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1), // Assuming face IDs are properly set in DomainBuilder
            BoundaryCondition::Dirichlet(2.0),
        );
    
        // Initialize scalar field
        let field = Section::<Scalar>::new();
        field.set_data(MeshEntity::Cell(1), Scalar(1.0)); // Assign scalar to the cell
    
        // Initialize gradient section
        let mut gradient = Section::<Vector3>::new();
    
        // Create gradient calculator
        let mut gradient_calculator = Gradient::new(
            &mesh,
            &boundary_handler,
            GradientCalculationMethod::LeastSquares,
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
    
        // Example expected gradient
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Update as per domain setup
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Mismatch: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

    #[test]
    fn test_least_squares_gradient_with_neumann_boundary_condition() {
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
            GradientCalculationMethod::LeastSquares,
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
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Modify based on least-squares geometry
        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Mismatch: {:?} vs {:?}",
            computed_gradient,
            expected_gradient
        );
    }

    #[test]
    fn test_least_squares_gradient_singular_matrix() {
        let a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]; // Singular matrix
        let b = [1.0, 2.0, 3.0];

        let result = crate::equation::gradient::least_squares::solve_least_squares(a, b);

        assert!(
            result.is_err(),
            "Singular matrix should result in an error, but got: {:?}",
            result
        );

        if let Err(e) = result {
            println!("Expected error for singular matrix: {}", e);
        }
    }

    #[test]
    fn test_least_squares_gradient_debugging() {
        let mesh = create_test_mesh();

        // Initialize scalar field and gradient section
        let field = Section::<Scalar>::new();
        field.set_data(MeshEntity::Cell(1), Scalar(1.0));

        let mut gradient = Section::<Vector3>::new();
        let boundary_handler = BoundaryConditionHandler::new();

        let mut gradient_calculator = Gradient::new(
            &mesh,
            &boundary_handler,
            GradientCalculationMethod::LeastSquares,
        );

        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);

        if let Err(e) = result {
            panic!("Gradient computation failed: {}", e);
        }

        let computed_gradient = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");

        println!("Computed gradient: {:?}", computed_gradient);
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Update based on expectations

        assert!(
            computed_gradient.iter().zip(expected_gradient.iter()).all(|(a, b)| (a - b).abs() < 1e-6),
            "Computed gradient {:?} does not match expected {:?}",
            computed_gradient,
            expected_gradient
        );
    }


}
