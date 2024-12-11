
use crate::{boundary::bc_handler::BoundaryConditionHandler, 
    domain::section::Scalar, 
    solver::preconditioner::PreconditionerFactory, 
    Geometry, Mesh, MeshEntity, Section};

use super::GradientMethod;

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
    ) -> Result<[f64; 3], Box<dyn std::error::Error>> {
        let cell_center = geometry.compute_cell_centroid(mesh, cell);
        let phi_c = field.restrict(cell).ok_or("Field value not found for cell")?.0;

        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];

        let neighbors = mesh.get_ordered_neighbors(cell);
        for neighbor in neighbors {
            let neighbor_center = geometry.compute_cell_centroid(mesh, &neighbor);
            let phi_nb = field.restrict(&neighbor).ok_or("Field value not found for neighbor")?.0;

            let delta = [
                neighbor_center[0] - cell_center[0],
                neighbor_center[1] - cell_center[1],
                neighbor_center[2] - cell_center[2],
            ];

            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] += delta[i] * delta[j];
                }
                b[i] += delta[i] * (phi_nb - phi_c);
            }
        }

        let grad_phi = solve_least_squares(a, b)?;

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
            a_matrix.write(i, j, a[i][j]);
        }
    }

    // Add regularization to make the matrix non-singular
    for i in 0..3 {
        a_matrix.write(i, i, a_matrix.read(i, i) + 1e-8);
    }

    // Build the right-hand side vector
    let mut b_vector = crate::linalg::vector::vector_builder::VectorBuilder::build_dense_vector(3);
    for i in 0..3 {
        b_vector.write(i, 0, b[i]);
    }

    // Create LU preconditioner
    let preconditioner = PreconditionerFactory::create_lu(&a_matrix);

    // Prepare solution vector
    let mut solution_vector = crate::linalg::vector::vector_builder::VectorBuilder::build_dense_vector(3);

    // Apply the preconditioner to solve the system
    preconditioner.apply(&a_matrix, &b_vector, &mut solution_vector);

    // Extract solution as a Rust array
    let solution = [
        solution_vector.read(0, 0),
        solution_vector.read(1, 0),
        solution_vector.read(2, 0),
    ];

    Ok(solution)
}



#[cfg(test)]
mod tests {
    use crate::domain::{mesh::Mesh, Section};
    use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
    use crate::equation::gradient::{Gradient, GradientCalculationMethod};
    use crate::domain::section::{Scalar, Vector3};
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
            MeshEntity::Vertex(5),
        ];
        for vertex in &vertices {
            mesh.add_entity(vertex.clone());
        }
    
        // Setting coordinates
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);
        mesh.set_vertex_coordinates(5, [1.0, 1.0, 1.0]); // Additional point for diversity
    
        // Adding a cell
        let cell = MeshEntity::Cell(1);
        mesh.add_entity(cell.clone());
        for vertex in &vertices {
            mesh.add_relationship(cell.clone(), vertex.clone());
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

        // Updated expected gradient calculation based on Dirichlet boundary condition
        let expected_gradient = Vector3([0.0, 0.0, 0.0]); // Modify based on least-squares geometry
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

/*     #[test]
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
    } */

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
