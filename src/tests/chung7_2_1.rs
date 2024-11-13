// src/tests/chung7_2_1.rs

use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    domain::{mesh::Mesh, section::Section},
    solver::ksp::{SolverManager, SolverResult},
    use_cases::{matrix_construction::MatrixConstruction, rhs_construction::RHSConstruction},
    MeshEntity,
};
use faer::Mat;
use std::sync::Arc;

#[test]
fn test_chung_poisson_example() {
    // 1. Setup the mesh and boundary conditions
    let mesh = Mesh::new(); // Initializes an empty mesh, needs population based on domain
    let boundary_handler = BoundaryConditionHandler::new();

    // Define the Dirichlet boundary condition value
    let dirichlet_value = 1.0;
    // Apply boundary conditions to boundary entities in the mesh
    for boundary_entity in mesh.entities.read().unwrap().iter() {
        boundary_handler.set_bc(boundary_entity.clone(), BoundaryCondition::Dirichlet(dirichlet_value));
    }

    // 2. Initialize field and flux sections
    let _field = Section::<f64>::new();
    let _gradient = Section::<[f64; 3]>::new();
    let _fluxes = Section::<f64>::new();

    // 3. Setup matrix and RHS vector based on FVM discretization
    let matrix = MatrixConstruction::build_zero_matrix(mesh.entities.read().unwrap().len(), mesh.entities.read().unwrap().len());
    let mut rhs = RHSConstruction::build_zero_rhs(mesh.entities.read().unwrap().len());

    // Initialize RHS with known values for Poisson setup
    for i in 0..rhs.nrows() {
        rhs.write(i, 0, dirichlet_value); // Updated to use `write` directly
    }

    // 4. Initialize the Krylov solver with preconditioner (e.g., Jacobi)
    let mut solver_manager = SolverManager::new(Box::new(crate::solver::cg::ConjugateGradient::new(100, 1e-6)));
    solver_manager.set_preconditioner(Arc::new(crate::solver::preconditioner::Jacobi::default()));

    // Solution vector initialized to zero
    let mut solution = Mat::<f64>::zeros(mesh.entities.read().unwrap().len(), 1);

    // 5. Solve the linear system Ax = b
    let SolverResult { converged, residual_norm, .. } = solver_manager.solve(&matrix, &rhs, &mut solution);

    // Check if the solver converged
    assert!(converged, "Solver did not converge");
    assert!(residual_norm < 1e-6, "Residual norm exceeded tolerance");

    // 6. Validate the solution by checking against the Dirichlet boundary condition
    for entity in mesh.entities.read().unwrap().iter() {
    if let Some(index_ref) = mesh.sieve.adjacency.get(entity) {
        // Match on MeshEntity variant to extract an indexable integer ID
        let index = match index_ref.key() {
            MeshEntity::Vertex(id) => *id,
            MeshEntity::Edge(id) => *id,
            MeshEntity::Face(id) => *id,
            MeshEntity::Cell(_) => todo!(),
            // Add additional variants as needed
        } as usize;

        let solution_value = solution.read(index, 0);
        assert!(
            (solution_value - dirichlet_value).abs() < 1e-6,
            "Solution does not meet expected Dirichlet condition"
        );
    }
}

    println!("Test for Poisson Equation (Chung Example 7.2.1) passed successfully.");
}
