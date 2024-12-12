use crate::{
    input_output::mmio, 
    interface_adapters::{matrix_adapter::MatrixAdapter, 
        vector_adapter::VectorAdapter}, 
    linalg::vector::vector_builder::VectorOperations, 
    solver::ksp::{SolverManager, SolverResult, KSP}
};
use std::sync::Arc;

/// Interface adapter to solve systems from MatrixMarket files.
pub struct SystemSolver;

impl SystemSolver {
    /// Solves the linear system using a specified solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `file_path`: Path to the MatrixMarket file containing the matrix data.
    /// - `solver`: A mutable reference to a solver implementing the `KSP` trait.
    /// - `preconditioner_factory`: Optional function to create a preconditioner (can be `None`).
    ///
    /// # Returns
    /// - `Result<(), Box<dyn std::error::Error>>`: `Ok` if the solution was successful, or an error.
    pub fn solve_from_file_with_solver<P, S>(
        file_path: P,
        solver: S,
        preconditioner_factory: Option<Box<dyn Fn() -> Arc<dyn crate::solver::preconditioner::Preconditioner>>>,
    ) -> Result<SolverResult, Box<dyn std::error::Error>>
    where
        P: AsRef<std::path::Path>,
        S: KSP + 'static,
    {
        // Parse the MatrixMarket file
        let (rows, cols, _nonzeros, row_indices, col_indices, values) = mmio::read_matrix_market(file_path)?;

        // Build the matrix
        let mut matrix = MatrixAdapter::new_dense_matrix(rows, cols);
        for ((&row, &col), &value) in row_indices.iter().zip(&col_indices).zip(&values) {
            MatrixAdapter::set_element(&mut matrix, row, col, value);
        }

        // Create vectors for the RHS and solution
        let mut rhs = VectorAdapter::new_dense_vector(cols);
        let mut solution = VectorAdapter::new_dense_vector(cols);

        // Initialize the RHS (example: set all to 1.0 for testing)
        for i in 0..rhs.size() {
            VectorAdapter::set_element(&mut rhs, i, 1.0);
        }

        // Configure the solver manager
        let mut solver_manager = SolverManager::new(Box::new(solver));

        // Set the preconditioner if provided
        if let Some(preconditioner_fn) = preconditioner_factory {
            let preconditioner = preconditioner_fn();
            solver_manager.set_preconditioner(preconditioner);
        }

        // Solve the system
        let result = solver_manager.solve(&matrix, &rhs, &mut solution);

        // Handle the results
        if result.converged {
            println!("Solver converged in {} iterations.", result.iterations);
            println!("Residual norm: {}", result.residual_norm);
        } else {
            eprintln!("Solver failed to converge. Residual norm: {}", result.residual_norm);
        }

        Ok(result) // Return the SolverResult instead of ()
    }
}



#[cfg(test)]
mod simple_tests {
    use super::*;
    use crate::solver::{gmres::GMRES, preconditioner::{Preconditioner, PreconditionerFactory}};
    use faer::mat::Mat;

    fn test_gmres_with_ilu_preconditioner(matrix_file: &str, rhs_file: &str) {
        // Parse the matrix and RHS from the files
        let (rows, cols, _, row_indices, col_indices, values) =
            mmio::read_matrix_market(matrix_file).expect("Failed to read matrix file");
        let (_, _, _, row_indices_rhs, _col_indices_rhs, values_rhs) =
            mmio::read_matrix_market(rhs_file).expect("Failed to read RHS MatrixMarket file");
    
        // Create the matrix
        let mut dense_matrix = Mat::zeros(rows, cols);
        for ((&row, &col), &value) in row_indices.iter().zip(&col_indices).zip(&values) {
            dense_matrix.write(row, col, value);
        }
    
        // Create the RHS vector
        let num_rows_rhs = row_indices_rhs.iter().max().unwrap_or(&0) + 1;
        let mut rhs = VectorAdapter::new_dense_vector(num_rows_rhs);
        for (&row, &value) in row_indices_rhs.iter().zip(&values_rhs) {
            VectorAdapter::set_element(&mut rhs, row, value);
        }
    
        // Use a boxed closure for the ILU preconditioner
        let preconditioner_factory: Box<dyn Fn() -> Arc<dyn Preconditioner>> =
            Box::new(move || PreconditionerFactory::create_ilu(&dense_matrix));
    
        // Solve the system
        let gmres_solver = GMRES::new(500, 1e-6, 250); // Initialize GMRES solver
        let result = SystemSolver::solve_from_file_with_solver(
            matrix_file,
            gmres_solver,
            Some(preconditioner_factory),
        );
    
        // Assert that the solver succeeded
        assert!(result.is_ok(), "Solver failed with error: {:?}", result.err());
    
        // Additional assertions can verify the solution's accuracy and convergence
        let solution_result = result.unwrap();
        assert!(
            solution_result.converged,
            "Solver did not converge. Residual norm: {}",
            solution_result.residual_norm
        );
        assert!(
            solution_result.residual_norm <= 1e-6,
            "Residual norm is too high: {}",
            solution_result.residual_norm
        );
        println!(
            "Solver converged in {} iterations with residual norm: {}",
            solution_result.iterations,
            solution_result.residual_norm
        );
    }
    

    #[test]
    fn test_solve_system_e05r0000() {
        test_gmres_with_ilu_preconditioner(
            "inputs/matrix/e05r0000/e05r0000.mtx",
            "inputs/matrix/e05r0000/e05r0000_rhs1.mtx",
        );
    }

    #[test]
    fn test_solve_system_e05r0300() {
        test_gmres_with_ilu_preconditioner(
            "inputs/matrix/e05r0300/e05r0300.mtx",
            "inputs/matrix/e05r0300/e05r0300_rhs1.mtx",
        );
    }

/*     #[test]
    fn test_solve_system_e30r0000() {
        test_gmres_with_ilu_preconditioner(
            "inputs/matrix/e30r0000/e30r0000.mtx",
            "inputs/matrix/e30r0000/e30r0000_rhs1.mtx",
        );
    }

    #[test]
    fn test_solve_system_e30r1000() {
        test_gmres_with_ilu_preconditioner(
            "inputs/matrix/e30r1000/e30r1000.mtx",
            "inputs/matrix/e30r1000/e30r1000_rhs1.mtx",
        );
    }

    #[test]
    fn test_solve_system_e30r5000() {
        test_gmres_with_ilu_preconditioner(
            "inputs/matrix/e30r5000/e30r5000.mtx",
            "inputs/matrix/e30r5000/e30r5000_rhs1.mtx",
        );
    } */
}



#[cfg(test)]
mod options_tests {
    use faer::Mat;

    use super::*;
    use crate::solver::gmres::GMRES;
    use crate::solver::cg::ConjugateGradient;
    use crate::solver::preconditioner::PreconditionerFactory;

    const MATRIX_FILE: &str = "inputs/matrix/e05r0000/e05r0000.mtx";
    const _RHS_FILE: &str = "inputs/matrix/e05r0000/e05r0000_rhs1.mtx";
    const TOL: f64 = 1e-6;

    fn validate_solver_result(result: Result<SolverResult, Box<dyn std::error::Error>>) {
        assert!(result.is_ok(), "Solver failed with error: {:?}", result.err());
        let solution_result = result.unwrap();
        assert!(
            solution_result.converged,
            "Solver did not converge. Residual norm: {}",
            solution_result.residual_norm
        );
        assert!(
            solution_result.residual_norm <= TOL,
            "Residual norm is too high: {}",
            solution_result.residual_norm
        );
        println!(
            "Solver converged in {} iterations with residual norm: {}",
            solution_result.iterations,
            solution_result.residual_norm
        );
    }

    #[test]
    fn test_gmres_with_jacobi_preconditioner() {
        let gmres_solver = GMRES::new(500, TOL, 500);
        let result = SystemSolver::solve_from_file_with_solver(
            MATRIX_FILE,
            gmres_solver,
            Some(Box::new(PreconditionerFactory::create_jacobi)),
        );
        validate_solver_result(result);
    }

    #[test]
    fn test_gmres_with_lu_preconditioner() {
        let gmres_solver = GMRES::new(500, TOL, 500);
        let result = SystemSolver::solve_from_file_with_solver(
            MATRIX_FILE,
            gmres_solver,
            Some(Box::new(|| PreconditionerFactory::create_lu(&Mat::identity(236, 236)))),
        );
        validate_solver_result(result);
    }

    #[test]
    fn test_cg_no_preconditioner() {
        let cg_solver = ConjugateGradient::new(500, TOL);
        let result = SystemSolver::solve_from_file_with_solver(MATRIX_FILE, cg_solver, None);
        validate_solver_result(result);
    }
}
