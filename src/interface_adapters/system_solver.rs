use crate::{
    input_output::mmio, 
    interface_adapters::{matrix_adapter::MatrixAdapter, vector_adapter::VectorAdapter}, 
    solver::ksp::{SolverManager, SolverResult, KSP}
};
use std::sync::Arc;
use std::path::{Path, PathBuf};

/// Interface adapter to solve systems from MatrixMarket files.
pub struct SystemSolver;

impl SystemSolver {
    /// Solves the linear system using a specified solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `file_path`: Path to the MatrixMarket file containing the matrix data.
    /// - `solver`: A solver implementing the `KSP` trait.
    /// - `preconditioner_factory`: Optional function to create a preconditioner (can be `None`).
    ///
    /// # Returns
    /// A `SolverResult` if successful, or an error.
    pub fn solve_from_file_with_solver<P, S>(
        file_path: P,
        solver: S,
        preconditioner_factory: Option<Box<dyn Fn() -> Arc<dyn crate::solver::preconditioner::Preconditioner>>>,
    ) -> Result<SolverResult, Box<dyn std::error::Error>>
    where
        P: AsRef<std::path::Path>,
        S: KSP + 'static,
    {
        // Parse the MatrixMarket file for the matrix
        let (rows, cols, _nonzeros, row_indices, col_indices, values) = mmio::read_matrix_market(file_path.as_ref())?;

        // Build the matrix
        let mut matrix = MatrixAdapter::new_dense_matrix(rows, cols);
        for ((&row, &col), &value) in row_indices.iter().zip(&col_indices).zip(&values) {
            MatrixAdapter::set_element(&mut matrix, row, col, value);
        }

        // Determine the corresponding RHS file by appending '_rhs1' before the .mtx extension.
        let rhs_path = Self::derive_rhs_path(file_path.as_ref())?;
        let (_, _, _, row_indices_rhs, _col_indices_rhs, values_rhs) = mmio::read_matrix_market(&rhs_path)?;

        // Create the RHS vector from the parsed RHS data
        let num_rows_rhs = row_indices_rhs.iter().max().unwrap_or(&0) + 1;
        let mut rhs = VectorAdapter::new_dense_vector(num_rows_rhs);
        for (&row, &value) in row_indices_rhs.iter().zip(&values_rhs) {
            VectorAdapter::set_element(&mut rhs, row, value);
        }

        // Create the solution vector
        let mut solution = VectorAdapter::new_dense_vector(cols);

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

        Ok(result)
    }

    /// Derives the RHS file path by inserting `_rhs1` before the `.mtx` extension.
    fn derive_rhs_path(matrix_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let parent = matrix_path.parent().unwrap_or_else(|| Path::new("."));
        let file_stem = matrix_path
            .file_stem()
            .ok_or("Matrix file has no stem")?
            .to_str()
            .ok_or("Matrix file stem is not valid UTF-8")?;

        // Construct the RHS file name
        let rhs_file_name = format!("{}_rhs1.mtx", file_stem);
        Ok(parent.join(rhs_file_name))
    }
}


#[cfg(test)]
mod simple_tests {
    use super::*;
    use crate::solver::gmres::GMRES;
    use crate::solver::preconditioner::{Preconditioner, PreconditionerFactory};

    /// Validates the result of the solver against convergence and residual norm criteria.
    fn validate_solver_result(
        result: Result<SolverResult, Box<dyn std::error::Error>>,
        max_res_norm: f64,
    ) {
        assert!(result.is_ok(), "Solver failed with error: {:?}", result.err());
        let solution_result = result.unwrap();
        assert!(
            solution_result.converged,
            "Solver did not converge. Residual norm: {}",
            solution_result.residual_norm
        );
        assert!(
            solution_result.residual_norm <= max_res_norm,
            "Residual norm is too high: {}",
            solution_result.residual_norm
        );
        println!(
            "Solver converged in {} iterations with residual norm: {}",
            solution_result.iterations,
            solution_result.residual_norm
        );
    }

    fn test_gmres_with_ilu_preconditioner(matrix_file: &str) {
        // Use an ILU preconditioner factory
        // We will parse the matrix inside the preconditioner factory:
        let (rows, cols, _, row_indices, col_indices, values) =
            mmio::read_matrix_market(matrix_file).expect("Failed to read matrix file");
        let mut dense_matrix = faer::mat::Mat::zeros(rows, cols);
        for ((&row, &col), &value) in row_indices.iter().zip(&col_indices).zip(&values) {
            dense_matrix.write(row, col, value);
        }
        let preconditioner_factory: Box<dyn Fn() -> Arc<dyn Preconditioner>> =
            Box::new(move || PreconditionerFactory::create_ilu(&dense_matrix));

        // Solve the system with GMRES
        let gmres_solver = GMRES::new(1000, 1e-6, 500);
        let result = SystemSolver::solve_from_file_with_solver(
            matrix_file,
            gmres_solver,
            Some(preconditioner_factory),
        );

        validate_solver_result(result, 1e-6);
    }

    #[test]
    fn test_solve_system_e05r0000() {
        test_gmres_with_ilu_preconditioner("inputs/matrix/e05r0000/e05r0000.mtx");
    }

    #[test]
    fn test_solve_system_e05r0300() {
        test_gmres_with_ilu_preconditioner("inputs/matrix/e05r0300/e05r0300.mtx");
    }

    // For larger cases (e30r0000, e30r1000, e30r5000), consider enabling tests below:
/*     #[test]
    fn test_solve_system_e30r0000() {
        test_gmres_with_ilu_preconditioner("inputs/matrix/e30r0000/e30r0000.mtx");
    } */

    // #[test]
    // fn test_solve_system_e30r1000() {
    //     test_gmres_with_ilu_preconditioner("inputs/matrix/e30r1000/e30r1000.mtx");
    // }

    // #[test]
    // fn test_solve_system_e30r5000() {
    //     test_gmres_with_ilu_preconditioner("inputs/matrix/e30r5000/e30r5000.mtx");
    // }
}

#[cfg(test)]
mod options_tests {
    use faer::Mat;

    use super::*;
    use crate::solver::gmres::GMRES;
    use crate::solver::cg::ConjugateGradient;
    use crate::solver::preconditioner::{Preconditioner, PreconditionerFactory};
    use std::sync::Arc;
    use crate::input_output::mmio;
    use crate::solver::ksp::SolverResult;

    const BASE_MATRIX_FILE: &str = "inputs/matrix/e05r0000/e05r0000.mtx";
    const TOL: f64 = 1e-5;

    /// Validates the result of the solver against:
    /// 1) Successful convergence (`solution_result.converged == true`)
    /// 2) A maximum residual norm `<= max_res_norm`.
    /// This ensures approximate equality on floating-point results rather than
    /// expecting exact bitwise matches.
    fn validate_solver_result(
        result: Result<SolverResult, Box<dyn std::error::Error>>,
        max_res_norm: f64,
    ) {
        // Check that the solver call did not return an error
        assert!(result.is_ok(), "Solver failed with error: {:?}", result.err());
        
        // Unwrap the solver's result
        let solution_result = result.unwrap();
        
        // Check it converged
        assert!(
            solution_result.converged,
            "Solver did not converge. Residual norm: {}",
            solution_result.residual_norm
        );

        // Approximate check: residual norm <= max_res_norm
        // (Solution #1: no exact comparison, only a tolerance-based check)
        assert!(
            solution_result.residual_norm <= max_res_norm,
            "Residual norm is too high: {} > {}",
            solution_result.residual_norm,
            max_res_norm
        );

        println!(
            "Solver converged in {} iterations with residual norm: {:.3e}",
            solution_result.iterations,
            solution_result.residual_norm
        );
    }

    #[test]
    fn test_gmres_with_jacobi_preconditioner() {
        let gmres_solver = GMRES::new(500, TOL, 500);
        let result = SystemSolver::solve_from_file_with_solver(
            BASE_MATRIX_FILE,
            gmres_solver,
            Some(Box::new(PreconditionerFactory::create_jacobi)),
        );
        validate_solver_result(result, TOL);
    }

    #[test]
    fn test_cg_no_preconditioner() {
        let cg_solver = ConjugateGradient::new(500, TOL);
        let result = SystemSolver::solve_from_file_with_solver(BASE_MATRIX_FILE, cg_solver, None);
        validate_solver_result(result, TOL);
    }

    #[test]
    fn test_gmres_with_amg_preconditioner() {
        // For a difficult matrix from the drivcav series, use AMG preconditioner.
        // We'll demonstrate the same approximate checks here.
        
        const AMG_MATRIX_FILE: &str = "inputs/matrix/e05r0300/e05r0300.mtx";
        const AMG_TOL: f64 = 1e-5;

        // Create a GMRES solver configured for this problem
        let gmres_solver = GMRES::new(500, AMG_TOL, 250);

        // Increase max_levels and tweak coarsening_threshold as needed
        let max_levels = 5;
        let coarsening_threshold = 0.01;

        // Parse matrix for AMG preconditioner construction
        let (rows, cols, _, row_indices, col_indices, values) =
            mmio::read_matrix_market(AMG_MATRIX_FILE).expect("Failed to read matrix file");
        let mut dense_matrix = Mat::zeros(rows, cols);
        for ((&row, &col), &value) in row_indices.iter().zip(&col_indices).zip(&values) {
            dense_matrix.write(row, col, value);
        }

        // Create an AMG preconditioner factory
        let preconditioner_factory: Box<dyn Fn() -> Arc<dyn Preconditioner>> =
            Box::new(move || PreconditionerFactory::create_amg(&dense_matrix, max_levels, coarsening_threshold));

        // Solve the system from the given MatrixMarket file
        let result = SystemSolver::solve_from_file_with_solver(
            AMG_MATRIX_FILE,
            gmres_solver,
            Some(preconditioner_factory)
        );

        // Now we do an approximate check of convergence and residual norm
        // (Solution #3: confirm the solver converges within `AMG_TOL`.)
        validate_solver_result(result, AMG_TOL);
    }
}
