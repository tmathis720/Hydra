//! Enhancements to the KSP module to introduce an interface adapter for flexible usage.
//!
//! This adds the `SolverManager` for high-level integration of solvers and preconditioners.

use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::sync::Arc;

#[derive(Debug)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

impl SolverResult {
    pub fn map_err<F, E>(self, f: F) -> Result<(), E>
    where
        F: FnOnce(String) -> E,
    {
        if self.converged {
            Ok(())
        } else {
            Err(f(format!(
                "Solver failed after {} iterations, residual norm: {}",
                self.iterations, self.residual_norm
            )))
        }
    }
}


/// KSP trait for Krylov solvers, encompassing solvers like CG and GMRES.
pub trait KSP {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}

/// Struct representing a high-level interface for managing solver configuration.
pub struct SolverManager {
    solver: Box<dyn KSP>,
    preconditioner: Option<Arc<dyn Preconditioner>>,
}

impl SolverManager {
    /// Creates a new `SolverManager` instance with a specified solver.
    ///
    /// # Arguments
    /// - `solver`: The Krylov solver to be used.
    ///
    /// # Returns
    /// A new `SolverManager` instance.
    pub fn new(solver: Box<dyn KSP>) -> Self {
        SolverManager {
            solver,
            preconditioner: None,
        }
    }

    /// Sets a preconditioner for the solver.
    ///
    /// # Arguments
    /// - `preconditioner`: The preconditioner to be used.
    pub fn set_preconditioner(&mut self, preconditioner: Arc<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }

    /// Solves a system `Ax = b` using the configured solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `a`: The system matrix `A`.
    /// - `b`: The right-hand side vector `b`.
    /// - `x`: The solution vector `x`, which will be updated with the computed solution.
    ///
    /// # Returns
    /// A `SolverResult` containing convergence information and the final residual norm.
    pub fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(a, b, x);
        }
        self.solver.solve(a, b, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::Jacobi;
    use crate::solver::cg::ConjugateGradient;
    use faer::{mat, Mat};

    #[test]
    fn test_solver_manager_with_jacobi_preconditioner() {
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];
        let b = mat![
            [1.0],
            [2.0],
        ];
        let mut x = Mat::<f64>::zeros(2, 1);

        // Initialize CG solver and solver manager
        let cg_solver = ConjugateGradient::new(100, 1e-6);
        let mut solver_manager = SolverManager::new(Box::new(cg_solver));

        // Set Jacobi preconditioner
        let jacobi_preconditioner = Arc::new(Jacobi::default());
        solver_manager.set_preconditioner(jacobi_preconditioner);

        // Solve the system
        let result = solver_manager.solve(&a, &b, &mut x);

        // Validate results
        assert!(result.converged, "Solver did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
        assert!(
            !crate::linalg::vector::traits::Vector::as_slice(&x).contains(&f64::NAN),
            "Solution contains NaN values"
        );
    }
}
