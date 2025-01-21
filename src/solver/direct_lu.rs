use faer::Mat;

use crate::linalg::{Matrix, Vector};
use crate::solver::ksp::{KSP, SolverResult};
use crate::solver::preconditioner::LU;

/// A direct solver using LU decomposition to solve `Ax = b`.
pub struct DirectLUSolver {
    lu_decomp: LU,
}

impl DirectLUSolver {
    /// Constructs a new `DirectLUSolver` from the given matrix.
    ///
    /// # Arguments
    /// - `a`: The system matrix `A` to decompose.
    ///
    /// # Panics
    /// Panics if the matrix `A` is not square.
    pub fn new(a: &Mat<f64>) -> Self {
        // Convert to dense matrix if necessary
        let lu = LU::new(&a);
        Self { lu_decomp: lu }
    }
}

impl KSP for DirectLUSolver {
    fn solve(
        &mut self,
        _a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        let b_slice = b.as_slice();
        let mut x_slice = vec![0.0; b.len()];

        // Solve using LU decomposition
        self.lu_decomp.apply(b_slice, &mut x_slice);

        // Copy the result back to the solution vector
        for (i, &val) in x_slice.iter().enumerate() {
            x.set(i, val);
        }

        SolverResult {
            converged: true,
            iterations: 1, // Direct solvers converge in one "iteration"
            residual_norm: 0.0, // Residual norm is assumed to be 0 due to exact solve
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    const TOLERANCE: f64 = 1e-4; // less strict tolerance for direct solver

    #[test]
    fn test_direct_lu_solver_simple_case() {
        // Define a 2x2 matrix and right-hand side vector
        let a = mat![
            [4.0, 2.0],
            [3.0, 3.0],
        ];
        let b = vec![10.0, 12.0];
        let mut x = vec![0.0; 2];

        // Expected solution
        let expected_x = vec![1.0, 3.0];

        // Create DirectLUSolver and solve
        let mut solver = DirectLUSolver::new(&a);
        let result = solver.solve(&a, &b, &mut x);

        // Validate the solution
        assert!(result.converged, "Solver did not converge");
        assert_eq!(result.iterations, 1, "Direct solver should converge in one iteration");
        for (computed, &expected) in x.iter().zip(&expected_x) {
            assert!(
                (computed - expected).abs() < TOLERANCE,
                "Computed value {} differs from expected value {}",
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_direct_lu_solver_identity_matrix() {
        // Define an identity matrix and right-hand side vector
        let a = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let b = vec![5.0, 10.0, 15.0];
        let mut x = vec![0.0; 3];

        // Create DirectLUSolver and solve
        let mut solver = DirectLUSolver::new(&a);
        let result = solver.solve(&a, &b, &mut x);

        // Validate the solution
        assert!(result.converged, "Solver did not converge");
        assert_eq!(result.iterations, 1, "Direct solver should converge in one iteration");
        assert_eq!(x, b, "Solution should match the right-hand side vector for an identity matrix");
    }

    #[test]
    fn test_direct_lu_solver_large_matrix() {
        // Define a larger 4x4 matrix and right-hand side vector
        let a = mat![
            [10.0, 2.0, 0.0, 0.0],
            [3.0, 10.0, 2.0, 0.0],
            [0.0, 3.0, 10.0, 2.0],
            [0.0, 0.0, 3.0, 10.0],
        ];
        let b = vec![12.0, 17.0, 20.0, 24.0];
        let mut x = vec![0.0; 4];

        // Expected solution
        let expected_x = vec![0.9679, 1.1603, 1.2467, 2.0260];

        // Create DirectLUSolver and solve
        let mut solver = DirectLUSolver::new(&a);
        let result = solver.solve(&a, &b, &mut x);

        // Validate the solution
        assert!(result.converged, "Solver did not converge");
        assert_eq!(result.iterations, 1, "Direct solver should converge in one iteration");
        for (computed, &expected) in x.iter().zip(&expected_x) {
            assert!(
                (computed - expected).abs() < TOLERANCE,
                "Computed value {} differs from expected value {}",
                computed,
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "Assertion failed: matrix.nrows() == matrix.ncols()\n- matrix.nrows() = 3\n- matrix.ncols() = 2")]
    fn test_direct_lu_solver_non_square_matrix() {
        // Define a non-square matrix
        let a = mat![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ];

        // Attempt to create DirectLUSolver (should panic)
        DirectLUSolver::new(&a);
    }
}
