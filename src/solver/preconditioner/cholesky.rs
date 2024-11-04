// src/solver/preconditioner/cholesky.rs
use faer::{mat::Mat, solvers::{Cholesky, SpSolver}, Side}; // Import Side for cholesky method argument
use crate::{linalg::Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::error::Error;

/// `CholeskyPreconditioner` holds the Cholesky decomposition result to
/// precondition a system for CG methods on symmetric positive definite matrices.
pub struct CholeskyPreconditioner {
    /// The Cholesky decomposition result
    l_factor: Cholesky<f64>,
}

impl CholeskyPreconditioner {
    /// Creates a new `CholeskyPreconditioner` by computing the Cholesky decomposition
    /// of a symmetric positive definite matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A reference to the matrix to be decomposed. Must be symmetric and positive definite.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the preconditioner if successful, or an error if the decomposition fails.
    pub fn new(matrix: &Mat<f64>) -> Result<Self, Box<dyn Error>> {
        // Specify the side for Cholesky decomposition as Lower
        let l_factor = matrix.cholesky(Side::Lower).map_err(|_| "Cholesky decomposition failed")?;
        Ok(Self { l_factor })
    }

    /// Applies the preconditioner to a given right-hand side vector to produce a preconditioned solution.
    ///
    /// # Arguments
    ///
    /// * `rhs` - A reference to the right-hand side matrix (vector) for the system.
    ///
    /// # Returns
    ///
    /// * `Ok(Mat<f64>)` containing the preconditioned solution, or an error if solving fails.
    /// Applies the preconditioner to a given vector `rhs` and returns the solution.
    ///
    /// # Arguments
    /// * `rhs` - The right-hand side vector as a `Mat<f64>`.
    pub fn apply(&self, rhs: &Mat<f64>) -> Result<Mat<f64>, Box<dyn Error>> {
        Ok(self.l_factor.solve(rhs))
    }
}

impl Preconditioner for CholeskyPreconditioner {
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let rhs_mat = Mat::from_fn(r.len(), 1, |i, _| r.get(i));
        if let Ok(solution) = self.apply(&rhs_mat) {
            for i in 0..z.len() {
                z.set(i, solution[(i, 0)]);
            }
        }
    }
}

/// Example function to apply Cholesky preconditioning in the CG algorithm.
/// This function returns the preconditioned solution.
///
/// # Arguments
///
/// * `matrix` - The system matrix.
/// * `rhs` - The right-hand side vector.
///
/// # Returns
///
/// * `Result<Mat<f64>, Box<dyn Error>>` containing the solution vector or an error if any stage fails.
pub fn apply_cholesky_preconditioner(matrix: &Mat<f64>, rhs: &Mat<f64>) -> Result<Mat<f64>, Box<dyn Error>> {
    // Initialize the preconditioner
    let preconditioner = CholeskyPreconditioner::new(matrix)?;
    
    // Apply the preconditioner to obtain the preconditioned right-hand side
    preconditioner.apply(rhs)
}

// src/solver/preconditioner/cholesky.rs

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat::Mat;
    use approx::assert_relative_eq; // For floating-point comparisons

    #[test]
    fn test_cholesky_preconditioner_creation() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Attempt to create the Cholesky preconditioner
        let preconditioner = CholeskyPreconditioner::new(&matrix);
        assert!(preconditioner.is_ok(), "Failed to create Cholesky preconditioner");
    }

    #[test]
    fn test_cholesky_preconditioner_application() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Define a right-hand side vector
        let mut rhs = Mat::<f64>::zeros(2, 1);
        rhs[(0, 0)] = 1.0;
        rhs[(1, 0)] = 2.0;

        // Initialize the preconditioner
        let preconditioner = CholeskyPreconditioner::new(&matrix).expect("Preconditioner creation failed");

        // Apply the preconditioner
        let result = preconditioner.apply(&rhs).expect("Preconditioner application failed");

        // Define expected output (corrected values)
        let mut expected = Mat::<f64>::zeros(2, 1);
        expected[(0, 0)] = 0.0909091; // Corrected expected value
        expected[(1, 0)] = 0.6363636;

        // Verify results using approximate equality
        assert_relative_eq!(result[(0, 0)], expected[(0, 0)], epsilon = 1e-6);
        assert_relative_eq!(result[(1, 0)], expected[(1, 0)], epsilon = 1e-6);
    }

    #[test]
    fn test_apply_cholesky_preconditioner_function() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Define a right-hand side vector
        let mut rhs = Mat::<f64>::zeros(2, 1);
        rhs[(0, 0)] = 1.0;
        rhs[(1, 0)] = 2.0;

        // Use the apply_cholesky_preconditioner function
        let result = apply_cholesky_preconditioner(&matrix, &rhs).expect("Function application failed");

        // Define expected output (corrected values)
        let mut expected = Mat::<f64>::zeros(2, 1);
        expected[(0, 0)] = 0.0909091; // Corrected expected value
        expected[(1, 0)] = 0.6363636;

        // Verify results using approximate equality
        assert_relative_eq!(result[(0, 0)], expected[(0, 0)], epsilon = 1e-6);
        assert_relative_eq!(result[(1, 0)], expected[(1, 0)], epsilon = 1e-6);
    }
}
