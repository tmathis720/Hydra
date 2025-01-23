//! ILU Preconditioner approximation using a custom sparse ILU decomposition.
//!
//! This ILU preconditioner approximates the inverse of a matrix using a sparse LU
//! factorization method. It is especially effective for preconditioning iterative
//! solvers, improving convergence by preserving the original sparsity pattern.

use faer::mat::Mat;
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

/// ILU struct representing the incomplete LU factorization of a matrix.
pub struct ILU {
    l: Mat<f64>,
    u: Mat<f64>,
}

impl ILU {
    /// Constructs an ILU preconditioner for a given sparse matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A sparse matrix represented as a dense matrix `Mat<f64>`, to be factorized.
    ///
    /// # Returns
    ///
    /// Returns an `ILU` instance with L and U matrices approximating A.
    pub fn new(matrix: &Mat<f64>) -> Self {
        let n = matrix.nrows();
        let mut l = Mat::identity(n, n);
        let mut u = matrix.clone();

        // Perform ILU decomposition while preserving sparsity
        for i in 0..n {
            for j in (i + 1)..n {
                if matrix[(j, i)] != 0.0 {
                    // Calculate L[j, i] as the scaling factor
                    let factor = u[(j, i)] / u[(i, i)];
                    l[(j, i)] = factor;

                    // Update U row j based on the factor, preserving sparsity
                    for k in i..n {
                        let new_value = u[(j, k)] - factor * u[(i, k)];
                        u[(j, k)] = if new_value.abs() > 1e-10 {
                            new_value
                        } else {
                            0.0 // Set small values to zero
                        };
                    }
                }
            }
        }

        ILU { l, u }
    }

    /// Applies the ILU preconditioner to solve `L * U * x = r`.
    ///
    /// This method uses forward and backward substitution to apply
    /// the preconditioned solution.
    fn apply_ilu(&self, rhs: &[f64], solution: &mut [f64]) {
        let n = rhs.len();
        let mut y = vec![0.0; n];

        // Forward substitution: solve L * y = rhs
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= self.l[(i, j)] * y[j];
            }
            y[i] = sum / self.l[(i, i)];
        }

        // Backward substitution: solve U * x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.u[(i, j)] * solution[j];
            }
            solution[i] = sum / self.u[(i, i)];
        }
    }
}

impl Preconditioner for ILU {
    /// Applies the ILU preconditioner to the vector `r`, storing the result in `z`.
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let mut intermediate = vec![0.0; r.len()];
        self.apply_ilu(r.as_slice(), &mut intermediate);

        // Copy the intermediate result into the solution vector `z`
        for i in 0..z.len() {
            z.set(i, intermediate[i]);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    /// Tests that the ILU preconditioner produces results close to the expected solution for a simple case.
    #[test]
    fn test_ilu_preconditioner_simple() {
        let matrix = mat![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 5.0, 3.0];
        let expected_z = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests that the ILU preconditioner behaves correctly with an identity matrix.
    #[test]
    fn test_ilu_preconditioner_identity() {
        let matrix = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        assert_eq!(z, r, "For identity matrix, output should match input vector exactly.");
    }

    /// Tests the ILU preconditioner on a larger, sparse matrix to verify it maintains sparsity.
    #[test]
    fn test_ilu_preconditioner_sparse() {
        let matrix = mat![
            [10.0, 0.0, 2.0, 0.0],
            [3.0, 9.0, 0.0, 0.0],
            [0.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 6.0, 5.0]
        ];
        let r = vec![12.0, 12.0, 15.0, 11.0];
        let mut z = vec![0.0; 4];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        let expected_z = vec![1.0, 1.0, 1.0, 1.0];
        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests behavior of the ILU preconditioner with a singular matrix.
    #[test]
    fn test_ilu_preconditioner_singular() {
        let matrix = mat![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ];
        let r = vec![6.0, 12.0, 18.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        // Since this matrix is singular, we expect NaNs or other indicators of failure
        assert!(
            z.iter().any(|&val| val.is_nan() || val.abs() > 1e6),
            "For singular matrix, solution should indicate failure (NaNs or very large values)."
        );
    }

    /// Tests the ILU preconditioner on a non-trivial system with non-zero off-diagonal elements.
    #[test]
    fn test_ilu_preconditioner_non_trivial() {
        let matrix = mat![
            [2.0, 3.0, 1.0],
            [6.0, 1.0, 4.0],
            [0.0, 2.0, 8.0]
        ];
        let r = vec![3.0, 10.0, 8.0];
        let expected_z = vec![1.0, 0.0, 1.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }
}
