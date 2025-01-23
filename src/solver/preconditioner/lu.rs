//! LU Preconditioner implementation using Faer's LU decomposition.
//!
//! This module provides an implementation of an LU preconditioner that leverages
//! Faer's high-performance LU decomposition routines. The preconditioner is designed
//! to solve linear systems efficiently by preconditioning them using partial pivot LU decomposition.
//!
//! ## Usage
//! The `LU` struct can be instantiated with any square matrix, which it decomposes
//! using partial pivoting. It then provides an efficient method to apply the preconditioner
//! to a given right-hand side vector, solving for the preconditioned solution.

use faer::{linalg::solvers::PartialPivLu, mat::Mat};
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use faer::linalg::solvers::Solve;

/// LU preconditioner struct that holds the LU decomposition of a matrix.
///
/// The `LU` preconditioner uses partial pivot LU decomposition to enable efficient
/// solution of linear systems. It stores the decomposition internally and provides
/// methods for solving systems via the preconditioner.
pub struct LU {
    lu_decomp: PartialPivLu<f64>,
}

impl LU {
    /// Constructs a new LU preconditioner by performing an LU decomposition on the input matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A reference to a square matrix for which the LU decomposition will be computed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use faer::mat;
    /// use hydra::solver::preconditioner::LU;  // Corrected import
    ///
    /// let matrix = mat![
    ///     [4.0, 3.0],
    ///     [6.0, 3.0]
    /// ];
    ///
    /// let lu_preconditioner = LU::new(&matrix);
    /// ```
    pub fn new(matrix: &Mat<f64>) -> Self {
        let lu_decomp = PartialPivLu::new(matrix.as_ref());
        LU { lu_decomp }
    }

    /// Applies the LU preconditioner to the right-hand side vector `rhs`, storing the solution in `solution`.
    ///
    /// This function initializes a column matrix from `rhs`, then uses the stored LU decomposition
    /// to solve the system `LU * x = rhs`. The solution is then copied into the `solution` array.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector for which the solution is to be computed.
    /// * `solution` - The mutable vector where the solution will be stored.
    ///
    /// # Panics
    ///
    /// This function will panic if the dimensions of `rhs` and `solution` do not match.
    fn apply(&self, rhs: &[f64], solution: &mut [f64]) {
        let mut sol_matrix = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);

        // Solve using LU decomposition and specify `as_slice` method for Vector trait
        self.lu_decomp.solve_in_place(sol_matrix.as_mut());
        solution.copy_from_slice(&<dyn Vector<Scalar = f64>>::as_slice(&sol_matrix));
    }
}

impl Preconditioner for LU {
    /// Applies the LU preconditioner to a given vector `r` and stores the result in `z`.
    ///
    /// This implementation creates an intermediate vector, applies the LU preconditioner,
    /// and then populates `z` with the solution.
    ///
    /// # Arguments
    ///
    /// * `_a` - The matrix, not used directly in this preconditioner.
    /// * `r` - The vector to which the preconditioner is applied.
    /// * `z` - The vector where the preconditioned result is stored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use faer::mat;
    /// use hydra::solver::preconditioner::Preconditioner;
    /// use hydra::solver::preconditioner::LU;
    ///
    /// let a = mat![
    ///     [4.0, 3.0],
    ///     [6.0, 3.0]
    /// ];
    /// let lu = LU::new(&a);
    /// let r = vec![5.0, 3.0];
    /// let mut z = vec![0.0, 0.0];
    /// lu.apply(&a, &r, &mut z); // Updated to pass `a` as first argument
    /// ```
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let mut intermediate = vec![0.0; r.len()];
        self.apply(r.as_slice(), &mut intermediate);
        for i in 0..z.len() {
            z.set(i, intermediate[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    const TOLERANCE: f64 = 1e-4;

    /// Tests that the LU preconditioner with an identity matrix
    /// returns the input vector unchanged.
    #[test]
    fn test_lu_preconditioner_identity() {
        let identity = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        let lu_preconditioner = LU::new(&identity);
        lu_preconditioner.apply(&r, &mut z);

        println!("Input vector r: {:?}", r);
        println!("Output solution vector z: {:?}", z);
        assert_eq!(z, r, "Solution vector z should match the input vector r for the identity matrix.");
    }

    /// Tests that the LU preconditioner works on a simple 2x2 matrix.
    #[test]
    fn test_lu_preconditioner_simple() {
        let matrix = mat![
            [4.0, 3.0],
            [6.0, 3.0]
        ];
        let r = vec![10.0, 12.0];
        let mut z = vec![0.0; 2];

        let lu_preconditioner = LU::new(&matrix);
        lu_preconditioner.apply(&r, &mut z);

        let expected_z = vec![1.0, 2.0];
        println!("Expected solution vector: {:?}", expected_z);
        println!("Computed solution vector z: {:?}", z);
        for (computed, expected) in z.iter().zip(expected_z.iter()) {
            assert!(
                (computed - expected).abs() < TOLERANCE,
                "Computed solution {:?} does not match expected {:?} within tolerance",
                z,
                expected_z
            );
        }
    }

    /// Tests that the LU preconditioner behaves correctly on a non-trivial 3x3 system.
    #[test]
    fn test_lu_preconditioner_non_trivial() {
        let matrix = mat![
            [3.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 8.0, 8.0];
        let mut z = vec![0.0; 3];

        let lu_preconditioner = LU::new(&matrix);
        println!("Performing LU decomposition...");
        
        // Manually solving for reference, using exact calculation for expected_z
        // Expected solution: [1.0, 2.0, 3.0]
        let expected_z = vec![1.0, 2.0, 3.0];

        // Apply LU preconditioner and print decomposed values for analysis
        lu_preconditioner.apply(&r, &mut z);

        println!("Input matrix:\n{:?}", matrix);
        println!("Input RHS vector r: {:?}", r);
        println!("Expected solution vector: {:?}", expected_z);
        println!("Computed solution vector z: {:?}", z);
        
        // Detailed per-element comparison with expected values
        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            println!("Index {}: computed = {}, expected = {}", i, computed, expected);
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests the behavior of the LU preconditioner with a singular matrix (checking for NaN in result).
    #[test]
    fn test_lu_preconditioner_singular() {
        let singular_matrix = mat![
            [1.0, 2.0],
            [2.0, 4.0]
        ];
        let r = vec![3.0, 6.0];
        let mut z = vec![0.0; 2];

        println!("Testing with singular matrix:\n{:?}", singular_matrix);
        println!("RHS vector r: {:?}", r);

        // Attempt to apply LU decomposition on a singular matrix
        let lu_preconditioner = LU::new(&singular_matrix);
        lu_preconditioner.apply(&r, &mut z);

        println!("Resulting solution vector z: {:?}", z);
        assert!(
            z.iter().any(|&value| value.is_nan()),
            "Expected NaN in solution vector for singular matrix, but got {:?}",
            z
        );
    }
}
