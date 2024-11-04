// src/linalg/matrix/trait.rs

use crate::linalg::Vector;

// Trait defining essential matrix operations (abstract over dense, sparse)
// Define that any type implementing Matrix must be Send and Sync
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>); // y = A * x
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    /// Computes the trace of the matrix (sum of diagonal elements).
    /// Returns the sum of elements where row index equals column index.
    fn trace(&self) -> Self::Scalar;

    /// Computes the Frobenius norm of the matrix.
    /// The Frobenius norm is defined as the square root of the sum of the absolute squares of its elements.
    fn frobenius_norm(&self) -> Self::Scalar;

    /// Converts the matrix to a slice of its underlying data in row-major order.
    ///
    /// # Returns
    /// A slice containing the matrix elements in row-major order.
    fn as_slice(&self) -> Box<[Self::Scalar]>;

    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}

/// Trait defining the functions for solver integration.
pub trait MatrixOperations {
    fn construct(rows: usize, cols: usize) -> Self;
    fn set_value(&mut self, row: usize, col: usize, value: f64);
    fn get_value(&self, row: usize, col: usize) -> f64;
    fn size(&self) -> (usize, usize);
}