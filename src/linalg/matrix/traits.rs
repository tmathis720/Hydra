// src/linalg/matrix/trait.rs

use crate::linalg::Vector;
use faer::Mat;

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
}