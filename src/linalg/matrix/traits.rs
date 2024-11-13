// src/linalg/matrix/traits.rs

use crate::linalg::Vector;

/// Trait defining essential matrix operations (abstract over dense, sparse)
/// Define that any type implementing Matrix must be Send and Sync
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>); // y = A * x
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    fn trace(&self) -> Self::Scalar;
    fn frobenius_norm(&self) -> Self::Scalar;
    fn as_slice(&self) -> Box<[Self::Scalar]>;
    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}

/// Trait defining matrix operations for building and manipulation
pub trait MatrixOperations: Send + Sync {
    fn construct(rows: usize, cols: usize) -> Self
    where
        Self: Sized;
    fn set(&mut self, row: usize, col: usize, value: f64);
    fn get(&self, row: usize, col: usize) -> f64;
    fn size(&self) -> (usize, usize);
}

/// Extended matrix operations trait for resizing
pub trait ExtendedMatrixOperations: MatrixOperations {
    fn resize(&mut self, new_rows: usize, new_cols: usize)
    where
        Self: Sized;
}
