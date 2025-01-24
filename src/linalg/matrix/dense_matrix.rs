// src/linalg/matrix/dense_matrix.rs

use crate::linalg::matrix::traits::{Matrix, MatrixOperations, ExtendedMatrixOperations};
use crate::linalg::Vector;

/// A simple dense matrix implementation using a contiguous `Vec<f64>` for storage.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>, // Row-major storage
}

impl DenseMatrix {
    /// Creates a new dense matrix with the given dimensions, initialized to zeros.
    ///
    /// # Parameters
    /// - `rows`: Number of rows.
    /// - `cols`: Number of columns.
    ///
    /// # Returns
    /// A new `DenseMatrix` instance.
    pub fn new(rows: usize, cols: usize) -> Self {
        DenseMatrix {
            nrows: rows,
            ncols: cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Computes the index in the underlying vector for the given `(row, col)` indices.
    ///
    /// # Parameters
    /// - `row`: Row index.
    /// - `col`: Column index.
    ///
    /// # Returns
    /// The flat index in the `data` vector.
    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.nrows, "Row index out of bounds.");
        assert!(col < self.ncols, "Column index out of bounds.");
        row * self.ncols + col
    }
}

/// Implement the `Matrix` trait for `DenseMatrix`.
impl Matrix for DenseMatrix {
    type Scalar = f64;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        assert_eq!(x.len(), self.ncols, "Vector size mismatch.");
        assert_eq!(y.len(), self.nrows, "Vector size mismatch.");

        for i in 0..self.nrows {
            let mut sum = 0.0;
            for j in 0..self.ncols {
                sum += self.data[self.index(i, j)] * x.get(j);
            }
            y.set(i, sum);
        }
    }

    fn get(&self, i: usize, j: usize) -> Self::Scalar {
        self.data[self.index(i, j)]
    }

    fn trace(&self) -> Self::Scalar {
        let min_dim = usize::min(self.nrows, self.ncols);
        (0..min_dim).map(|i| self.data[self.index(i, i)]).sum()
    }

    fn frobenius_norm(&self) -> Self::Scalar {
        self.data.iter().map(|&val| val * val).sum::<f64>().sqrt()
    }

    fn as_slice(&self) -> Box<[Self::Scalar]> {
        self.data.clone().into_boxed_slice()
    }

    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]> {
        // Note: Returning a mutable slice as boxed requires cloning
        self.data.clone().into_boxed_slice()
    }
}

/// Implement the `MatrixOperations` trait for `DenseMatrix`.
impl MatrixOperations for DenseMatrix {
    fn construct(rows: usize, cols: usize) -> Self {
        DenseMatrix::new(rows, cols)
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        let index = self.index(row, col);
        self.data[index] = value;
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)]
    }

    fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// Implement the `ExtendedMatrixOperations` trait for `DenseMatrix`.
impl ExtendedMatrixOperations for DenseMatrix {
    fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let mut new_data = vec![0.0; new_rows * new_cols];
        let min_rows = usize::min(self.nrows, new_rows);
        let min_cols = usize::min(self.ncols, new_cols);

        for i in 0..min_rows {
            for j in 0..min_cols {
                let old_index = self.index(i, j);
                let new_index = i * new_cols + j;
                new_data[new_index] = self.data[old_index];
            }
        }

        self.nrows = new_rows;
        self.ncols = new_cols;
        self.data = new_data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matrix_creation() {
        let matrix = DenseMatrix::new(3, 4);
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 4);
        assert_eq!(matrix.data.len(), 12);
        assert!(matrix.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dense_matrix_set_get() {
        let mut matrix = DenseMatrix::new(3, 3);
        matrix.set(1, 1, 42.0);
        assert_eq!(MatrixOperations::get(&matrix, 1, 1), 42.0);
    }

    #[test]
    fn test_dense_matrix_resize() {
        let mut matrix = DenseMatrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 4.0);

        matrix.resize(3, 3);
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(MatrixOperations::get(&matrix, 0, 0), 1.0);
        assert_eq!(MatrixOperations::get(&matrix, 1, 1), 4.0);
        assert_eq!(MatrixOperations::get(&matrix, 2, 2), 0.0);
    }

    #[test]
    fn test_dense_matrix_trace() {
        let mut matrix = DenseMatrix::new(3, 3);
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);
        assert_eq!(matrix.trace(), 6.0);
    }

    #[test]
    fn test_dense_matrix_frobenius_norm() {
        let mut matrix = DenseMatrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let norm = (1.0f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((matrix.frobenius_norm() - norm).abs() < 1e-10);
    }
}
