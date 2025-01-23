// src/interface_adapters/matrix_adapter.rs

use crate::{
    linalg::matrix::{Matrix,MatrixOperations, ExtendedMatrixOperations},
    solver::preconditioner::Preconditioner,
    Vector,
};
use faer::Mat;

/// A struct that adapts matrix operations to the Hydra environment.
pub struct MatrixAdapter;

impl MatrixAdapter {
    /// Creates a new dense matrix with specified dimensions.
    pub fn new_dense_matrix(rows: usize, cols: usize) -> Mat<f64> {
        Mat::zeros(rows, cols)
    }

    /// Resizes a matrix if the type supports resizing, using specialized handling.
    pub fn resize_matrix<T: ExtendedMatrixOperations>(matrix: &mut T, new_rows: usize, new_cols: usize) {
        matrix.resize(new_rows, new_cols);
    }

    /// Sets a specific element within the matrix.
    pub fn set_element<T: MatrixOperations>(matrix: &mut T, row: usize, col: usize, value: f64) {
        matrix.set(row, col, value);
    }

    /// Retrieves an element from the matrix.
    pub fn get_element<T: MatrixOperations>(matrix: &T, row: usize, col: usize) -> f64 {
        matrix.get(row, col)
    }

    /// Applies a preconditioner to the matrix, demonstrating compatibility with solvers.
    pub fn apply_preconditioner(
        preconditioner: &dyn Preconditioner,
        matrix: &dyn Matrix<Scalar = f64>,
        input: &dyn Vector<Scalar = f64>,
        output: &mut dyn Vector<Scalar = f64>,
    ) {
        preconditioner.apply(matrix, input, output);
    }
}

impl ExtendedMatrixOperations for Mat<f64> {
    fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let mut new_matrix = Mat::<f64>::zeros(new_rows, new_cols);
        for j in 0..usize::min(self.ncols(), new_cols) {
            for i in 0..usize::min(self.nrows(), new_rows) {
                new_matrix[(i, j)] = self[(i, j)];
            }
        }
        *self = new_matrix; // Replace the current matrix with the resized one
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dense_matrix() {
        let rows = 3;
        let cols = 3;
        let matrix = MatrixAdapter::new_dense_matrix(rows, cols);

        assert_eq!(matrix.nrows(), rows);
        assert_eq!(matrix.ncols(), cols);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_set_and_get_element() {
        let mut matrix = MatrixAdapter::new_dense_matrix(3, 3);
        MatrixAdapter::set_element(&mut matrix, 1, 1, 5.0);
        let value = MatrixAdapter::get_element(&matrix, 1, 1);

        assert_eq!(value, 5.0);
    }
}
