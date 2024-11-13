// src/use_cases/matrix_construction.rs

use crate::interface_adapters::matrix_adapter::MatrixAdapter;
use crate::linalg::matrix::{MatrixOperations, ExtendedMatrixOperations};
use faer::Mat;

/// Constructs and initializes a matrix for simulation.
/// Provides functions for creating, resizing, and initializing matrices.
pub struct MatrixConstruction;

impl MatrixConstruction {
    /// Constructs a dense matrix with specified dimensions and fills it with zeros.
    pub fn build_zero_matrix(rows: usize, cols: usize) -> Mat<f64> {
        MatrixAdapter::new_dense_matrix(rows, cols)
    }

    /// Initializes a matrix with a specific value.
    /// This can be useful for setting initial conditions for simulations.
    pub fn initialize_matrix_with_value<T: MatrixOperations>(matrix: &mut T, value: f64) {
        let (rows, cols) = matrix.size();
        for row in 0..rows {
            for col in 0..cols {
                MatrixAdapter::set_element(matrix, row, col, value);
            }
        }
    }

    /// Resizes a matrix to new dimensions, maintaining existing data if possible.
    pub fn resize_matrix<T: ExtendedMatrixOperations>(matrix: &mut T, new_rows: usize, new_cols: usize) {
        MatrixAdapter::resize_matrix(matrix, new_rows, new_cols);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_zero_matrix() {
        let rows = 4;
        let cols = 3;
        let matrix = MatrixConstruction::build_zero_matrix(rows, cols);

        assert_eq!(matrix.nrows(), rows, "Matrix row count should match specified rows.");
        assert_eq!(matrix.ncols(), cols, "Matrix column count should match specified cols.");
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.read(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    #[test]
    fn test_initialize_matrix_with_value() {
        let mut matrix = MatrixConstruction::build_zero_matrix(3, 3);
        let init_value = 5.0;
        MatrixConstruction::initialize_matrix_with_value(&mut matrix, init_value);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert_eq!(
                    matrix.read(i, j),
                    init_value,
                    "Each matrix element should be initialized to the specified value."
                );
            }
        }
    }

    #[test]
    fn test_resize_matrix() {
        let mut matrix = MatrixConstruction::build_zero_matrix(2, 2);
        matrix.write(0, 0, 1.0);
        matrix.write(1, 1, 2.0);
        
        MatrixConstruction::resize_matrix(&mut matrix, 3, 3);

        assert_eq!(matrix.nrows(), 3, "Matrix should have 3 rows after resizing.");
        assert_eq!(matrix.ncols(), 3, "Matrix should have 3 columns after resizing.");
        assert_eq!(matrix.read(0, 0), 1.0, "Original data should be preserved.");
        assert_eq!(matrix.read(1, 1), 2.0, "Original data should be preserved.");
        assert_eq!(matrix.read(2, 2), 0.0, "New elements should be initialized to zero.");
    }
}
