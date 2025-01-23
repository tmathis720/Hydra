use crate::linalg::matrix::{Matrix, traits::MatrixOperations};
use crate::solver::preconditioner::Preconditioner;
use faer::Mat; // Example using faer for dense matrix support.

pub struct MatrixBuilder;

impl MatrixBuilder {
    /// Builds a matrix of the specified type with given initial size.
    pub fn build_matrix<T: MatrixOperations>(rows: usize, cols: usize) -> T {
        let matrix = T::construct(rows, cols);
        matrix
    }

    /// Builds a dense matrix with faer's `Mat` structure.
    pub fn build_dense_matrix(rows: usize, cols: usize) -> Mat<f64> {
        Mat::<f64>::zeros(rows, cols)
    }

    /// Resizes the provided matrix dynamically while maintaining memory safety.
    pub fn resize_matrix<T: MatrixOperations + ExtendedMatrixOperations>(
        matrix: &mut T,
        new_rows: usize,
        new_cols: usize,
    ) {
        matrix.resize(new_rows, new_cols);
    }

    /// Demonstrates matrix compatibility with preconditioners by applying a preconditioner.
    pub fn apply_preconditioner<P: Preconditioner>(
        preconditioner: &P,
        matrix: &dyn Matrix<Scalar = f64>,
    ) {
        let input_vector = vec![0.0; matrix.ncols()];
        let mut result_vector = vec![0.0; matrix.nrows()];
        
        preconditioner.apply(matrix, &input_vector, &mut result_vector);
    }
}

pub trait ExtendedMatrixOperations: MatrixOperations {
    /// Dynamically resizes the matrix.
    fn resize(&mut self, new_rows: usize, new_cols: usize);
}

impl ExtendedMatrixOperations for Mat<f64> {
    fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let mut new_matrix = Mat::<f64>::zeros(new_rows, new_cols);
        for j in 0..usize::min(self.ncols(), new_cols) {
            for i in 0..usize::min(self.nrows(), new_rows) {
                new_matrix[(i, j)] = self[(i, j)];
            }
        }
        *self = new_matrix;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix::traits::{MatrixOperations, Matrix};
    use crate::Vector;

    #[test]
    fn test_build_dense_matrix() {
        let rows = 3;
        let cols = 3;
        let matrix = MatrixBuilder::build_dense_matrix(rows, cols);

        assert_eq!(matrix.nrows(), rows, "Number of rows should match.");
        assert_eq!(matrix.ncols(), cols, "Number of columns should match.");

        // Verify that the matrix is initialized with zeros.
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix[(i, j)], 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    #[test]
    fn test_build_matrix_generic() {
        struct DummyMatrix {
            data: Vec<Vec<f64>>,
            rows: usize,
            cols: usize,
        }

        impl MatrixOperations for DummyMatrix {
            fn construct(rows: usize, cols: usize) -> Self {
                DummyMatrix {
                    data: vec![vec![0.0; cols]; rows],
                    rows,
                    cols,
                }
            }
            fn size(&self) -> (usize, usize) {
                (self.rows, self.cols)
            }
            
            fn set(&mut self, row: usize, col: usize, value: f64) {
                self.data[row][col] = value;
            }
            
            fn get(&self, row: usize, col: usize) -> f64 {
                self.data[row][col]
            }
        }

        let rows = 4;
        let cols = 5;
        let matrix = MatrixBuilder::build_matrix::<DummyMatrix>(rows, cols);

        assert_eq!(matrix.size(), (rows, cols), "Matrix size should match the specified dimensions.");

        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.get(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    #[test]
    fn test_resize_matrix() {
        let mut matrix = MatrixBuilder::build_dense_matrix(2, 2);
        matrix[(0, 0)] = 1.0;
        matrix[(0, 1)] = 2.0;
        matrix[(1, 0)] = 3.0;
        matrix[(1, 1)] = 4.0;

        MatrixBuilder::resize_matrix(&mut matrix, 3, 3);

        assert_eq!(matrix.nrows(), 3, "Matrix should have 3 rows after resizing.");
        assert_eq!(matrix.ncols(), 3, "Matrix should have 3 columns after resizing.");

        let expected_values = vec![
            vec![1.0, 2.0, 0.0],
            vec![3.0, 4.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix[(i, j)], expected_values[i][j], "Matrix data mismatch at ({}, {}).", i, j);
            }
        }
    }

    #[test]
    fn test_apply_preconditioner() {
        struct DummyPreconditioner;
        impl Preconditioner for DummyPreconditioner {
            fn apply(
                &self,
                _a: &dyn Matrix<Scalar = f64>,
                _r: &dyn Vector<Scalar = f64>,
                z: &mut dyn Vector<Scalar = f64>,
            ) {
                for i in 0..z.len() {
                    z.set(i, 1.0);
                }
            }
        }

        let matrix = MatrixBuilder::build_dense_matrix(2, 2);
        let preconditioner = DummyPreconditioner;
        let input_vector = vec![0.5, 0.5];
        let mut result_vector = vec![0.0, 0.0];

        preconditioner.apply(&matrix, &input_vector, &mut result_vector);

        for val in result_vector.iter() {
            assert_eq!(*val, 1.0, "Each element in the result vector should be 1.0 after applying the preconditioner.");
        }
    }
}
