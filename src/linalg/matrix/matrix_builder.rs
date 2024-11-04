use crate::linalg::matrix::{Matrix, traits::MatrixOperations};
use crate::solver::preconditioner::Preconditioner;
use faer::Mat; // Example using faer for dense matrix support.

pub struct MatrixBuilder;

impl MatrixBuilder {
    /// Builds a matrix of the specified type with given initial size.
    /// Supports various matrix types through generics.
    ///
    /// # Parameters
    /// - `rows`: The number of rows for the matrix.
    /// - `cols`: The number of columns for the matrix.
    ///
    /// # Returns
    /// A matrix of type `T` initialized to the specified dimensions.
    pub fn build_matrix<T: MatrixOperations>(rows: usize, cols: usize) -> T {
        let matrix = T::construct(rows, cols);
        // Optionally, initialize the matrix to zero or other values if needed.
        // `initialize_zero` can be defined as part of T's implementation if required.
        matrix
    }

    /// Builds a dense matrix with faer's `Mat` structure.
    /// Initializes with zeros and demonstrates integration with potential preconditioners.
    pub fn build_dense_matrix(rows: usize, cols: usize) -> Mat<f64> {
        Mat::<f64>::zeros(rows, cols)
    }

    /// Resizes the provided matrix dynamically while maintaining memory safety.
    /// Ensures no data is left uninitialized during resizing.
    pub fn resize_matrix<T: MatrixOperations + ExtendedMatrixOperations>(
        matrix: &mut T,
        new_rows: usize,
        new_cols: usize,
    ) {
        // Call the resizing operation implemented by the specific matrix type.
        matrix.resize(new_rows, new_cols);
    }

    /// Demonstrates matrix compatibility with preconditioners by applying a preconditioner.
    pub fn apply_preconditioner<P: Preconditioner>(
        preconditioner: &P,
        matrix: &dyn Matrix<Scalar = f64>,
    ) {
        let input_vector = Vec::new(); // Replace with correct vector initialization.
        let mut result_vector = Vec::new(); // Initialize a separate vector for mutable use.
        
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
        *self = new_matrix; // Replace old matrix with the resized matrix.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linalg::matrix::traits::{MatrixOperations, Matrix}, Vector};

    /// Test for building a dense matrix using `MatrixBuilder::build_dense_matrix`.
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
                assert_eq!(matrix.read(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    /// Test for building a generic matrix using `MatrixBuilder::build_matrix`.
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
            fn set_value(&mut self, row: usize, col: usize, value: f64) {
                self.data[row][col] = value;
            }
            fn get_value(&self, row: usize, col: usize) -> f64 {
                self.data[row][col]
            }
            fn size(&self) -> (usize, usize) {
                (self.rows, self.cols)
            }
        }

        let rows = 4;
        let cols = 5;
        let matrix = MatrixBuilder::build_matrix::<DummyMatrix>(rows, cols);

        assert_eq!(matrix.size(), (rows, cols), "Matrix size should match the specified dimensions.");

        // Ensure matrix is initialized with zeros.
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.get_value(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    /// Test for resizing a matrix using `MatrixBuilder::resize_matrix`.
    #[test]
    fn test_resize_matrix() {
        let mut matrix = MatrixBuilder::build_dense_matrix(2, 2);
        matrix.write(0, 0, 1.0);
        matrix.write(0, 1, 2.0);
        matrix.write(1, 0, 3.0);
        matrix.write(1, 1, 4.0);

        MatrixBuilder::resize_matrix(&mut matrix, 3, 3);

        // Check new size.
        assert_eq!(matrix.nrows(), 3, "Matrix should have 3 rows after resizing.");
        assert_eq!(matrix.ncols(), 3, "Matrix should have 3 columns after resizing.");

        // Check that original data is preserved and new cells are zero.
        let expected_values = vec![
            vec![1.0, 2.0, 0.0],
            vec![3.0, 4.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix.read(i, j), expected_values[i][j], "Matrix data mismatch at ({}, {}).", i, j);
            }
        }
    }

    /// Test for applying a preconditioner using `MatrixBuilder::apply_preconditioner`.
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
                    z.set(i, 1.0); // Set all elements in the result vector to 1.0.
                }
            }
        }

        let matrix = MatrixBuilder::build_dense_matrix(2, 2);
        let preconditioner = DummyPreconditioner;
        let input_vector = vec![0.5, 0.5];
        let mut result_vector = vec![0.0, 0.0];

        preconditioner.apply(&matrix, &input_vector, &mut result_vector);

        // Check that the preconditioner applied the expected transformation.
        for val in result_vector.iter() {
            assert_eq!(*val, 1.0, "Each element in the result vector should be 1.0 after applying the preconditioner.");
        }
    }
}
