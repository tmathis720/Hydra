// src/linalg/matrix.rs

use super::vector::Vector;
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

// Implement Matrix trait for faer_core::Mat
impl Matrix for Mat<f64> {
    type Scalar = f64;

    fn nrows(&self) -> usize {
        self.nrows() // Return the number of rows in the matrix
    }

    fn ncols(&self) -> usize {
        self.ncols() // Return the number of columns in the matrix
    }

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        // Perform matrix-vector multiplication
        // Utilizing optimized `faer` routines for better performance
        // Assuming that `faer` provides an optimized mat_vec, but since it's not used here,
        // we keep the manual implementation as per original code

        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self.read(i, j) * x.get(j);
            }
            y.set(i, sum);
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.read(i, j) // Read the matrix element at position (i, j)
    }

    fn trace(&self) -> f64 {
        let min_dim = usize::min(self.nrows(), self.ncols());
        let mut trace_sum = 0.0;
        for i in 0..min_dim {
            trace_sum += self.read(i, i);
        }
        trace_sum
    }

    fn frobenius_norm(&self) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.read(i, j);
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use std::sync::Arc;

    /// Helper function to create a faer::Mat<f64> from a 2D Vec.
    fn create_faer_matrix(data: Vec<Vec<f64>>) -> Mat<f64> {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };
        let mut mat = Mat::zeros(nrows, ncols);

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                mat.write(i, j, val);
            }
        }

        mat
    }

    /// Helper function to create a faer::Mat<f64> as a column vector.
    fn create_faer_vector(data: Vec<f64>) -> Mat<f64> {
        let nrows = data.len();
        let ncols = 1;
        let mut mat = Mat::zeros(nrows, ncols);

        for (i, &val) in data.iter().enumerate() {
            mat.write(i, 0, val);
        }

        mat
    }

    #[test]
    fn test_nrows_ncols() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        assert_eq!(mat_ref.nrows(), 3);
        assert_eq!(mat_ref.ncols(), 3);
    }

    #[test]
    fn test_get() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data.clone());
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_eq!(mat_ref.get(i, j), val);
            }
        }
    }

    #[test]
    fn test_mat_vec_with_vec_f64() {
        // Define matrix A
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using Vec<f64>
        let x = vec![1.0, 0.0, -1.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        // y[0] = 1*1 + 2*0 + 3*(-1) = 1 - 3 = -2
        // y[1] = 4*1 + 5*0 + 6*(-1) = 4 - 6 = -2
        // y[2] = 7*1 + 8*0 + 9*(-1) = 7 - 9 = -2
        let expected = vec![-2.0, -2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_with_faer_vector() {
        // Define matrix A
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using faer::Mat<f64> as a column vector
        let x_data = vec![1.0, 0.0, -1.0];
        let x = create_faer_vector(x_data);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i, 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i, 0),
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_identity_with_vec_f64() {
        // Define an identity matrix
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using Vec<f64>
        let x = vec![5.0, -3.0, 2.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result is x itself
        let expected = vec![5.0, -3.0, 2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_zero_matrix_with_faer_vector() {
        // Define a zero matrix
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using faer::Mat<f64> as a column vector
        let x = create_faer_vector(vec![3.0, 4.0]);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result is a zero vector
        let expected = vec![0.0, 0.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i , 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i , 0),
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_non_square_matrix_with_vec_f64() {
        // Define a non-square matrix (2x3)
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x (size 3) using Vec<f64>
        let x = vec![1.0, 0.0, -1.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y (size 2) using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_non_square_matrix_with_faer_vector() {
        // Define a non-square matrix (2x3)
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x (size 3) using faer::Mat<f64> as a column vector
        let x = create_faer_vector(vec![1.0, 0.0, -1.0]);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y (size 2) using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i , 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i , 0),
                val
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_row() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds row should panic
        mat_ref.get(2, 1);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_column() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds column should panic
        mat_ref.get(1, 2);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref = Arc::new(mat);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let mat_clone = Arc::clone(&mat_ref);
                thread::spawn(move || {
                    // Define vector x using Vec<f64>
                    let x = vec![1.0, 0.0, -1.0];
                    let x_ref: &dyn Vector<Scalar = f64> = &x;

                    // Initialize vector y using Vec<f64>
                    let mut y = vec![0.0; mat_clone.nrows()];
                    let mut y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

                    // Perform y = A * x
                    mat_clone.mat_vec(x_ref, y_ref);

                    // Expected result is [-2.0, -2.0, -2.0]
                    let expected = vec![-2.0, -2.0, -2.0];

                    for (i, &val) in expected.iter().enumerate() {
                        assert!(
                            (y[i] - val).abs() < 1e-10,
                            "y[{}] = {}, expected {}",
                            i,
                            y[i],
                            val
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_trace() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected trace: 1.0 + 5.0 + 9.0 = 15.0
        let expected_trace_square = 15.0;
        let computed_trace_square = mat_ref_square.trace();
        assert!(
            (computed_trace_square - expected_trace_square).abs() < 1e-10,
            "Trace of square matrix: expected {}, got {}",
            expected_trace_square,
            computed_trace_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected trace: 10.0 + 50.0 = 60.0 (min(nrows, ncols) = 2)
        let expected_trace_non_square = 60.0;
        let computed_trace_non_square = mat_ref_non_square.trace();
        assert!(
            (computed_trace_non_square - expected_trace_non_square).abs() < 1e-10,
            "Trace of non-square matrix: expected {}, got {}",
            expected_trace_non_square,
            computed_trace_non_square
        );

        // Define a matrix with no diagonal (nrows = 0 or ncols = 0)
        let data_empty = vec![]; // 0x0 matrix
        let mat_empty = create_faer_matrix(data_empty);
        let mat_ref_empty: &dyn Matrix<Scalar = f64> = &mat_empty;

        // Expected trace: 0.0
        let expected_trace_empty = 0.0;
        let computed_trace_empty = mat_ref_empty.trace();
        assert!(
            (computed_trace_empty - expected_trace_empty).abs() < 1e-10,
            "Trace of empty matrix: expected {}, got {}",
            expected_trace_empty,
            computed_trace_empty
        );

        // Define a 3x2 matrix
        let data_rect = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let mat_rect = create_faer_matrix(data_rect);
        let mat_ref_rect: &dyn Matrix<Scalar = f64> = &mat_rect;

        // Expected trace: 1.0 + 4.0 = 5.0 (min(nrows, ncols) = 2)
        let expected_trace_rect = 5.0;
        let computed_trace_rect = mat_ref_rect.trace();
        assert!(
            (computed_trace_rect - expected_trace_rect).abs() < 1e-10,
            "Trace of 3x2 matrix: expected {}, got {}",
            expected_trace_rect,
            computed_trace_rect
        );
    }

    #[test]
    fn test_frobenius_norm() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)
        // = sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81)
        // = sqrt(285) ≈ 16.881943016134134
        let expected_fro_norm_square = 16.881943016134134;
        let computed_fro_norm_square = mat_ref_square.frobenius_norm();
        assert!(
            (computed_fro_norm_square - expected_fro_norm_square).abs() < 1e-5,
            "Frobenius norm of square matrix: expected {}, got {}",
            expected_fro_norm_square,
            computed_fro_norm_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected Frobenius norm: sqrt(10^2 + 20^2 + 30^2 + 40^2 + 50^2 + 60^2)
        // = sqrt(100 + 400 + 900 + 1600 + 2500 + 3600)
        // = sqrt(9100) ≈ 95.394
        let expected_fro_norm_non_square = 95.39392014169457;
        let computed_fro_norm_non_square = mat_ref_non_square.frobenius_norm();
        assert!(
            (computed_fro_norm_non_square - expected_fro_norm_non_square).abs() < 1e-5,
            "Frobenius norm of non-square matrix: expected {}, got {}",
            expected_fro_norm_non_square,
            computed_fro_norm_non_square
        );

        // Define a zero matrix (0x0)
        let data_empty = vec![]; // 0x0 matrix
        let mat_empty = create_faer_matrix(data_empty);
        let mat_ref_empty: &dyn Matrix<Scalar = f64> = &mat_empty;

        // Expected Frobenius norm: 0.0
        let expected_fro_norm_empty = 0.0;
        let computed_fro_norm_empty = mat_ref_empty.frobenius_norm();
        assert!(
            (computed_fro_norm_empty - expected_fro_norm_empty).abs() < 1e-5,
            "Frobenius norm of empty matrix: expected {}, got {}",
            expected_fro_norm_empty,
            computed_fro_norm_empty
        );

        // Define a 3x2 matrix
        let data_rect = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let mat_rect = create_faer_matrix(data_rect);
        let mat_ref_rect: &dyn Matrix<Scalar = f64> = &mat_rect;

        // Expected Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)
        // = sqrt(1 + 4 + 9 + 16 + 25 + 36)
        // = sqrt(91) ≈ 9.539392014169456
        let expected_fro_norm_rect = 9.539392014169456;
        let computed_fro_norm_rect = mat_ref_rect.frobenius_norm();
        assert!(
            (computed_fro_norm_rect - expected_fro_norm_rect).abs() < 1e-05,
            "Frobenius norm of 3x2 matrix: expected {}, got {}",
            expected_fro_norm_rect,
            computed_fro_norm_rect
        );
    }
}
