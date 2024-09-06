/// This module provides functions for basic matrix operations.
///
/// Currently, it includes:
/// - Matrix multiplication (`multiply_matrices`)

/// Multiplies two matrices.
///
/// # Arguments
///
/// * `a` - A slice of vectors representing the first matrix.
/// * `b` - A slice of vectors representing the second matrix.
///
/// # Returns
///
/// A `Vec<Vec<f64>>` representing the resulting matrix.
///
/// # Panics
///
/// Panics if the number of columns in the first matrix is not equal to the number of rows in the second matrix.
/// 
pub fn multiply_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();

    assert_eq!(a_cols, b_rows, "Number of columns in A must be equal to the number of rows in B");

    let mut result = vec![vec![0.0; b_cols]; a_rows];

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Transposes a given matrix.
///
/// # Arguments
///
/// * `matrix` - A reference to the matrix to transpose.
///
/// # Returns
///
/// A new matrix that is the transpose of the input matrix.
pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

/// Multiplies a matrix by a vector.
///
/// # Arguments
///
/// * `matrix` - A reference to the matrix.
/// * `vector` - A reference to the vector.
///
/// # Returns
///
/// A vector that is the result of the matrix-vector multiplication.
///
/// # Panics
///
/// Panics if the number of columns in the matrix does not equal the length of the vector.
pub fn multiply_matrix_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    assert_eq!(cols, vector.len(), "Matrix columns must match vector length");

    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_matrices() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ];
        let b = vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0]
        ];
        let expected = vec![
            vec![58.0, 64.0],
            vec![139.0, 154.0]
        ];
        let result = multiply_matrices(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Number of columns in A must be equal to the number of rows in B")]
    fn test_multiply_matrices_panic() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ];
        let b = vec![
            vec![5.0, 6.0, 7.0]
        ];
        multiply_matrices(&a, &b);
    }
}



#[test]
fn test_transpose() {
    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ];
    let expected = vec![
        vec![1.0, 4.0],
        vec![2.0, 5.0],
        vec![3.0, 6.0]
    ];
    let result = transpose(&matrix);
    assert_eq!(result, expected);
}




#[test]
fn test_multiply_matrix_vector() {
    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ];
    let vector = vec![7.0, 8.0, 9.0];
    let expected = vec![50.0, 122.0];
    let result = multiply_matrix_vector(&matrix, &vector);
    assert_eq!(result, expected);
}
