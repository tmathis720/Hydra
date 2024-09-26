use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use crate::solver::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

// Example of a Jacobi preconditioner using Arc<Mutex<T>> for safe parallelism
pub struct Jacobi;

impl Preconditioner for Jacobi {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let z = Arc::new(Mutex::new(z));  // Wrap z in Arc<Mutex<T>> for thread-safe access
        let a_rows: Vec<usize> = (0..a.nrows()).collect();

        // Use par_iter to process each row in parallel
        a_rows.into_par_iter().for_each(|i| {
            let ai = a.get(i, i);
            if ai != 0.0 {
                let ri = r.get(i);

                // Lock the mutex to get mutable access to z
                let mut z_guard = z.lock().unwrap();
                z_guard.set(i, ri / ai);  // Set the value in z (z[i] = r[i] / a[i][i])
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;  // Import the Jacobi preconditioner from the parent module
    use faer_core::{mat, Mat, MatMut};

    #[test]
    fn test_jacobi_preconditioner_simple() {
        // Create a simple diagonal matrix 'a'
        let a = mat![
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0]
        ];

        // Create a right-hand side vector 'r'
        let r = mat![
            [8.0],
            [9.0],
            [4.0]
        ];

        // Expected result after applying the Jacobi preconditioner
        let expected_z = mat![
            [2.0],  // 8 / 4
            [3.0],  // 9 / 3
            [2.0],  // 4 / 2
        ];

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(3, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }

    #[test]
    fn test_jacobi_preconditioner_with_zero_diagonal() {
        // Create a diagonal matrix 'a' with a zero diagonal entry
        let a = mat![
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  // Zero on the diagonal
            [0.0, 0.0, 2.0]
        ];

        // Create a right-hand side vector 'r'
        let r = mat![
            [8.0],
            [9.0],
            [4.0]
        ];

        // Expected result: The second row should not be updated due to zero diagonal
        let expected_z = mat![
            [2.0],  // 8 / 4
            [0.0],  // Division by zero, should leave z[i] = 0.0
            [2.0],  // 4 / 2
        ];

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(3, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result, with zero handling
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }

    #[test]
    fn test_jacobi_preconditioner_large_matrix() {
        // Create a larger diagonal matrix 'a'
        let n = 100;
        let mut a = Mat::<f64>::zeros(n, n);
        let mut r = Mat::<f64>::zeros(n, 1);
        let mut expected_z = Mat::<f64>::zeros(n, 1);

        // Fill 'a' and 'r' with values
        for i in 0..n {
            a.write(i, i, (i + 1) as f64);  // Diagonal matrix with increasing values
            r.write(i, 0, (i + 1) as f64 * 2.0);  // Right-hand side vector
            expected_z.write(i, 0, 2.0);  // Expected result (since r[i] = 2 * a[i])
        }

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(n, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }
}
