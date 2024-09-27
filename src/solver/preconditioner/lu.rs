use faer::{linalg::solvers::PartialPivLu, mat::Mat, Parallelism};
use faer::linalg::solvers::SpSolver;  // Import the trait for solve_in_place
use crate::solver::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

pub struct LU {
    lu_decomp: PartialPivLu<f64>,
}

impl LU {
    pub fn new(matrix: &Mat<f64>) -> Self {
        let lu_decomp = PartialPivLu::new(matrix.as_ref());  // Create LU decomposition
        LU { lu_decomp }
    }

    fn apply(&self, rhs: &[f64], solution: &mut [f64]) {
        let rhs_matrix = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);  // Create matrix from rhs
        let mut sol_matrix = Mat::zeros(rhs.len(), 1);  // Initialize empty solution matrix
        
        // Perform the solve using the LU decomposition
        self.lu_decomp.solve_in_place(sol_matrix.as_mut());  // Use solve_in_place method

        // Copy the solution back to the output
        for i in 0..solution.len() {
            solution[i] = sol_matrix.read(i, 0);
        }
    }
}

impl Preconditioner for LU {
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let mut intermediate = vec![0.0; r.len()];  // Intermediate result storage
        self.apply(r.as_slice(), &mut intermediate);  // Use slices directly for `r`

        // Use direct element access to copy results into `z`
        for i in 0..z.len() {
            z.set(i, intermediate[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{mat, Mat};

    #[test]
    fn test_lu_preconditioner_simple() {
        // A simple 3x3 LU-factored matrix
        let lu = mat![
            [2.0, 3.0, 1.0],  // U
            [0.5, 0.5, 0.5],  // L and U
            [0.5, 1.0, 0.5]   // L and U
        ];

        let r = mat![
            [5.0],  // RHS vector
            [4.5],
            [1.0]
        ];

        // Expected solution z = [1.0, 1.0, -1.0]
        let expected_z = mat![
            [1.0],
            [1.0],
            [-1.0]
        ];

        let mut z = Mat::<f64>::zeros(3, 1);  // Initialize result vector

        // Create LU preconditioner and apply it
        let lu_preconditioner = LU::new(&lu);
        let r_values: Vec<f64> = (0..r.nrows()).map(|i| r.read(i, 0)).collect();  // Convert r into a Vec
        let mut z_values = vec![0.0; z.nrows()];  // Create a mutable Vec for z

        lu_preconditioner.apply(&r_values, &mut z_values);  // Apply LU preconditioner

        // Copy result back into z for verification
        for (i, &val) in z_values.iter().enumerate() {
            z.write(i, 0, val);
        }

        // Verify the result
        for i in 0..z.nrows() {
            assert!((z.read(i, 0) - expected_z.read(i, 0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lu_preconditioner_identity() {
        let lu = mat![
            [1.0, 0.0, 0.0],  // Identity matrix
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        let r = mat![
            [3.0],  // RHS vector
            [5.0],
            [7.0]
        ];

        let expected_z = r.clone();

        let mut z = Mat::<f64>::zeros(3, 1);  // Initialize result vector

        let lu_preconditioner = LU::new(&lu);
        let r_values: Vec<f64> = (0..r.nrows()).map(|i| r.read(i, 0)).collect();  // Convert r into a Vec
        let mut z_values = vec![0.0; z.nrows()];  // Create a mutable Vec for z

        lu_preconditioner.apply(&r_values, &mut z_values);  // Apply LU preconditioner

        // Copy result back into z for verification
        for (i, &val) in z_values.iter().enumerate() {
            z.write(i, 0, val);
        }

        // Verify the result
        for i in 0..z.nrows() {
            assert!((z.read(i, 0) - expected_z.read(i, 0)).abs() < 1e-6);
        }
    }
}

