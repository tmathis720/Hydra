use faer::{linalg::solvers::PartialPivLu, mat::Mat, mat::MatMut, Parallelism};
use faer::linalg::solvers::SpSolver;  // Import the trait for solve_in_place
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

pub struct LU {
    lu_decomp: PartialPivLu<f64>,
}

impl LU {
    pub fn new(matrix: &Mat<f64>) -> Self {
        // Print the matrix being decomposed
        println!("Decomposing matrix:\n{:?}", matrix);

        let lu_decomp = PartialPivLu::new(matrix.as_ref());  // Create LU decomposition
        
        // Print the internal structure of the LU decomposition
        println!("LU decomposition details:\n{:?}", lu_decomp);

        LU { lu_decomp }
    }

    fn apply(&self, rhs: &[f64], solution: &mut [f64]) {
        let mut sol_matrix = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);  // Initialize sol_matrix with RHS
    
        // Print the RHS matrix before solving
        println!("RHS matrix before solving:\n{:?}", sol_matrix);
    
        // Perform the solve using the LU decomposition
        self.lu_decomp.solve_in_place(sol_matrix.as_mut());
    
        // Print the solution matrix after solving
        println!("Solution matrix after solving:\n{:?}", sol_matrix);
    
        // Copy the solution back to the output
        for i in 0..solution.len() {
            solution[i] = sol_matrix[(i, 0)];
        }
    
        // Print the final solution vector
        println!("Final solution vector: {:?}", solution);
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

        println!("Preconditioner applied: z = {:?}", z.as_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{mat, Mat};

    #[test]
    fn test_lu_preconditioner_simple() {
        // Original matrix A
        let a = mat![
            [2.0, 3.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.5, 1.0, 0.5]
        ];

        let r = mat![
            [5.0],  // RHS vector
            [4.5],
            [1.0]
        ];

        // Correct expected solution z = [10.0, -7.0, 6.0]
        let expected_z = mat![
            [10.0],
            [-7.0],
            [6.0]
        ];

        let mut z = Mat::<f64>::zeros(3, 1);  // Initialize result vector

        // Print initial values
        println!("Testing LU preconditioner with matrix:\n{:?}", a);
        println!("RHS vector:\n{:?}", r);

        // Create LU preconditioner and apply it
        let lu_preconditioner = LU::new(&a);
        let r_values: Vec<f64> = (0..r.nrows()).map(|i| r.read(i, 0)).collect();  // Convert r into a Vec
        let mut z_values = vec![0.0; z.nrows()];  // Create a mutable Vec for z

        lu_preconditioner.apply(&r_values, &mut z_values);  // Apply LU preconditioner

        // Copy result back into z for verification
        for (i, &val) in z_values.iter().enumerate() {
            z.write(i, 0, val);
        }

        // Print the computed solution
        println!("Computed solution z = {:?}", z);

        // Verify the result
        for i in 0..z.nrows() {
            println!(
                "Comparing expected z[{}]: {:?} with computed z[{}]: {:?}",
                i,
                expected_z.read(i, 0),
                i,
                z.read(i, 0)
            );
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

        // Print initial values
        println!("Testing LU preconditioner with identity matrix:\n{:?}", lu);
        println!("RHS vector:\n{:?}", r);

        let lu_preconditioner = LU::new(&lu);
        let r_values: Vec<f64> = (0..r.nrows()).map(|i| r.read(i, 0)).collect();  // Convert r into a Vec
        let mut z_values = vec![0.0; z.nrows()];  // Create a mutable Vec for z

        lu_preconditioner.apply(&r_values, &mut z_values);  // Apply LU preconditioner

        // Copy result back into z for verification
        for (i, &val) in z_values.iter().enumerate() {
            z.write(i, 0, val);
        }

        // Print the computed solution
        println!("Computed solution z = {:?}", z);

        // Verify the result
        for i in 0..z.nrows() {
            println!(
                "Comparing expected z[{}]: {:?} with computed z[{}]: {:?}",
                i,
                expected_z.read(i, 0),
                i,
                z.read(i, 0)
            );
            assert!((z.read(i, 0) - expected_z.read(i, 0)).abs() < 1e-6);
        }
    }
}

