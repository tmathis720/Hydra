// matrix.rs

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
}

// Implement Matrix trait for faer_core::Mat
// Implement the Matrix trait for faer_core::Mat
impl Matrix for Mat<f64> {
    type Scalar = f64;

    fn nrows(&self) -> usize {
        self.nrows()  // Return the number of rows in the matrix
    }

    fn ncols(&self) -> usize {
        self.ncols()  // Return the number of columns in the matrix
    }

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        // Perform matrix-vector multiplication
        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self.read(i, j) * x.get(j);  // Access the matrix elements and multiply with vector x
            }
            y.set(i, sum);  // Set the result in vector y
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.read(i, j)  // Read the matrix element at position (i, j)
    }


}