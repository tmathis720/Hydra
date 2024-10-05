use crate::linalg::Vector;
use faer::Mat;
use super::traits::Matrix;

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

    // Implement the as_slice method by copying data to a Vec
    // Updated `as_slice` to return a `Box<[f64]>`
    fn as_slice(&self) -> Box<[f64]> {
        let mut data = Vec::new();
        let nrows = self.nrows();
        let ncols = self.ncols();
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(self.as_ref()[(i, j)]);
            }
        }
        data.into_boxed_slice()
    }

    fn as_slice_mut(&mut self) -> Box<[f64]> {
        let mut data = Vec::new();
        let nrows = self.nrows();
        let ncols = self.ncols();
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(self.as_mut()[(i, j)]);
            }
        }
        data.into_boxed_slice()
    }
}