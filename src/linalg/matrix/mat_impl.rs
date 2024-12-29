use crate::linalg::Vector;
use faer::Mat;
use super::traits::{Matrix, MatrixOperations};

// Implement Matrix trait for faer_core::Mat
impl Matrix for Mat<f64> {
    type Scalar = f64;

    fn nrows(&self) -> usize {
        self.nrows() // Return the number of rows in the matrix
    }

    fn ncols(&self) -> usize {
        self.ncols() // Return the number of columns in the matrix
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

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        // Multiply the matrix with vector x and store the result in vector y.
        let nrows = self.nrows();
        let ncols = self.ncols();

        // Assuming y has been properly sized
        for i in 0..nrows {
            let mut sum = 0.0;
            for j in 0..ncols {
                sum += self.read(i, j) * x.get(j);
            }
            y.set(i, sum);
        }
    }

    fn get(&self, i: usize, j: usize) -> Self::Scalar {
        // Safely fetches the element at (i, j)
        self.read(i, j)
    }
}

// Implement MatrixOperations trait for faer_core::Mat
impl MatrixOperations for Mat<f64> {
    fn construct(rows: usize, cols: usize) -> Self {
        Mat::<f64>::zeros(rows, cols) // Construct a matrix initialized to zeros
    }

    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols()) // Return the dimensions of the matrix
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        // Set the element at (row, col) to value
        self.write(row, col, value);
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        // Fetches the element at (row, col)
        self.read(row, col)
    }
    
}


