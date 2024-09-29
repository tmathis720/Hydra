use faer::Mat;


pub mod ksp;
pub mod cg;
pub mod preconditioner;

pub use ksp::KSP;
pub use cg::ConjugateGradient;


/* // Trait defining essential matrix operations (abstract over dense, sparse)
// Define that any type implementing Matrix must be Send and Sync
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>); // y = A * x
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
}

// Define that any type implementing Vector must be Send and Sync
pub trait Vector: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Self::Scalar;
    fn set(&mut self, i: usize, value: Self::Scalar);
    fn as_slice(&self) -> &[f64];
}

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

// Implement the Vector trait for faer_core::Mat (assuming a column vector structure)
impl Vector for Mat<f64> {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.nrows()  // The length of the vector is the number of rows (since it's a column vector)
    }

    fn get(&self, i: usize) -> f64 {
        self.read(i, 0)  // Access the i-th element in the column vector (first column)
    }

    fn set(&mut self, i: usize, value: f64) {
        self.write(i, 0, value);  // Set the i-th element in the column vector
    }

    fn as_slice(&self) -> &[f64] {
        self.as_ref()
            .col(0)
            .try_as_slice()  // Use `try_as_slice()`
            .expect("Column is not contiguous")  // Handle the potential `None` case
    }
}

impl Vector for Vec<f64> {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, i: usize) -> f64 {
        self[i]
    }

    fn set(&mut self, i: usize, value: f64) {
        self[i] = value;
    }

    fn as_slice(&self) -> &[f64] {
        &self
    }
}
 */