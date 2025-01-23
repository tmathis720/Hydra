// src/vector/mat_impl.rs

use faer::Mat;
use super::traits::Vector;

impl Vector for Mat<f64> {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.nrows() // The length of the vector is the number of rows (since it's a column vector)
    }

    fn get(&self, i: usize) -> f64 {
        self[(i, 0)] // Access the i-th element in the column vector (first column)
    }

    fn set(&mut self, i: usize, value: f64) {
        self[(i, 0)] = value; // Set the i-th element in the column vector
    }

    fn as_slice(&self) -> &[f64] {
        let col = self.as_ref().col(0); // Access the first column.
        let slice = unsafe { std::slice::from_raw_parts(col.as_ptr(), col.nrows()) };
        slice
    }
    
    fn as_mut_slice(&mut self) -> &mut [f64] {
        let col_mut = self.as_mut().col_mut(0); // Access the first column mutably.
        unsafe {
            std::slice::from_raw_parts_mut(col_mut.as_ptr() as *mut f64, col_mut.nrows()) // Create a mutable slice from the raw pointer and length.
        }
    }

    fn dot(&self, other: &dyn Vector<Scalar = f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self.get(i, 0) * other.get(i);
        }
        sum
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt() // Compute L2 norm
    }

    fn scale(&mut self, scalar: f64) {
        for i in 0..self.len() {
            let value = self.get(i, 0) * scalar;
            self.set(i, value);
        }
    }

    fn axpy(&mut self, a: f64, x: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = a * x.get(i) + self.get(i, 0);
            self.set(i, value);
        }
    }

    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i, 0) + other.get(i);
            self.set(i, value);
        }
    }

    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i, 0) * other.get(i);
            self.set(i, value);
        }
    }

    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i, 0) / other.get(i);
            self.set(i, value);
        }
    }

    fn cross(&mut self, other: &dyn Vector<Scalar = f64>) -> Result<(), &'static str> {
        if self.len() != 3 || other.len() != 3 {
            return Err("Cross product is only defined for 3-dimensional vectors");
        }

        let x = self.get(1, 0) * other.get(2) - self.get(2, 0) * other.get(1);
        let y = self.get(2, 0) * other.get(0) - self.get(0, 0) * other.get(2);
        let z = self.get(0, 0) * other.get(1) - self.get(1, 0) * other.get(0);

        self[(0, 0)] = x;
        self[(1, 0)] = y;
        self[(2, 0)] = z;

        Ok(())
    }

    fn sum(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.len() {
            total += self.get(i, 0);
        }
        total
    }

    fn max(&self) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..self.len() {
            max_val = f64::max(max_val, *self.get(i, 0));
        }
        max_val
    }

    fn min(&self) -> f64 {
        let mut min_val = f64::INFINITY;
        for i in 0..self.len() {
            min_val = f64::min(min_val, *self.get(i, 0));
        }
        min_val
    }

    fn mean(&self) -> f64 {
        if self.len() == 0 {
            0.0
        } else {
            self.sum() / self.len() as f64
        }
    }

    fn variance(&self) -> f64 {
        if self.len() == 0 {
            0.0
        } else {
            let mean = self.mean();
            let mut variance_sum = 0.0;
            for i in 0..self.len() {
                let diff = self.get(i, 0) - mean;
                variance_sum += diff * diff;
            }
            variance_sum / self.len() as f64
        }
    }
}
