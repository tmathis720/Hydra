// src/vector/vec_impl.rs

use super::traits::Vector;

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

    fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self[..]
    }

    fn dot(&self, other: &dyn Vector<Scalar = f64>) -> f64 {
        self.iter().zip(other.as_slice()).map(|(x, y)| x * y).sum()
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn scale(&mut self, scalar: f64) {
        for value in self.iter_mut() {
            *value *= scalar;
        }
    }

    fn axpy(&mut self, a: f64, x: &dyn Vector<Scalar = f64>) {
        for (i, value) in self.iter_mut().enumerate() {
            *value = a * x.get(i) + *value;
        }
    }

    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = f64>) {
        for (i, value) in self.iter_mut().enumerate() {
            *value += other.get(i);
        }
    }

    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = f64>) {
        for (i, value) in self.iter_mut().enumerate() {
            *value *= other.get(i);
        }
    }

    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = f64>) {
        for (i, value) in self.iter_mut().enumerate() {
            *value /= other.get(i);
        }
    }

    fn cross(&mut self, other: &dyn Vector<Scalar = f64>) -> Result<(), &'static str> {
        if self.len() != 3 || other.len() != 3 {
            return Err("Cross product is only defined for 3-dimensional vectors");
        }

        // Compute the cross product and update `self`
        let x = self[1] * other.get(2) - self[2] * other.get(1);
        let y = self[2] * other.get(0) - self[0] * other.get(2);
        let z = self[0] * other.get(1) - self[1] * other.get(0);

        self[0] = x;
        self[1] = y;
        self[2] = z;

        Ok(())
    }

    fn sum(&self) -> f64 {
        self.iter().sum()
    }

    fn max(&self) -> f64 {
        self.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn min(&self) -> f64 {
        self.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / self.len() as f64
        }
    }

    fn variance(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            let mean = self.mean();
            let variance_sum: f64 = self.iter().map(|&x| (x - mean).powi(2)).sum();
            variance_sum / self.len() as f64
        }
    }
}
