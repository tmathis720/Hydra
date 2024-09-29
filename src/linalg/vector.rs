// vector.rs

use faer::Mat;

// Trait definition for vectors, expanding on the existing one
pub trait Vector: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Self::Scalar;
    fn set(&mut self, i: usize, value: Self::Scalar);
    fn as_slice(&self) -> &[f64];
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;  // Add dot product
    fn norm(&self) -> Self::Scalar;  // Add norm computation
    fn scale(&mut self, scalar: Self::Scalar);  // Add the scale function here
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
}

// Implementing the Vector trait for faer_core::Mat (column vector assumption)
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

    fn dot(&self, other: &dyn Vector<Scalar = f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self.get(i, 0) * other.get(i);
        }
        sum
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()  // Compute L2 norm
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a simple vector for testing
    fn create_test_vector() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0]
    }

    // Test for `len` method
    #[test]
    fn test_vector_len() {
        let vec = create_test_vector();
        assert_eq!(vec.len(), 4, "Vector length should be 4");
    }

    // Test for `get` method
    #[test]
    fn test_vector_get() {
        let vec = create_test_vector();
        assert_eq!(vec.get(0), 1.0, "First element should be 1.0");
        assert_eq!(vec.get(2), 3.0, "Third element should be 3.0");
    }

    // Test for `set` method
    #[test]
    fn test_vector_set() {
        let mut vec = create_test_vector();
        vec.set(1, 10.0); // Set the second element to 10.0
        assert_eq!(vec.get(1), 10.0, "Second element should be updated to 10.0");
    }

    // Test for `dot` product method
    #[test]
    fn test_vector_dot() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        
        let dot_product = vec1.dot(&vec2);
        assert_eq!(dot_product, 32.0, "Dot product should be 32.0 (1*4 + 2*5 + 3*6)");
    }

    // Test for `norm` method (Euclidean norm)
    #[test]
    fn test_vector_norm() {
        let vec = vec![3.0, 4.0];
        
        let norm = vec.norm();
        assert!((norm - 5.0).abs() < 1e-6, "Norm should be 5.0 (sqrt(3^2 + 4^2))");
    }

    // Test for `as_slice` method
    #[test]
    fn test_vector_as_slice() {
        let vec = create_test_vector();
        let slice = vec.as_slice();
        
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0], "Slice should match the original vector");
    }

    // Test for `scale` method
    #[test]
    fn test_vector_scale() {
        let mut vec = create_test_vector();
        vec.scale(2.0);  // Scale the vector by 2.0
        
        assert_eq!(vec.as_slice(), &[2.0, 4.0, 6.0, 8.0], "All elements should be scaled by 2.0");
    }

    #[test]
    fn test_vector_axpy() {
        let mut y = vec![1.0, 1.0, 1.0, 1.0];  // Y vector
        let x = vec![2.0, 2.0, 2.0, 2.0];  // X vector

        y.axpy(2.0, &x);  // Perform the operation: y = 2 * x + y

        assert_eq!(y.as_slice(), &[5.0, 5.0, 5.0, 5.0], "Result should be [5.0, 5.0, 5.0, 5.0]");
    }

    #[test]
    fn test_vector_element_wise_add() {
        let mut vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![4.0, 3.0, 2.0, 1.0];

        vec1.element_wise_add(&vec2);  // Perform element-wise addition

        assert_eq!(vec1.as_slice(), &[5.0, 5.0, 5.0, 5.0], "Result should be [5.0, 5.0, 5.0, 5.0]");
    }

    #[test]
    fn test_vector_element_wise_mul() {
        let mut vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![4.0, 3.0, 2.0, 1.0];

        vec1.element_wise_mul(&vec2);  // Perform element-wise multiplication

        assert_eq!(vec1.as_slice(), &[4.0, 6.0, 6.0, 4.0], "Result should be [4.0, 6.0, 6.0, 4.0]");
    }

    #[test]
    fn test_vector_element_wise_div() {
        let mut vec1 = vec![4.0, 9.0, 16.0, 25.0];
        let vec2 = vec![2.0, 3.0, 4.0, 5.0];

        vec1.element_wise_div(&vec2);  // Perform element-wise division

        assert_eq!(vec1.as_slice(), &[2.0, 3.0, 4.0, 5.0], "Result should be [2.0, 3.0, 4.0, 5.0]");
    }
}