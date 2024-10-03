// vector.rs

use faer::Mat;

/// Trait defining a set of common operations for vectors. 
/// It abstracts over different vector types, enabling flexible implementations 
/// for standard dense vectors or more complex matrix structures.
/// 
/// # Requirements:
/// Implementations of `Vector` must be thread-safe (`Send` and `Sync`).
pub trait Vector: Send + Sync {

    /// The scalar type of the vector elements.
    type Scalar: Copy + Send + Sync;

    /// Returns the length (number of elements) of the vector.
    fn len(&self) -> usize;

    /// Retrieves the element at index `i`.
    /// 
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn get(&self, i: usize) -> Self::Scalar;

    /// Sets the element at index `i` to `value`.
    /// 
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn set(&mut self, i: usize, value: Self::Scalar);

    /// Provides a slice of the underlying data.
    fn as_slice(&self) -> &[f64];

    /// Computes the dot product of `self` with another vector `other`.
    /// 
    /// # Example
    /// ```
    /// //let vec1 = vec![1.0, 2.0, 3.0];
    /// //let vec2 = vec![4.0, 5.0, 6.0];
    /// //let dot_product = vec1.dot(&vec2);
    /// ```
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;

    /// Computes the Euclidean norm (L2 norm) of the vector.
    /// 
    /// # Example
    /// ```
    /// //let vec = vec![3.0, 4.0];
    /// //let norm = vec.norm();
    /// //assert_eq!(norm, 5.0);
    /// ```
    fn norm(&self) -> Self::Scalar;

    /// Scales the vector by multiplying each element by the scalar `scalar`.
    fn scale(&mut self, scalar: Self::Scalar);

    /// Performs the operation `self = a * x + self`, also known as AXPY.
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);

    /// Adds another vector `other` to `self` element-wise.
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Multiplies `self` by another vector `other` element-wise.
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Divides `self` by another vector `other` element-wise.
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Computes the cross product with another vector `other` (for 3D vectors only).
    /// 
    /// # Errors
    /// Returns an error if the vectors are not 3-dimensional.
    fn cross(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str>;

    /// Computes the sum of all elements in the vector.
    fn sum(&self) -> Self::Scalar;

    /// Returns the maximum element of the vector.
    fn max(&self) -> Self::Scalar;

    /// Returns the minimum element of the vector.
    fn min(&self) -> Self::Scalar;

    /// Returns the mean value of the vector.
    fn mean(&self) -> Self::Scalar;

    /// Returns the variance of the vector.
    fn variance(&self) -> Self::Scalar;

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

    fn cross(&mut self, other: &dyn Vector<Scalar = f64>) -> Result<(), &'static str> {
        if self.len() != 3 || other.len() != 3 {
            return Err("Cross product is only defined for 3-dimensional vectors");
        }

        // Compute the cross product and update `self`
        let x = self.get(1,0) * other.get(2) - self.get(2,0) * other.get(1);
        let y = self.get(2,0) * other.get(0) - self.get(0,0) * other.get(2);
        let z = self.get(0,0) * other.get(1) - self.get(1,0) * other.get(0);

        self.write(0, 0, x);
        self.write(1, 0, y);
        self.write(2, 0, z);

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
            return 0.0;
        }
        self.sum() / self.len() as f64
    }

    fn variance(&self) -> f64 {
        if self.len() == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let mut variance_sum = 0.0;
        for i in 0..self.len() {
            let diff = self.get(i,0) - mean;
            variance_sum += diff * diff;
        }
        variance_sum / self.len() as f64
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
        if self.len() == 0 {
            return 0.0;
        }
        self.sum() / self.len() as f64
    }

    fn variance(&self) -> f64 {
        if self.len() == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance_sum: f64 = self.iter().map(|&x| (x - mean).powi(2)).sum();
        variance_sum / self.len() as f64
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

    // Test cross product for Vec<f64>
    #[test]
    fn test_vec_cross_product() {
        let mut vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        vec1.cross(&vec2).expect("Cross product should succeed");

        assert_eq!(vec1, vec![0.0, 0.0, 1.0], "Cross product should be [0.0, 0.0, 1.0]");
    }

    // Test cross product for faer::Mat<f64>
    #[test]
    fn test_mat_cross_product() {
        let mut mat1 = Mat::from_fn(3, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let mat2 = Mat::from_fn(3, 1, |i, _| if i == 1 { 1.0 } else { 0.0 });
        mat1.cross(&mat2).expect("Cross product should succeed");

        assert_eq!(mat1.as_slice(), &[0.0, 0.0, 1.0], "Cross product should be [0.0, 0.0, 1.0]");
    }

    #[test]
    fn test_vec_sum() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(vec.sum(), 10.0, "Sum should be 10.0");
    }

    #[test]
    fn test_mat_sum() {
        let mat = Mat::from_fn(4, 1, |i, _| (i + 1) as f64); // Matrix with values [1.0, 2.0, 3.0, 4.0]
        assert_eq!(mat.sum(), 10.0, "Sum should be 10.0");
    }

    #[test]
    fn test_vec_max() {
        let vec = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(vec.max(), 5.0, "Max should be 5.0");
    }

    #[test]
    fn test_mat_max() {
        let mat = Mat::from_fn(4, 1, |i, _| (i as f64) * 2.0); // Matrix with values [0.0, 2.0, 4.0, 6.0]
        assert_eq!(mat.max(), 6.0, "Max should be 6.0");
    }

    #[test]
    fn test_vec_min() {
        let vec = vec![1.0, 3.0, 2.0, -5.0, 4.0];
        assert_eq!(vec.min(), -5.0, "Min should be -5.0");
    }

    #[test]
    fn test_mat_min() {
        let mat = Mat::from_fn(4, 1, |i, _| (i as f64) * 2.0 - 3.0); // Matrix with values [-3.0, -1.0, 1.0, 3.0]
        assert_eq!(mat.min(), -3.0, "Min should be -3.0");
    }

    #[test]
    fn test_vec_mean() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(vec.mean(), 3.0, "Mean should be 3.0");
    }

    #[test]
    fn test_mat_mean() {
        let mat = Mat::from_fn(5, 1, |i, _| (i as f64) + 1.0); // Matrix with values [1.0, 2.0, 3.0, 4.0, 5.0]
        assert_eq!(mat.mean(), 3.0, "Mean should be 3.0");
    }

    #[test]
    fn test_empty_vec_mean() {
        let vec: Vec<f64> = vec![];
        assert_eq!(vec.mean(), 0.0, "Mean should be 0.0 for empty vector");
    }

    #[test]
    fn test_empty_mat_mean() {
        let mat = Mat::from_fn(0, 1, |_i, _| 0.0); // Empty matrix
        assert_eq!(mat.mean(), 0.0, "Mean should be 0.0 for empty matrix");
    }

    #[test]
    fn test_vec_variance() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((vec.variance() - 2.0).abs() < 1e-6, "Variance should be 2.0");
    }

    #[test]
    fn test_mat_variance() {
        let mat = Mat::from_fn(5, 1, |i, _| (i as f64) + 1.0); // Matrix with values [1.0, 2.0, 3.0, 4.0, 5.0]
        assert!((mat.variance() - 2.0).abs() < 1e-6, "Variance should be 2.0");
    }

    #[test]
    fn test_empty_vec_variance() {
        let vec: Vec<f64> = vec![];
        assert_eq!(vec.variance(), 0.0, "Variance should be 0.0 for empty vector");
    }

    #[test]
    fn test_empty_mat_variance() {
        let mat = Mat::from_fn(0, 1, |_i, _| 0.0); // Empty matrix
        assert_eq!(mat.variance(), 0.0, "Variance should be 0.0 for empty matrix");
    }
}