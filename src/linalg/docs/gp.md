Generate a detailed users guide for the `Linear Algebra` module for Hydra. I am going to provide the code for all of the parts of the `Linear Algebra` module below, and you can analyze and build the detailed outline based on this version of the source code.

`src/linalg/mod.rs`

```rust
pub mod matrix;
pub mod vector;

pub use matrix::traits::Matrix;
pub use vector::traits::Vector;
```

---

`src/linalg/vector/mod.rs`

```rust
// src/vector/mod.rs

pub mod traits;
pub mod vec_impl;
pub mod mat_impl;
pub mod vector_builder;

pub use traits::Vector;

#[cfg(test)]
mod tests;
```

---

`src/linalg/vector/traits.rs`

```rust
// src/vector/traits.rs


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

    /// Provides a mutable slice of the underlying data.
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];

    /// Computes the dot product of `self` with another vector `other`.
    ///
    /// # Example
    /// 
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec1: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let vec2: Vec<f64> = vec![4.0, 5.0, 6.0];
    /// let dot_product = vec1.dot(&vec2);
    /// assert_eq!(dot_product, 32.0);
    /// ```
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;

    /// Computes the Euclidean norm (L2 norm) of the vector.
    ///
    /// # Example
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec: Vec<f64> = vec![3.0, 4.0];
    /// let norm = vec.norm();
    /// assert_eq!(norm, 5.0);
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
```

---

`src/linalg/vector/vec_impl.rs`

```rust
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
```

---

`src/linalg/vector/mat_impl.rs`

```rust
// src/vector/mat_impl.rs

use faer::Mat;
use super::traits::Vector;

impl Vector for Mat<f64> {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.nrows() // The length of the vector is the number of rows (since it's a column vector)
    }

    fn get(&self, i: usize) -> f64 {
        self.read(i, 0) // Access the i-th element in the column vector (first column)
    }

    fn set(&mut self, i: usize, value: f64) {
        self.write(i, 0, value); // Set the i-th element in the column vector
    }

    fn as_slice(&self) -> &[f64] {
        self.as_ref()
            .col(0)
            .try_as_slice() // Use `try_as_slice()`
            .expect("Column is not contiguous") // Handle the potential `None` case
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.as_mut()
            .col_mut(0)
            .try_as_slice_mut()
            .expect("Column is not contiguous")
    }

    fn dot(&self, other: &dyn Vector<Scalar = f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self.get(i,0) * other.get(i);
        }
        sum
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt() // Compute L2 norm
    }

    fn scale(&mut self, scalar: f64) {
        for i in 0..self.len() {
            let value = self.get(i,0) * scalar;
            self.set(i, value);
        }
    }

    fn axpy(&mut self, a: f64, x: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = a * x.get(i) + self.get(i,0);
            self.set(i, value);
        }
    }

    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i,0) + other.get(i);
            self.set(i, value);
        }
    }

    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i,0) * other.get(i);
            self.set(i, value);
        }
    }

    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = f64>) {
        for i in 0..self.len() {
            let value = self.get(i,0) / other.get(i);
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
            total += self.get(i,0);
        }
        total
    }

    fn max(&self) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..self.len() {
            max_val = f64::max(max_val, *self.get(i,0));
        }
        max_val
    }

    fn min(&self) -> f64 {
        let mut min_val = f64::INFINITY;
        for i in 0..self.len() {
            min_val = f64::min(min_val, *self.get(i,0));
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
                let diff = self.get(i,0) - mean;
                variance_sum += diff * diff;
            }
            variance_sum / self.len() as f64
        }
    }
}
```

---

`src/linalg/vector/tests.rs`

```rust
// src/vector/tests.rs

use super::traits::Vector;
use faer::Mat;

/// Helper function to create a simple vector for testing.
fn create_test_vector() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0]
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let x = vec![2.0, 2.0, 2.0, 2.0];      // X vector

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
```

---

`src/linalg/vector/vector_builder.rs`

```rust
use faer::Mat; // Example for faer vector support.

pub struct VectorBuilder;

impl VectorBuilder {
    /// Builds a vector of the specified type with a given initial size.
    /// Supports various vector types through generics.
    ///
    /// # Parameters
    /// - `size`: The length of the vector.
    ///
    /// # Returns
    /// A vector of type `T` initialized to the specified length.
    pub fn build_vector<T: VectorOperations>(size: usize) -> T {
        T::construct(size)
    }

    /// Builds a dense vector using faer's `Mat` structure as a column vector.
    /// Initializes with zeros.
    pub fn build_dense_vector(size: usize) -> Mat<f64> {
        Mat::<f64>::zeros(size, 1)
    }

    /// Resizes the provided vector dynamically while maintaining memory safety.
    /// Ensures no data is left uninitialized during resizing.
    pub fn resize_vector<T: VectorOperations + ExtendedVectorOperations>(
        vector: &mut T,
        new_size: usize,
    ) {
        vector.resize(new_size);
    }
}

pub trait VectorOperations {
    fn construct(size: usize) -> Self
    where
        Self: Sized;
    fn set_value(&mut self, index: usize, value: f64);
    fn get_value(&self, index: usize) -> f64;
    fn size(&self) -> usize;
}

pub trait ExtendedVectorOperations: VectorOperations {
    fn resize(&mut self, new_size: usize)
    where
        Self: Sized;
}

impl VectorOperations for Vec<f64> {
    fn construct(size: usize) -> Self {
        vec![0.0; size]
    }

    fn set_value(&mut self, index: usize, value: f64) {
        self[index] = value;
    }

    fn get_value(&self, index: usize) -> f64 {
        self[index]
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl ExtendedVectorOperations for Vec<f64> {
    fn resize(&mut self, new_size: usize) {
        self.resize(new_size, 0.0);
    }
}

impl VectorOperations for Mat<f64> {
    fn construct(size: usize) -> Self {
        Mat::<f64>::zeros(size, 1)
    }

    fn set_value(&mut self, index: usize, value: f64) {
        self.write(index, 0, value);
    }

    fn get_value(&self, index: usize) -> f64 {
        self.read(index, 0)
    }

    fn size(&self) -> usize {
        self.nrows()
    }
}

impl ExtendedVectorOperations for Mat<f64> {
    fn resize(&mut self, new_size: usize) {
        let mut new_vector = Mat::<f64>::zeros(new_size, 1);
        for i in 0..usize::min(self.nrows(), new_size) {
            new_vector.write(i, 0, self.read(i, 0));
        }
        *self = new_vector;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dense_vector() {
        let size = 5;
        let vector = VectorBuilder::build_dense_vector(size);

        assert_eq!(vector.nrows(), size, "Vector length should match the specified size.");
        assert_eq!(vector.ncols(), 1, "Vector should be a column vector.");

        for i in 0..size {
            assert_eq!(vector.read(i, 0), 0.0, "Vector should be initialized to zero.");
        }
    }

    #[test]
    fn test_build_vector_generic() {
        let size = 4;
        let vector = VectorBuilder::build_vector::<Vec<f64>>(size);

        assert_eq!(vector.len(), size, "Vector length should match the specified size.");
        for val in vector.iter() {
            assert_eq!(*val, 0.0, "Vector should be initialized to zero.");
        }
    }

    #[test]
    fn test_resize_vector() {
        let mut vector = VectorBuilder::build_dense_vector(3);
        vector.write(0, 0, 1.0);
        vector.write(1, 0, 2.0);
        vector.write(2, 0, 3.0);

        VectorBuilder::resize_vector(&mut vector, 5);

        assert_eq!(vector.nrows(), 5, "Vector length should be updated after resizing.");
        let expected_values = vec![1.0, 2.0, 3.0, 0.0, 0.0];
        for i in 0..5 {
            assert_eq!(vector.read(i, 0), expected_values[i], "Vector element mismatch at index {}.", i);
        }
    }
}
```

---

`src/linalg/matrix/mod.rs`

```rust
// src/linalg/matrix/mod.rs

pub mod traits;
pub mod mat_impl;
pub mod matrix_builder;

pub use traits::Matrix;
pub use traits::MatrixOperations;
pub use traits::ExtendedMatrixOperations;

#[cfg(test)]
mod tests;
```

---

`src/linalg/matrix/traits.rs`

```rust
// src/linalg/matrix/traits.rs

use crate::linalg::Vector;

/// Trait defining essential matrix operations (abstract over dense, sparse)
/// Define that any type implementing Matrix must be Send and Sync
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>); // y = A * x
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    fn trace(&self) -> Self::Scalar;
    fn frobenius_norm(&self) -> Self::Scalar;
    fn as_slice(&self) -> Box<[Self::Scalar]>;
    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}

/// Trait defining matrix operations for building and manipulation
pub trait MatrixOperations: Send + Sync {
    fn construct(rows: usize, cols: usize) -> Self
    where
        Self: Sized;
    fn set(&mut self, row: usize, col: usize, value: f64);
    fn get(&self, row: usize, col: usize) -> f64;
    fn size(&self) -> (usize, usize);
}

/// Extended matrix operations trait for resizing
pub trait ExtendedMatrixOperations: MatrixOperations {
    fn resize(&mut self, new_rows: usize, new_cols: usize)
    where
        Self: Sized;
}
```

---

`src/linalg/matrix/mat_impl.rs`

```rust
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
```

---

`src/linalg/matrix/matrix_builder.rs`

```rust
use crate::linalg::matrix::{Matrix, traits::MatrixOperations};
use crate::solver::preconditioner::Preconditioner;
use faer::Mat; // Example using faer for dense matrix support.

pub struct MatrixBuilder;

impl MatrixBuilder {
    /// Builds a matrix of the specified type with given initial size.
    /// Supports various matrix types through generics.
    ///
    /// # Parameters
    /// - `rows`: The number of rows for the matrix.
    /// - `cols`: The number of columns for the matrix.
    ///
    /// # Returns
    /// A matrix of type `T` initialized to the specified dimensions.
    pub fn build_matrix<T: MatrixOperations>(rows: usize, cols: usize) -> T {
        let matrix = T::construct(rows, cols);
        matrix
    }

    /// Builds a dense matrix with faer's `Mat` structure.
    /// Initializes with zeros and demonstrates integration with potential preconditioners.
    pub fn build_dense_matrix(rows: usize, cols: usize) -> Mat<f64> {
        Mat::<f64>::zeros(rows, cols)
    }

    /// Resizes the provided matrix dynamically while maintaining memory safety.
    /// Ensures no data is left uninitialized during resizing.
    pub fn resize_matrix<T: MatrixOperations + ExtendedMatrixOperations>(
        matrix: &mut T,
        new_rows: usize,
        new_cols: usize,
    ) {
        // Call the resizing operation implemented by the specific matrix type.
        matrix.resize(new_rows, new_cols);
    }

    /// Demonstrates matrix compatibility with preconditioners by applying a preconditioner.
    pub fn apply_preconditioner<P: Preconditioner>(
        preconditioner: &P,
        matrix: &dyn Matrix<Scalar = f64>,
    ) {
        let input_vector = vec![0.0; matrix.ncols()]; // Initialize input vector with zeros.
        let mut result_vector = vec![0.0; matrix.nrows()]; // Initialize result vector with zeros.
        
        preconditioner.apply(matrix, &input_vector, &mut result_vector);
    }
}

pub trait ExtendedMatrixOperations: MatrixOperations {
    /// Dynamically resizes the matrix.
    fn resize(&mut self, new_rows: usize, new_cols: usize);
}

impl ExtendedMatrixOperations for Mat<f64> {
    fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let mut new_matrix = Mat::<f64>::zeros(new_rows, new_cols);
        for j in 0..usize::min(self.ncols(), new_cols) {
            for i in 0..usize::min(self.nrows(), new_rows) {
                new_matrix[(i, j)] = self[(i, j)];
            }
        }
        *self = new_matrix; // Replace old matrix with the resized matrix.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linalg::matrix::traits::{MatrixOperations, Matrix}, Vector};

    /// Test for building a dense matrix using `MatrixBuilder::build_dense_matrix`.
    #[test]
    fn test_build_dense_matrix() {
        let rows = 3;
        let cols = 3;
        let matrix = MatrixBuilder::build_dense_matrix(rows, cols);

        assert_eq!(matrix.nrows(), rows, "Number of rows should match.");
        assert_eq!(matrix.ncols(), cols, "Number of columns should match.");

        // Verify that the matrix is initialized with zeros.
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.read(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    /// Test for building a generic matrix using `MatrixBuilder::build_matrix`.
    #[test]
    fn test_build_matrix_generic() {
        struct DummyMatrix {
            data: Vec<Vec<f64>>,
            rows: usize,
            cols: usize,
        }

        impl MatrixOperations for DummyMatrix {
            fn construct(rows: usize, cols: usize) -> Self {
                DummyMatrix {
                    data: vec![vec![0.0; cols]; rows],
                    rows,
                    cols,
                }
            }
            fn size(&self) -> (usize, usize) {
                (self.rows, self.cols)
            }
            
            fn set(&mut self, row: usize, col: usize, value: f64) {
                self.data[row][col] = value;
            }
            
            fn get(&self, row: usize, col: usize) -> f64 {
                self.data[row][col]
            }
        }

        let rows = 4;
        let cols = 5;
        let matrix = MatrixBuilder::build_matrix::<DummyMatrix>(rows, cols);

        assert_eq!(matrix.size(), (rows, cols), "Matrix size should match the specified dimensions.");

        // Ensure matrix is initialized with zeros.
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(matrix.get(i, j), 0.0, "Matrix should be initialized to zero.");
            }
        }
    }

    /// Test for resizing a matrix using `MatrixBuilder::resize_matrix`.
    #[test]
    fn test_resize_matrix() {
        let mut matrix = MatrixBuilder::build_dense_matrix(2, 2);
        matrix.write(0, 0, 1.0);
        matrix.write(0, 1, 2.0);
        matrix.write(1, 0, 3.0);
        matrix.write(1, 1, 4.0);

        MatrixBuilder::resize_matrix(&mut matrix, 3, 3);

        // Check new size.
        assert_eq!(matrix.nrows(), 3, "Matrix should have 3 rows after resizing.");
        assert_eq!(matrix.ncols(), 3, "Matrix should have 3 columns after resizing.");

        // Check that original data is preserved and new cells are zero.
        let expected_values = vec![
            vec![1.0, 2.0, 0.0],
            vec![3.0, 4.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix.read(i, j), expected_values[i][j], "Matrix data mismatch at ({}, {}).", i, j);
            }
        }
    }

    /// Test for applying a preconditioner using `MatrixBuilder::apply_preconditioner`.
    #[test]
    fn test_apply_preconditioner() {
        struct DummyPreconditioner;
        impl Preconditioner for DummyPreconditioner {
            fn apply(
                &self,
                _a: &dyn Matrix<Scalar = f64>,
                _r: &dyn Vector<Scalar = f64>,
                z: &mut dyn Vector<Scalar = f64>,
            ) {
                for i in 0..z.len() {
                    z.set(i, 1.0); // Set all elements in the result vector to 1.0.
                }
            }
        }

        let matrix = MatrixBuilder::build_dense_matrix(2, 2);
        let preconditioner = DummyPreconditioner;
        let input_vector = vec![0.5, 0.5];
        let mut result_vector = vec![0.0, 0.0];

        preconditioner.apply(&matrix, &input_vector, &mut result_vector);

        // Check that the preconditioner applied the expected transformation.
        for val in result_vector.iter() {
            assert_eq!(*val, 1.0, "Each element in the result vector should be 1.0 after applying the preconditioner.");
        }
    }
}
```

---

`src/linalg/matrix/tests.rs`

```rust
use super::traits::Matrix;
use crate::linalg::Vector;

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use std::sync::Arc;

    /// Helper function to create a faer::Mat<f64> from a 2D Vec.
    fn create_faer_matrix(data: Vec<Vec<f64>>) -> Mat<f64> {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };
        let mut mat = Mat::zeros(nrows, ncols);

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                mat.write(i, j, val);
            }
        }

        mat
    }

    /// Helper function to create a faer::Mat<f64> as a column vector.
    fn create_faer_vector(data: Vec<f64>) -> Mat<f64> {
        let nrows = data.len();
        let ncols = 1;
        let mut mat = Mat::zeros(nrows, ncols);

        for (i, &val) in data.iter().enumerate() {
            mat.write(i, 0, val);
        }

        mat
    }

    #[test]
    fn test_nrows_ncols() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        assert_eq!(mat_ref.nrows(), 3);
        assert_eq!(mat_ref.ncols(), 3);
    }

    #[test]
    fn test_get() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data.clone());
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_eq!(mat_ref.get(i, j), val);
            }
        }
    }

    #[test]
    fn test_mat_vec_with_vec_f64() {
        // Define matrix A
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using Vec<f64>
        let x = vec![1.0, 0.0, -1.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        // y[0] = 1*1 + 2*0 + 3*(-1) = 1 - 3 = -2
        // y[1] = 4*1 + 5*0 + 6*(-1) = 4 - 6 = -2
        // y[2] = 7*1 + 8*0 + 9*(-1) = 7 - 9 = -2
        let expected = vec![-2.0, -2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_with_faer_vector() {
        // Define matrix A
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using faer::Mat<f64> as a column vector
        let x_data = vec![1.0, 0.0, -1.0];
        let x = create_faer_vector(x_data);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i, 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i, 0),
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_identity_with_vec_f64() {
        // Define an identity matrix
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using Vec<f64>
        let x = vec![5.0, -3.0, 2.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result is x itself
        let expected = vec![5.0, -3.0, 2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_zero_matrix_with_faer_vector() {
        // Define a zero matrix
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x using faer::Mat<f64> as a column vector
        let x = create_faer_vector(vec![3.0, 4.0]);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result is a zero vector
        let expected = vec![0.0, 0.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i , 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i , 0),
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_non_square_matrix_with_vec_f64() {
        // Define a non-square matrix (2x3)
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x (size 3) using Vec<f64>
        let x = vec![1.0, 0.0, -1.0];
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y (size 2) using Vec<f64>
        let mut y = vec![0.0; mat_ref.nrows()];
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y[i] - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                val
            );
        }
    }

    #[test]
    fn test_mat_vec_non_square_matrix_with_faer_vector() {
        // Define a non-square matrix (2x3)
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Define vector x (size 3) using faer::Mat<f64> as a column vector
        let x = create_faer_vector(vec![1.0, 0.0, -1.0]);
        let x_ref: &dyn Vector<Scalar = f64> = &x;

        // Initialize vector y (size 2) using faer::Mat<f64> as a column vector
        let mut y = create_faer_vector(vec![0.0; mat_ref.nrows()]);
        let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

        // Perform y = A * x
        mat_ref.mat_vec(x_ref, y_ref);

        // Expected result
        let expected = vec![-2.0, -2.0];

        for (i, &val) in expected.iter().enumerate() {
            assert!(
                (y.get(i , 0) - val).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y.get(i , 0),
                val
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_row() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds row should panic
        mat_ref.get(2, 1);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_column() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds column should panic
        mat_ref.get(1, 2);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref = Arc::new(mat);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let mat_clone = Arc::clone(&mat_ref);
                thread::spawn(move || {
                    // Define vector x using Vec<f64>
                    let x = vec![1.0, 0.0, -1.0];
                    let x_ref: &dyn Vector<Scalar = f64> = &x;

                    // Initialize vector y using Vec<f64>
                    let mut y = vec![0.0; mat_clone.nrows()];
                    let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

                    // Perform y = A * x
                    mat_clone.mat_vec(x_ref, y_ref);

                    // Expected result is [-2.0, -2.0, -2.0]
                    let expected = vec![-2.0, -2.0, -2.0];

                    for (i, &val) in expected.iter().enumerate() {
                        assert!(
                            (y[i] - val).abs() < 1e-10,
                            "y[{}] = {}, expected {}",
                            i,
                            y[i],
                            val
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_trace() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected trace: 1.0 + 5.0 + 9.0 = 15.0
        let expected_trace_square = 15.0;
        let computed_trace_square = mat_ref_square.trace();
        assert!(
            (computed_trace_square - expected_trace_square).abs() < 1e-10,
            "Trace of square matrix: expected {}, got {}",
            expected_trace_square,
            computed_trace_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected trace: 10.0 + 50.0 = 60.0 (min(nrows, ncols) = 2)
        let expected_trace_non_square = 60.0;
        let computed_trace_non_square = mat_ref_non_square.trace();
        assert!(
            (computed_trace_non_square - expected_trace_non_square).abs() < 1e-10,
            "Trace of non-square matrix: expected {}, got {}",
            expected_trace_non_square,
            computed_trace_non_square
        );

        // Define a matrix with no diagonal (nrows = 0 or ncols = 0)
        let data_empty = vec![]; // 0x0 matrix
        let mat_empty = create_faer_matrix(data_empty);
        let mat_ref_empty: &dyn Matrix<Scalar = f64> = &mat_empty;

        // Expected trace: 0.0
        let expected_trace_empty = 0.0;
        let computed_trace_empty = mat_ref_empty.trace();
        assert!(
            (computed_trace_empty - expected_trace_empty).abs() < 1e-10,
            "Trace of empty matrix: expected {}, got {}",
            expected_trace_empty,
            computed_trace_empty
        );

        // Define a 3x2 matrix
        let data_rect = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let mat_rect = create_faer_matrix(data_rect);
        let mat_ref_rect: &dyn Matrix<Scalar = f64> = &mat_rect;

        // Expected trace: 1.0 + 4.0 = 5.0 (min(nrows, ncols) = 2)
        let expected_trace_rect = 5.0;
        let computed_trace_rect = mat_ref_rect.trace();
        assert!(
            (computed_trace_rect - expected_trace_rect).abs() < 1e-10,
            "Trace of 3x2 matrix: expected {}, got {}",
            expected_trace_rect,
            computed_trace_rect
        );
    }

    #[test]
    fn test_frobenius_norm() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)
        // = sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81)
        // = sqrt(285) ≈ 16.881943016134134
        let expected_fro_norm_square = 16.881943016134134;
        let computed_fro_norm_square = mat_ref_square.frobenius_norm();
        assert!(
            (computed_fro_norm_square - expected_fro_norm_square).abs() < 1e-5,
            "Frobenius norm of square matrix: expected {}, got {}",
            expected_fro_norm_square,
            computed_fro_norm_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected Frobenius norm: sqrt(10^2 + 20^2 + 30^2 + 40^2 + 50^2 + 60^2)
        // = sqrt(100 + 400 + 900 + 1600 + 2500 + 3600)
        // = sqrt(9100) ≈ 95.394
        let expected_fro_norm_non_square = 95.39392014169457;
        let computed_fro_norm_non_square = mat_ref_non_square.frobenius_norm();
        assert!(
            (computed_fro_norm_non_square - expected_fro_norm_non_square).abs() < 1e-5,
            "Frobenius norm of non-square matrix: expected {}, got {}",
            expected_fro_norm_non_square,
            computed_fro_norm_non_square
        );

        // Define a zero matrix (0x0)
        let data_empty = vec![]; // 0x0 matrix
        let mat_empty = create_faer_matrix(data_empty);
        let mat_ref_empty: &dyn Matrix<Scalar = f64> = &mat_empty;

        // Expected Frobenius norm: 0.0
        let expected_fro_norm_empty = 0.0;
        let computed_fro_norm_empty = mat_ref_empty.frobenius_norm();
        assert!(
            (computed_fro_norm_empty - expected_fro_norm_empty).abs() < 1e-5,
            "Frobenius norm of empty matrix: expected {}, got {}",
            expected_fro_norm_empty,
            computed_fro_norm_empty
        );

        // Define a 3x2 matrix
        let data_rect = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let mat_rect = create_faer_matrix(data_rect);
        let mat_ref_rect: &dyn Matrix<Scalar = f64> = &mat_rect;

        // Expected Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)
        // = sqrt(1 + 4 + 9 + 16 + 25 + 36)
        // = sqrt(91) ≈ 9.539392014169456
        let expected_fro_norm_rect = 9.539392014169456;
        let computed_fro_norm_rect = mat_ref_rect.frobenius_norm();
        assert!(
            (computed_fro_norm_rect - expected_fro_norm_rect).abs() < 1e-05,
            "Frobenius norm of 3x2 matrix: expected {}, got {}",
            expected_fro_norm_rect,
            computed_fro_norm_rect
        );
    }

    #[test]
    fn test_matrix_as_slice() {
        // Properly initialize the 2x2 matrix with zero values
        let mut matrix = Mat::<f64>::zeros(2, 2);
    
        // Write values safely into the matrix
        matrix.write(0, 0, 1.0);  // Ensure proper error handling with unwrap
        matrix.write(0, 1, 2.0);  // Each write must be checked
        matrix.write(1, 0, 3.0);
        matrix.write(1, 1, 4.0);
    
        // Expected slice in row-major order: [1.0, 2.0, 3.0, 4.0]
        let expected_slice: Box<[f64]> = Box::new([1.0, 2.0, 3.0, 4.0]);
    
        // Explicitly call the `as_slice` method from the Matrix trait
        let slice = crate::linalg::matrix::traits::Matrix::as_slice(&matrix);
    
        // Check that the slice contains the expected values
        assert_eq!(slice, expected_slice);
    }

    #[test]
    fn test_thread_safety2() {
        use std::thread;

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref = Arc::new(mat);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let mat_clone = Arc::clone(&mat_ref);
                thread::spawn(move || {
                    // Define vector x using Vec<f64>
                    let x = vec![1.0, 0.0, -1.0];
                    let x_ref: &dyn Vector<Scalar = f64> = &x;

                    // Initialize vector y using Vec<f64>
                    let mut y = vec![0.0; mat_clone.nrows()];
                    let y_ref: &mut dyn Vector<Scalar = f64> = &mut y;

                    // Perform y = A * x
                    mat_clone.mat_vec(x_ref, y_ref);

                    // Expected result is [-2.0, -2.0, -2.0]
                    let expected = vec![-2.0, -2.0, -2.0];

                    for (i, &val) in expected.iter().enumerate() {
                        assert!(
                            (y[i] - val).abs() < 1e-10,
                            "y[{}] = {}, expected {}",
                            i,
                            y[i],
                            val
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_trace2() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected trace: 1.0 + 5.0 + 9.0 = 15.0
        let expected_trace_square = 15.0;
        let computed_trace_square = mat_ref_square.trace();
        assert!(
            (computed_trace_square - expected_trace_square).abs() < 1e-10,
            "Trace of square matrix: expected {}, got {}",
            expected_trace_square,
            computed_trace_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected trace: 10.0 + 50.0 = 60.0 (min(nrows, ncols) = 2)
        let expected_trace_non_square = 60.0;
        let computed_trace_non_square = mat_ref_non_square.trace();
        assert!(
            (computed_trace_non_square - expected_trace_non_square).abs() < 1e-10,
            "Trace of non-square matrix: expected {}, got {}",
            expected_trace_non_square,
            computed_trace_non_square
        );
    }

    #[test]
    fn test_frobenius_norm2() {
        // Define a square matrix
        let data_square = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat_square = create_faer_matrix(data_square);
        let mat_ref_square: &dyn Matrix<Scalar = f64> = &mat_square;

        // Expected Frobenius norm: sqrt(1^2 + 2^2 + ... + 9^2)
        let expected_fro_norm_square = 16.881943016134134;
        let computed_fro_norm_square = mat_ref_square.frobenius_norm();
        assert!(
            (computed_fro_norm_square - expected_fro_norm_square).abs() < 1e-5,
            "Frobenius norm of square matrix: expected {}, got {}",
            expected_fro_norm_square,
            computed_fro_norm_square
        );

        // Define a non-square matrix (2x3)
        let data_non_square = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        let mat_non_square = create_faer_matrix(data_non_square);
        let mat_ref_non_square: &dyn Matrix<Scalar = f64> = &mat_non_square;

        // Expected Frobenius norm: sqrt(10^2 + 20^2 + ... + 60^2)
        let expected_fro_norm_non_square = 95.39392014169457;
        let computed_fro_norm_non_square = mat_ref_non_square.frobenius_norm();
        assert!(
            (computed_fro_norm_non_square - expected_fro_norm_non_square).abs() < 1e-5,
            "Frobenius norm of non-square matrix: expected {}, got {}",
            expected_fro_norm_non_square,
            computed_fro_norm_non_square
        );
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_row2() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds row should panic
        mat_ref.get(2, 1);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds_column2() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let mat = create_faer_matrix(data);
        let mat_ref: &dyn Matrix<Scalar = f64> = &mat;

        // Accessing out-of-bounds column should panic
        mat_ref.get(1, 2);
    }
}

```

Generate a detailed users guide for the `Linear Algebra` module for Hydra. I am going to provide the code for all of the parts of the `Linear Algebra` module below, and you can analyze and build the detailed outline based on this version of the source code.