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
