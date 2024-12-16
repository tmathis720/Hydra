use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use crate::Vector;
use std::ops::{AddAssign, Mul};
use std::ops::{Add, Sub, Neg, Div};

/// Represents a 3D vector with three floating-point components.
///
/// The `Vector3` struct is a simple abstraction for 3D vectors, providing methods and operator
/// overloads for basic arithmetic operations, indexing, and iteration. This implementation is
/// designed for use in computational geometry, physics simulations, or similar fields requiring
/// manipulation of 3D data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3(pub [f64; 3]);

impl AddAssign for Vector3 {
    /// Implements the `+=` operator for `Vector3`.
    ///
    /// Adds another `Vector3` to this vector component-wise. The operation is performed
    /// in place, modifying the original vector.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    ///
    /// Multiplies each component of the vector by the scalar value `rhs`. The resulting
    /// `Vector3` is a new vector, leaving the original vector unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector3([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Vector3 {
    /// Returns an iterator over the components of the vector.
    ///
    /// The iterator allows read-only access to the vector's components in order.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for Vector3 {
    type Output = f64;

    /// Implements indexing for `Vector3`.
    ///
    /// Allows direct read-only access to a specific component of the vector by its index.
    /// Valid indices are `0`, `1`, and `2` corresponding to the `x`, `y`, and `z` components.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Sub for Vector3 {
    type Output = Vector3;

    /// Implements subtraction for `Vector3`.
    ///
    /// Computes the difference between two vectors component-wise, returning a new vector.
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl Neg for Vector3 {
    type Output = Vector3;

    /// Implements negation for `Vector3`.
    ///
    /// Negates each component of the vector, returning a new vector with opposite direction.
    fn neg(self) -> Self::Output {
        Vector3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl std::ops::IndexMut<usize> for Vector3 {
    /// Implements mutable indexing for `Vector3`.
    ///
    /// Allows direct modification of a specific component of the vector by its index.
    /// Valid indices are `0`, `1`, and `2` corresponding to the `x`, `y`, and `z` components.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Vector3 {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 3>;

    /// Converts the vector into an iterator of its components.
    ///
    /// Consumes the `Vector3` and produces an iterator that yields the `x`, `y`, and `z`
    /// components in order.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector3 {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    /// Converts a reference to the vector into an iterator of its components.
    ///
    /// Produces an iterator that yields immutable references to the `x`, `y`, and `z`
    /// components in order.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Add for Vector3 {
    type Output = Vector3;

    /// Implements addition for `Vector3`.
    ///
    /// Adds two vectors component-wise, returning a new vector with the result.
    fn add(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<'a> Add for &'a Vector3 {
    type Output = Vector3;

    /// Implements addition for references to `Vector3`.
    ///
    /// Adds two vectors component-wise, returning a new `Vector3` without consuming the operands.
    /// This implementation allows adding borrowed `Vector3` instances, which can improve
    /// performance by avoiding unnecessary cloning or copying.
    fn add(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Vector3 {
    /// Computes the magnitude (norm) of the vector.
    pub fn magnitude(&self) -> f64 {
        self.0.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Computes the dot product of two vectors.
    pub fn dot(&self, other: &Vector3) -> f64 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }
}

impl Mul<Vector3> for f64 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    ///
    /// Multiplies a scalar `f64` by a `Vector3`, scaling each component of the vector by the scalar.
    /// This implementation consumes the `Vector3` operand and produces a new scaled vector.
    fn mul(self, rhs: Vector3) -> Self::Output {
        Vector3([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
        ])
    }
}

impl Mul<&Vector3> for f64 {
    type Output = Vector3;

    /// Implements scalar multiplication for a reference to `Vector3`.
    ///
    /// Multiplies a scalar `f64` by a borrowed `Vector3`, scaling each component of the vector
    /// without consuming the `Vector3`.
    fn mul(self, rhs: &Vector3) -> Self::Output {
        Vector3([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
        ])
    }
}

/// Represents a 3x3 tensor with floating-point components.
///
/// The `Tensor3x3` struct is a simple abstraction for rank-2 tensors in 3D space.
/// It supports component-wise arithmetic operations, including addition and scalar multiplication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    /// Implements the `+=` operator for `Tensor3x3`.
    ///
    /// Adds another `Tensor3x3` to this tensor component-wise. The operation is performed
    /// in place, modifying the original tensor.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] += other.0[i][j];
            }
        }
    }
}

impl Mul<f64> for Tensor3x3 {
    type Output = Tensor3x3;

    /// Implements scalar multiplication for `Tensor3x3`.
    ///
    /// Multiplies each component of the tensor by a scalar `rhs`. The resulting tensor
    /// is a new `Tensor3x3`, leaving the original tensor unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.0[i][j] * rhs;
            }
        }
        Tensor3x3(result)
    }
}

/// Represents a scalar value.
///
/// The `Scalar` struct is a wrapper for a floating-point value, used in mathematical
/// and physical computations where type safety or domain-specific semantics are desired.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    /// Implements the `+=` operator for `Scalar`.
    ///
    /// Adds another scalar to this scalar. The operation is performed in place.
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    /// Implements scalar multiplication for `Scalar`.
    ///
    /// Multiplies the scalar value by another scalar `rhs`. The resulting value
    /// is wrapped in a new `Scalar`.
    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

/// Represents a 2D vector with two floating-point components.
///
/// The `Vector2` struct is a simple abstraction for 2D vectors, providing methods and
/// operator overloads for basic arithmetic operations. It is suitable for computations
/// in 2D geometry, physics, or graphics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector2(pub [f64; 2]);

impl AddAssign for Vector2 {
    /// Implements the `+=` operator for `Vector2`.
    ///
    /// Adds another `Vector2` to this vector component-wise. The operation is performed
    /// in place, modifying the original vector.
    fn add_assign(&mut self, other: Self) {
        for i in 0..2 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    /// Implements scalar multiplication for `Vector2`.
    ///
    /// Multiplies each component of the vector by a scalar `rhs`. The resulting vector
    /// is a new `Vector2`, leaving the original vector unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector2([self.0[0] * rhs, self.0[1] * rhs])
    }
}


/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.
///
/// The `Section` structure is designed to store data (of generic type `T`) linked to entities
/// in a computational mesh (`MeshEntity`). It provides methods for efficient data management,
/// parallel updates, and mathematical operations. This abstraction is particularly useful
/// in simulations and finite element/volume computations where values like scalars or vectors
/// are associated with mesh components.
#[derive(Clone, Debug)]
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.
    ///
    /// The `DashMap` ensures thread-safe operations and allows concurrent reads and writes
    /// on the data without explicit locking, making it ideal for parallel computations.
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T>
where
    T: Clone + AddAssign + Mul<f64, Output = T> + Send + Sync,
{
    /// Creates a new `Section` with an empty data map.
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Associates a given `MeshEntity` with a value of type `T`.
    ///
    /// If the `MeshEntity` already exists in the section, its value is overwritten.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` to associate with the value.
    /// - `value`: The value of type `T` to store.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves a copy of the data associated with the specified `MeshEntity`, if it exists.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` whose data is being requested.
    ///
    /// # Returns
    /// An `Option<T>` containing the associated value if it exists, or `None` if the entity
    /// is not in the section.
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates all data values in the section in parallel using the provided function.
    ///
    /// # Parameters
    /// - `update_fn`: A function that takes a mutable reference to a value of type `T`
    ///   and updates it. This function must be thread-safe (`Sync` + `Send`) as updates
    ///   are applied concurrently.
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
    {
        // Collect all keys to avoid holding references during parallel iteration.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Update values in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Updates the section by adding the derivative multiplied by a time step `dt`.
    ///
    /// This method performs an in-place update of the section's values, adding the product
    /// of a derivative (from another section) and a scalar time step `dt`. If an entity
    /// exists in the derivative but not in the current section, it is added.
    ///
    /// # Parameters
    /// - `derivative`: A `Section` containing the derivative values.
    /// - `dt`: A scalar value representing the time step.
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) {
        for entry in derivative.data.iter() {
            let entity = entry.key();
            let deriv_value = entry.value().clone() * dt;

            // Update existing value or insert a new one.
            if let Some(mut state_value) = self.data.get_mut(entity) {
                *state_value.value_mut() += deriv_value;
            } else {
                self.data.insert(*entity, deriv_value);
            }
        }
    }

    /// Returns a list of all `MeshEntity` objects associated with this section.
    ///
    /// # Returns
    /// A `Vec<MeshEntity>` containing all the keys from the section's data map.
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Returns all data stored in the section as a vector of immutable copies.
    ///
    /// # Returns
    /// A `Vec<T>` containing all the values stored in the section.
    /// Requires `T` to implement `Clone`.
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Clears all data from the section.
    ///
    /// This method removes all entries from the section, leaving it empty.
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Scales all data values in the section by the specified factor.
    ///
    /// This method multiplies each value in the section by the given scalar factor.
    /// The updates are applied in parallel for efficiency.
    ///
    /// # Parameters
    /// - `factor`: The scalar value by which to scale all entries.
    pub fn scale(&self, factor: f64) {
        self.parallel_update(|value| {
            *value = value.clone() * factor;
        });
    }
}

// Add for Section<Scalar>
impl Add for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements addition for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise addition of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are added. If a key exists
    /// in only one section, its value is copied to the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the sum of the two sections.
    fn add(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 += value.0; // Add values if the key exists in both sections
            } else {
                result.set_data(*key, *value); // Insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Sub for Section<Scalar>
impl Sub for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements subtraction for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 -= value.0; // Subtract values if the key exists in both sections
            } else {
                result.set_data(*key, Scalar(-value.0)); // Negate and insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Neg for Section<Scalar>
impl Neg for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements negation for `Section<Scalar>`.
    ///
    /// This operator negates each value in the `Section<Scalar>` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values negated.
    fn neg(self) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 = -value.0; // Negate the scalar value
        }
        result
    }
}

// Div for Section<Scalar>
impl Div<f64> for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements scalar division for `Section<Scalar>`.
    ///
    /// Divides each value in the `Section<Scalar>` by a scalar `rhs` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    /// - `rhs`: A scalar `f64` divisor.
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values scaled by `1/rhs`.
    fn div(self, rhs: f64) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 /= rhs; // Divide the scalar value by `rhs`
        }
        result
    }
}

// Sub for Section<Vector3>
impl Sub for Section<Vector3> {
    type Output = Section<Vector3>;

    /// Implements subtraction for `Section<Vector3>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Vector3>` instances.
    /// If a key exists in both sections, their corresponding vectors are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Vector3>` operand (consumed).
    /// - `rhs`: The second `Section<Vector3>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Vector3>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = Section::new(); // Create a new section to hold the result

        // Process all keys from the `rhs` section
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair();
            if let Some(current) = self.data.get(key) {
                result.set_data(*key, *current.value() - *value); // Subtract if the key exists in both sections
            } else {
                result.set_data(*key, -*value); // Negate and add if the key only exists in `rhs`
            }
        }

        // Process all keys from the `self` section that are not in `rhs`
        for entry in self.data.iter() {
            let (key, value) = entry.pair();
            if !rhs.data.contains_key(key) {
                result.set_data(*key, *value); // Add the value if the key only exists in `self`
            }
        }

        result
    }
}

impl Vector for Section<Scalar> {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, i: usize) -> Self::Scalar {
        let keys: Vec<_> = self.data.iter().map(|entry| entry.key().clone()).collect();
        let key = keys.get(i).expect("Index out of bounds");
        self.data
            .get(key)
            .map(|v| v.0) // Access the scalar value from Scalar(f64)
            .expect("Key not found in section data")
    }

    fn set(&mut self, i: usize, value: Self::Scalar) {
        let keys: Vec<_> = self.data.iter().map(|entry| entry.key().clone()).collect();
        let key = keys.get(i).expect("Index out of bounds");
        if let Some(mut entry) = self.data.get_mut(key) {
            entry.value_mut().0 = value; // Set the scalar value in Scalar(f64)
        } else {
            panic!("Key not found in section data");
        }
    }

    fn as_slice(&self) -> &[Self::Scalar] {
        panic!("Section does not support contiguous slices due to its DashMap-based structure.")
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Scalar] {
        panic!("Section does not support mutable slices due to its DashMap-based structure.")
    }

    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar {
        self.data
            .iter()
            .map(|entry| {
                let id = entry.key().get_id();
                let self_value = entry.value().0;
                let other_value = other.get(id);
                self_value * other_value
            })
            .sum()
    }

    fn norm(&self) -> Self::Scalar {
        self.data
            .iter()
            .map(|entry| entry.value().0.powi(2))
            .sum::<Self::Scalar>()
            .sqrt()
    }

    fn scale(&mut self, scalar: Self::Scalar) {
        self.data.iter_mut().for_each(|mut entry| {
            entry.value_mut().0 *= scalar;
        });
    }

    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>) {
        self.data.iter_mut().for_each(|mut entry| {
            let id = entry.key().get_id();
            entry.value_mut().0 += a * x.get(id);
        });
    }

    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) {
        self.data.iter_mut().for_each(|mut entry| {
            let id = entry.key().get_id();
            entry.value_mut().0 += other.get(id);
        });
    }

    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) {
        self.data.iter_mut().for_each(|mut entry| {
            let id = entry.key().get_id();
            entry.value_mut().0 *= other.get(id);
        });
    }

    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) {
        self.data.iter_mut().for_each(|mut entry| {
            let id = entry.key().get_id();
            let divisor = other.get(id);
            if divisor == 0.0 {
                panic!("Division by zero");
            }
            entry.value_mut().0 /= divisor;
        });
    }

    fn cross(&mut self, _other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str> {
        Err("Cross product is not defined for scalars")
    }

    fn sum(&self) -> Self::Scalar {
        self.data.iter().map(|entry| entry.value().0).sum()
    }

    fn max(&self) -> Self::Scalar {
        self.data
            .iter()
            .map(|entry| entry.value().0)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn min(&self) -> Self::Scalar {
        self.data
            .iter()
            .map(|entry| entry.value().0)
            .fold(f64::INFINITY, f64::min)
    }

    fn mean(&self) -> Self::Scalar {
        let sum: Self::Scalar = self.sum();
        sum / self.len() as Self::Scalar
    }

    fn variance(&self) -> Self::Scalar {
        let mean = self.mean();
        self.data
            .iter()
            .map(|entry| (entry.value().0 - mean).powi(2))
            .sum::<Self::Scalar>()
            / self.len() as Self::Scalar
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    /// Helper function to create a `MeshEntity` for testing purposes.
    ///
    /// This function creates a `Vertex` variant of `MeshEntity` with the given ID.
    /// Adjust this function if other `MeshEntity` variants need to be tested.
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id)
    }

    #[test]
    /// Tests the addition of two `Vector3` instances.
    ///
    /// Validates that component-wise addition is performed correctly.
    fn test_vector3_add() {
        let v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([4.0, 5.0, 6.0]);
        let result = v1 + v2;

        assert_eq!(result.0, [5.0, 7.0, 9.0]);
    }

    #[test]
    /// Tests the `+=` operation for `Vector3`.
    ///
    /// Validates that component-wise addition is correctly applied in-place.
    fn test_vector3_add_assign() {
        let mut v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([0.5, 0.5, 0.5]);
        v1 += v2;

        assert_eq!(v1.0, [1.5, 2.5, 3.5]);
    }

    #[test]
    /// Tests scalar multiplication for `Vector3`.
    ///
    /// Validates that each component of the vector is scaled correctly.
    fn test_vector3_mul() {
        let v = Vector3([1.0, 2.0, 3.0]);
        let scaled = v * 2.0;

        assert_eq!(scaled.0, [2.0, 4.0, 6.0]);
    }

    #[test]
    /// Tests the `+=` operation for `Tensor3x3`.
    ///
    /// Validates that component-wise addition is applied correctly to tensors.
    fn test_tensor3x3_add_assign() {
        let mut t1 = Tensor3x3([[1.0; 3]; 3]);
        let t2 = Tensor3x3([[0.5; 3]; 3]);
        t1 += t2;

        assert_eq!(t1.0, [[1.5; 3]; 3]);
    }

    #[test]
    /// Tests scalar multiplication for `Tensor3x3`.
    ///
    /// Validates that all components of the tensor are scaled correctly.
    fn test_tensor3x3_mul() {
        let t = Tensor3x3([[1.0; 3]; 3]);
        let scaled = t * 2.0;

        assert_eq!(scaled.0, [[2.0; 3]; 3]);
    }

    #[test]
    /// Tests setting and retrieving data in a `Section<Scalar>`.
    ///
    /// Ensures that values can be stored and accessed correctly by their associated `MeshEntity`.
    fn test_section_set_and_restrict_data() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        let value = Scalar(3.14);

        section.set_data(entity, value);
        let retrieved = section.restrict(&entity);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, 3.14);
    }

    #[test]
    /// Tests parallel updates for a `Section<Scalar>`.
    ///
    /// Validates that all values in the section are updated correctly and efficiently in parallel.
    fn test_section_parallel_update() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=10).map(create_test_mesh_entity).collect();

        for (i, entity) in entities.iter().enumerate() {
            section.set_data(*entity, Scalar(i as f64));
        }

        section.parallel_update(|value| {
            value.0 *= 2.0; // Double each value
        });

        for (i, entity) in entities.iter().enumerate() {
            assert_eq!(section.restrict(entity).unwrap().0, (i as f64) * 2.0);
        }
    }

    #[test]
    /// Tests updating a `Section<Scalar>` with a derivative section.
    ///
    /// Ensures that the update correctly adds the scaled derivative to the section's values.
    fn test_section_update_with_derivative() {
        let section: Section<Scalar> = Section::new();
        let derivative: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);

        section.set_data(entity, Scalar(1.0));
        derivative.set_data(entity, Scalar(0.5));

        section.update_with_derivative(&derivative, 2.0); // Time step is 2.0

        assert_eq!(section.restrict(&entity).unwrap().0, 2.0);
    }

    #[test]
    /// Tests retrieving all `MeshEntity` objects from a `Section<Scalar>`.
    ///
    /// Ensures that all entities stored in the section are returned correctly.
    fn test_section_entities() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=5).map(create_test_mesh_entity).collect();

        for entity in &entities {
            section.set_data(*entity, Scalar(1.0));
        }

        let retrieved_entities = section.entities();
        assert_eq!(retrieved_entities.len(), entities.len());
    }

    #[test]
    /// Tests clearing all data from a `Section<Scalar>`.
    ///
    /// Validates that the section becomes empty after the clear operation.
    fn test_section_clear() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        section.clear();

        assert!(section.restrict(&entity).is_none());
    }

    #[test]
    /// Tests scaling all values in a `Section<Scalar>`.
    ///
    /// Ensures that all values are scaled correctly by the given factor.
    fn test_section_scale() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(2.0));

        section.scale(3.0); // Scale by 3.0

        assert_eq!(section.restrict(&entity).unwrap().0, 6.0);
    }

    /// Utility function for debugging `Section` contents during test failures.
    ///
    /// Prints all key-value pairs in the `Section` for inspection.
    /// This function is not part of the test suite but can be used for debugging purposes.
    fn debug_section_data<T>(section: &Section<T>)
    where
        T: std::fmt::Debug,
    {
        println!("Section data:");
        for entry in section.data.iter() {
            println!("{:?} -> {:?}", entry.key(), entry.value());
        }
    }

    #[test]
    /// Example test to demonstrate debugging output for `Section` contents.
    ///
    /// Useful for inspecting data during test failures.
    fn test_debugging_output() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        debug_section_data(&section);
    }

    #[test]
    fn test_vector_trait_for_section() {
        use crate::domain::mesh_entity::MeshEntity;

        // Create a new section
        let mut section = Section::new();
        section.set_data(MeshEntity::Cell(0), Scalar(1.0));
        section.set_data(MeshEntity::Cell(1), Scalar(2.0));
        section.set_data(MeshEntity::Cell(2), Scalar(3.0));

        // Test `len`
        assert_eq!(section.len(), 3);

        // Test `get`
        assert_eq!(section.get(0), 1.0);
        assert_eq!(section.get(1), 2.0);
        assert_eq!(section.get(2), 3.0);

        // Test `set`
        section.set(1, 5.0);
        assert_eq!(section.get(1), 5.0);

        // Test `dot`
        let other = Section::new();
        other.set_data(MeshEntity::Cell(0), Scalar(2.0));
        other.set_data(MeshEntity::Cell(1), Scalar(3.0));
        other.set_data(MeshEntity::Cell(2), Scalar(4.0));
        assert_eq!(section.dot(&other), 1.0 * 2.0 + 5.0 * 3.0 + 3.0 * 4.0);
    }

}
