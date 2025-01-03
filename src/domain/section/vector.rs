use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use crate::{Section, Vector};

use super::scalar::Scalar;


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
