use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use std::ops::{AddAssign, Mul};
use std::ops::{Add, Sub, Neg, Div};

/// Represents a 3D vector with three floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Vector3(pub [f64; 3]);

impl AddAssign for Vector3 {
    /// Implements addition assignment for `Vector3`.
    /// Adds the components of another `Vector3` to this vector component-wise.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    /// Multiplies each component of the vector by a scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector3([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Vector3 {
    /// Provides an iterator over the vector's components.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for Vector3 {
    type Output = f64;

    /// Indexes into the vector by position.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Vector3 {
    /// Provides mutable access to the indexed component of the vector.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Vector3 {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 3>;

    /// Converts the vector into an iterator of its components.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector3 {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    /// Converts a reference to the vector into an iterator of its components.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Represents a 3x3 tensor with floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    /// Implements addition assignment for `Tensor3x3`.
    /// Adds the components of another tensor to this tensor component-wise.
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
    /// Multiplies each component of the tensor by a scalar `rhs`.
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
#[derive(Clone, Copy, Debug)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    /// Implements addition assignment for `Scalar`.
    /// Adds another scalar value to this scalar.
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    /// Implements scalar multiplication for `Scalar`.
    /// Multiplies this scalar by another scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

/// Represents a 2D vector with two floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Vector2(pub [f64; 2]);

impl AddAssign for Vector2 {
    /// Implements addition assignment for `Vector2`.
    /// Adds the components of another `Vector2` to this vector component-wise.
    fn add_assign(&mut self, other: Self) {
        for i in 0..2 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    /// Implements scalar multiplication for `Vector2`.
    /// Multiplies each component of the vector by a scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector2([self.0[0] * rhs, self.0[1] * rhs])
    }
}

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.
#[derive(Clone, Debug)]
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.
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
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves a copy of the data associated with the specified `MeshEntity`, if it exists.
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates all data values in the section in parallel using the provided function.
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
    {
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Updates the section by adding the derivative multiplied by a time step `dt`.
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) {
        for entry in derivative.data.iter() {
            let entity = entry.key();
            let deriv_value = entry.value().clone() * dt;
            if let Some(mut state_value) = self.data.get_mut(entity) {
                *state_value.value_mut() += deriv_value;
            } else {
                self.data.insert(*entity, deriv_value);
            }
        }
    }

    /// Returns a list of all `MeshEntity` objects associated with this section.
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Returns all data stored in the section as a vector of immutable copies.
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Clears all data from the section.
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Scales all data values in the section by the specified factor.
    pub fn scale(&self, factor: f64) {
        self.parallel_update(|value| {
            *value = value.clone() * factor;
        });
    }
}

// Add for Section<Scalar>
impl Add for Section<Scalar> {
    type Output = Section<Scalar>;

    fn add(self, rhs: Self) -> Self::Output {
        let result = self.clone();
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 += value.0;
            } else {
                result.set_data(*key, *value);
            }
        }
        result
    }
}


// Sub for Section<Scalar>
impl Sub for Section<Scalar> {
    type Output = Section<Scalar>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result = self.clone();
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 -= value.0;
            } else {
                result.set_data(*key, Scalar(-value.0));
            }
        }
        result
    }
}


// Neg for Section<Scalar>
impl Neg for Section<Scalar> {
    type Output = Section<Scalar>;

    fn neg(self) -> Self::Output {
        let result = self.clone();
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 = -value.0;
        }
        result
    }
}


// Div for Section<Scalar>
impl Div<f64> for Section<Scalar> {
    type Output = Section<Scalar>;

    fn div(self, rhs: f64) -> Self::Output {
        let result = self.clone();
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 /= rhs;
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    // Helper function to create a MeshEntity for testing
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id) // Adjust according to the MeshEntity variant in your implementation
    }

    #[test]
    fn test_vector3_add_assign() {
        let mut v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([0.5, 0.5, 0.5]);
        v1 += v2;

        assert_eq!(v1.0, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_vector3_mul() {
        let v = Vector3([1.0, 2.0, 3.0]);
        let scaled = v * 2.0;

        assert_eq!(scaled.0, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor3x3_add_assign() {
        let mut t1 = Tensor3x3([[1.0; 3]; 3]);
        let t2 = Tensor3x3([[0.5; 3]; 3]);
        t1 += t2;

        assert_eq!(t1.0, [[1.5; 3]; 3]);
    }

    #[test]
    fn test_tensor3x3_mul() {
        let t = Tensor3x3([[1.0; 3]; 3]);
        let scaled = t * 2.0;

        assert_eq!(scaled.0, [[2.0; 3]; 3]);
    }

    #[test]
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
    fn test_section_parallel_update() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=10).map(create_test_mesh_entity).collect();

        for (i, entity) in entities.iter().enumerate() {
            section.set_data(*entity, Scalar(i as f64));
        }

        section.parallel_update(|value| {
            value.0 *= 2.0;
        });

        for (i, entity) in entities.iter().enumerate() {
            assert_eq!(section.restrict(entity).unwrap().0, (i as f64) * 2.0);
        }
    }

    #[test]
    fn test_section_update_with_derivative() {
        let section: Section<Scalar> = Section::new();
        let derivative: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);

        section.set_data(entity, Scalar(1.0));
        derivative.set_data(entity, Scalar(0.5));

        section.update_with_derivative(&derivative, 2.0);

        assert_eq!(section.restrict(&entity).unwrap().0, 2.0);
    }

    #[test]
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
    fn test_section_clear() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        section.clear();

        assert!(section.restrict(&entity).is_none());
    }

    #[test]
    fn test_section_scale() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(2.0));

        section.scale(3.0);

        assert_eq!(section.restrict(&entity).unwrap().0, 6.0);
    }

    // Debugging utilities for better output on failure
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
    fn test_debugging_output() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        debug_section_data(&section);
    }
}
