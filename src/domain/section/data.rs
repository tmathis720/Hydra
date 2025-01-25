use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use core::fmt;
use std::ops::{AddAssign, Mul};

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

/// Enum to represent errors in `Section` operations.
#[derive(Debug)]
pub enum SectionError {
    EntityNotFound(MeshEntity),
    InvalidOperation(String),
    DivisionByZero,
}

impl fmt::Display for SectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SectionError::EntityNotFound(entity) => {
                write!(f, "Entity {:?} not found in Section", entity)
            }
            SectionError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            SectionError::DivisionByZero => write!(f, "Division by zero encountered"),
        }
    }
}

impl std::error::Error for SectionError {}


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
    /// A `Result<T, SectionError>` containing the associated value if it exists, or an error
    /// if the entity is not in the section.
    pub fn restrict(&self, entity: &MeshEntity) -> Result<T, SectionError> {
        self.data
            .get(entity)
            .map(|v| v.clone())
            .ok_or_else(|| SectionError::EntityNotFound(entity.clone()))
    }

    /// Updates all data values in the section in parallel using the provided function.
    ///
    /// # Parameters
    /// - `update_fn`: A function that takes a mutable reference to a value of type `T`
    ///   and updates it. This function must be thread-safe (`Sync` + `Send`) as updates
    ///   are applied concurrently.
    ///
    /// # Returns
    /// A `Result<(), SectionError>` indicating success or failure.
    ///
    /// # Errors
    /// Returns a `SectionError::EntityNotFound` if any key is missing during the update.
    pub fn parallel_update<F>(&self, update_fn: F) -> Result<(), SectionError>
    where
        F: Fn(&mut T) + Sync + Send,
    {
        // Collect all keys to avoid holding references during parallel iteration.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Update values in parallel.
        let results: Vec<Result<(), SectionError>> = keys
            .into_par_iter()
            .map(|key| {
                if let Some(mut entry) = self.data.get_mut(&key) {
                    update_fn(entry.value_mut());
                    Ok(())
                } else {
                    Err(SectionError::EntityNotFound(key))
                }
            })
            .collect();

        // Combine results, returning the first error if any occur.
        results.into_iter().collect::<Result<(), SectionError>>()
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
    ///
    /// # Returns
    /// A `Result<(), SectionError>` indicating success or failure.
    ///
    /// # Errors
    /// Returns a `SectionError::EntityNotFound` if the derivative contains an invalid entity.
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) -> Result<(), SectionError> {
        let results: Vec<Result<(), SectionError>> = derivative
            .data
            .iter()
            .map(|entry| {
                let entity = entry.key();
                let deriv_value = entry.value().clone() * dt;

                // Update existing value or insert a new one.
                if let Some(mut state_value) = self.data.get_mut(entity) {
                    *state_value.value_mut() += deriv_value;
                    Ok(())
                } else {
                    self.data.insert(*entity, deriv_value);
                    Ok(())
                }
            })
            .collect();

        // Combine results, returning the first error if any occur.
        results.into_iter().collect::<Result<(), SectionError>>()
    }

    /// Returns a list of all `MeshEntity` objects associated with this section.
    ///
    /// # Returns
    /// A `Result<Vec<MeshEntity>, SectionError>` containing all the keys from the section's data map,
    /// or an error if the section is empty.
    pub fn entities(&self) -> Result<Vec<MeshEntity>, SectionError> {
        let entities: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        if entities.is_empty() {
            Err(SectionError::InvalidOperation(
                "Section contains no entities".to_string(),
            ))
        } else {
            Ok(entities)
        }
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
    ///
    /// # Returns
    /// A `Result<(), SectionError>` indicating success or failure.
    pub fn scale(&self, factor: f64) -> Result<(), SectionError> {
        self.parallel_update(|value| {
            *value = value.clone() * factor;
        })
    }
}
