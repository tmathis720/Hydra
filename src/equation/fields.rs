use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};
use super::super::domain::section::{Vector3, Tensor3x3, Scalar, Vector2};

/// Trait `UpdateState` defines methods for updating and comparing the state of objects.
pub trait UpdateState {
    /// Updates the state of an object using a derivative and a time step (`dt`).
    fn update_state(&mut self, derivative: &Self, dt: f64);

    /// Computes the difference between the current state and another state.
    fn difference(&self, other: &Self) -> Self;

    /// Computes the norm (magnitude) of the state for convergence checks.
    fn norm(&self) -> f64;
}

/// Represents the fields (scalar, vector, and tensor) stored for a simulation domain.
#[derive(Clone, Debug)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<Scalar>>, // Scalar fields (e.g., temperature, energy).
    pub vector_fields: FxHashMap<String, Section<Vector3>>, // Vector fields (e.g., velocity, momentum).
    pub tensor_fields: FxHashMap<String, Section<Tensor3x3>>, // Tensor fields (e.g., stress tensors).
}

impl Fields {
    /// Creates a new, empty instance of `Fields`.
    pub fn new() -> Self {
        Self {
            scalar_fields: FxHashMap::default(),
            vector_fields: FxHashMap::default(),
            tensor_fields: FxHashMap::default(),
        }
    }

    /// Retrieves the scalar field value for a specific entity.
    pub fn get_scalar_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Scalar> {
        self.scalar_fields.get(name)?.restrict(entity)
    }

    /// Sets the scalar field value for a specific entity, creating the field if it doesn't exist.
    pub fn set_scalar_field_value(&mut self, name: &str, entity: MeshEntity, value: Scalar) {
        if let Some(field) = self.scalar_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.scalar_fields.insert(name.to_string(), field);
        }
    }

    /// Retrieves the vector field value for a specific entity.
    pub fn get_vector_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Vector3> {
        self.vector_fields.get(name)?.restrict(entity)
    }

    /// Sets the vector field value for a specific entity, creating the field if it doesn't exist.
    pub fn set_vector_field_value(&mut self, name: &str, entity: MeshEntity, value: Vector3) {
        if let Some(field) = self.vector_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.vector_fields.insert(name.to_string(), field);
        }
    }

    /// Updates the fields using the given fluxes.
    /// This operation typically happens after fluxes are computed across the mesh.
    pub fn update_from_fluxes(&mut self, fluxes: &Fluxes) {
        for entry in fluxes.energy_fluxes.data.iter() {
            let field = self
                .scalar_fields
                .entry("energy".to_string())
                .or_insert_with(Section::new);
            println!(
                "Updating scalar field 'energy' for entity {:?} with value {:?}",
                entry.key(),
                entry.value()
            );
            field.set_data(*entry.key(), *entry.value());
        }

        for entry in fluxes.momentum_fluxes.data.iter() {
            let field = self
                .vector_fields
                .entry("momentum".to_string())
                .or_insert_with(Section::new);
            println!(
                "Updating vector field 'momentum' for entity {:?} with value {:?}",
                entry.key(),
                entry.value()
            );
            field.set_data(*entry.key(), *entry.value());
        }
    }
}

impl UpdateState for Fields {
    /// Updates the current state of scalar and vector fields by applying
    /// the derivative (rate of change) multiplied by a time step (`dt`).
    ///
    /// This function iterates over each field in the `derivative` structure, adding
    /// the scaled derivative values to the corresponding fields in `self`. If a field
    /// does not exist in the current state, it creates a new field and applies the update.
    ///
    /// # Arguments
    /// * `derivative` - A `Fields` instance representing the rate of change for each field.
    /// * `dt` - The time step, used to scale the derivative before updating.
    fn update_state(&mut self, derivative: &Fields, dt: f64) {
        // Update scalar fields
        for (key, section) in &derivative.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get_mut(key) {
                // Update existing scalar field with the derivative
                state_section.update_with_derivative(section, dt);
            } else {
                // Create a new scalar field if it does not exist
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.scalar_fields.insert(key.clone(), new_section);
            }
        }

        // Update vector fields
        for (key, section) in &derivative.vector_fields {
            if let Some(state_section) = self.vector_fields.get_mut(key) {
                // Update existing vector field with the derivative
                state_section.update_with_derivative(section, dt);
            } else {
                // Create a new vector field if it does not exist
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.vector_fields.insert(key.clone(), new_section);
            }
        }
    }

    /// Computes the difference between the current state and another `Fields` instance.
    ///
    /// The difference is calculated for each field by subtracting the values in the `other`
    /// instance from the corresponding fields in `self`. If a field is not present in the
    /// current state, it is skipped.
    ///
    /// # Arguments
    /// * `other` - A `Fields` instance to compare against.
    ///
    /// # Returns
    /// A new `Fields` instance containing the difference for each field.
    fn difference(&self, other: &Self) -> Self {
        let mut result = self.clone();

        // Compute the difference for scalar fields
        for (key, section) in &other.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get(key) {
                result.scalar_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }

        // Compute the difference for vector fields
        for (key, section) in &other.vector_fields {
            if let Some(state_section) = self.vector_fields.get(key) {
                result.vector_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }

        result
    }

    /// Computes the norm (magnitude) of the current state.
    ///
    /// The norm is calculated by summing the squares of all scalar field values and
    /// then taking the square root of the total. This provides a scalar metric
    /// representing the overall magnitude of the state, useful for convergence checks.
    ///
    /// # Returns
    /// A `f64` value representing the norm of the state.
    fn norm(&self) -> f64 {
        self.scalar_fields
            .values()
            .flat_map(|section| {
                // Iterate over each scalar field value, computing its square
                section
                    .data
                    .iter()
                    .map(|entry| entry.value().0 * entry.value().0)
            })
            .sum::<f64>() // Sum all squared values
            .sqrt() // Take the square root of the sum
    }
}


/// Represents flux data for various field quantities.
pub struct Fluxes {
    pub momentum_fluxes: Section<Vector3>, // Fluxes for momentum.
    pub energy_fluxes: Section<Scalar>,   // Fluxes for energy.
    pub turbulence_fluxes: Section<Vector2>, // Fluxes for turbulence models.
}

impl Fluxes {
    /// Creates a new instance of `Fluxes` with empty sections.
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    /// Adds a momentum flux for the given entity.
    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: Vector3) {
        if let Some(mut current) = self.momentum_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    /// Adds an energy flux for the given entity.
    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: Scalar) {
        if let Some(mut current) = self.energy_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

    /// Adds a turbulence flux for the given entity.
    pub fn add_turbulence_flux(&mut self, entity: MeshEntity, value: Vector2) {
        if let Some(mut current) = self.turbulence_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.turbulence_fluxes.set_data(entity, value);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{section::Vector3, MeshEntity};

    fn create_test_entity(id: usize) -> MeshEntity {
        MeshEntity::Cell(id)
    }

    #[test]
    fn test_fields_initialization() {
        let fields = Fields::new();
        assert!(fields.scalar_fields.is_empty());
        assert!(fields.vector_fields.is_empty());
    }

    #[test]
    fn test_set_and_get_scalar_field_value() {
        let mut fields = Fields::new();
        let entity = create_test_entity(1);

        fields.set_scalar_field_value("test", entity, Scalar(1.0));
        let value = fields.get_scalar_field_value("test", &entity);

        assert_eq!(value.unwrap(), Scalar(1.0));
    }

    #[test]
    fn test_set_and_get_vector_field_value() {
        let mut fields = Fields::new();
        let entity = create_test_entity(1);

        fields.set_vector_field_value("velocity", entity, Vector3([1.0, 0.0, 0.0]));
        let value = fields.get_vector_field_value("velocity", &entity);

        assert_eq!(value.unwrap(), Vector3([1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_fluxes_add_and_update_fields() {
        let mut fields = Fields::new();
        let mut fluxes = Fluxes::new();
        let entity = create_test_entity(1);

        fluxes.add_energy_flux(entity, Scalar(2.0));
        fluxes.add_momentum_flux(entity, Vector3([0.5, 0.5, 0.5]));

        fields.update_from_fluxes(&fluxes);

        assert_eq!(
            fields.get_scalar_field_value("energy", &entity).unwrap(),
            Scalar(2.0)
        );
        assert_eq!(
            fields.get_vector_field_value("momentum", &entity).unwrap(),
            Vector3([0.5, 0.5, 0.5])
        );
    }

    #[test]
    fn test_update_state() {
        let mut fields = Fields::new();
        let mut derivative = Fields::new();
        let entity = create_test_entity(1);

        derivative.set_scalar_field_value("test", entity, Scalar(1.0));
        fields.update_state(&derivative, 2.0);

        assert_eq!(fields.get_scalar_field_value("test", &entity).unwrap(), Scalar(2.0));
    }

    #[test]
    fn test_difference_and_norm() {
        let mut fields = Fields::new();
        let mut other = Fields::new();
        let entity = create_test_entity(1);

        fields.set_scalar_field_value("test", entity, Scalar(2.0));
        other.set_scalar_field_value("test", entity, Scalar(1.0));

        let diff = fields.difference(&other);
        assert_eq!(diff.get_scalar_field_value("test", &entity).unwrap(), Scalar(1.0));

        let norm = fields.norm();
        assert!((norm - 2.0).abs() < 1e-6);
    }
}
