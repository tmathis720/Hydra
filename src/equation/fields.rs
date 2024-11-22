use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};
use super::super::domain::section::{Vector3, Tensor3x3, Scalar, Vector2};

pub trait UpdateState {
    fn update_state(&mut self, derivative: &Self, dt: f64);
    fn difference(&self, other: &Self) -> Self; // Returns the difference between states
    fn norm(&self) -> f64; // Returns the norm (magnitude) of the state
}

#[derive(Clone, Debug)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<Scalar>>,
    pub vector_fields: FxHashMap<String, Section<Vector3>>,
    pub tensor_fields: FxHashMap<String, Section<Tensor3x3>>,
}


impl Fields {
    pub fn new() -> Self {
        Self {
            scalar_fields: FxHashMap::default(),
            vector_fields: FxHashMap::default(),
            tensor_fields: FxHashMap::default(),
        }
    }

    pub fn get_scalar_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Scalar> {
        self.scalar_fields.get(name)?.restrict(entity)
    }

    pub fn set_scalar_field_value(&mut self, name: &str, entity: MeshEntity, value: Scalar) {
        if let Some(field) = self.scalar_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.scalar_fields.insert(name.to_string(), field);
        }
    }

    pub fn get_vector_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Vector3> {
        self.vector_fields.get(name)?.restrict(entity)
    }

    pub fn set_vector_field_value(&mut self, name: &str, entity: MeshEntity, value: Vector3) {
        if let Some(field) = self.vector_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.vector_fields.insert(name.to_string(), field);
        }
    }

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
    fn update_state(&mut self, derivative: &Fields, dt: f64) {
        for (key, section) in &derivative.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            } else {
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.scalar_fields.insert(key.clone(), new_section);
            }
        }
        for (key, section) in &derivative.vector_fields {
            if let Some(state_section) = self.vector_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            } else {
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.vector_fields.insert(key.clone(), new_section);
            }
        }
    }

    fn difference(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (key, section) in &other.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get(key) {
                result.scalar_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }
        for (key, section) in &other.vector_fields {
            if let Some(state_section) = self.vector_fields.get(key) {
                result.vector_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }
        result
    }

    fn norm(&self) -> f64 {
        self.scalar_fields
            .values()
            .flat_map(|section| {
                section
                    .data
                    .iter()
                    .map(|entry| entry.value().0 * entry.value().0)
            })
            .sum::<f64>()
            .sqrt()
    }
}

pub struct Fluxes {
    pub momentum_fluxes: Section<Vector3>,
    pub energy_fluxes: Section<Scalar>,
    pub turbulence_fluxes: Section<Vector2>,
}

impl Fluxes {
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: Vector3) {
        if let Some(mut current) = self.momentum_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: Scalar) {
        if let Some(mut current) = self.energy_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

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
