use std::ops::{Add, Mul};
use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};
use super::super::domain::section::{Vector3, Tensor3x3, Scalar, Vector2};

pub trait UpdateState {
    fn update_state(&mut self, derivative: &Self, dt: f64);
}

#[derive(Clone)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<Scalar>>,
    pub vector_fields: FxHashMap<String, Section<Vector3>>,
    pub tensor_fields: FxHashMap<String, Section<Tensor3x3>>,
}

pub trait FieldIterator {
    type Item: Add<Output = Self::Item> + Mul<f64, Output = Self::Item> + Clone;

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Self::Item> + 'a>;
    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Self::Item> + 'a>;
}

impl FieldIterator for Fields {
    type Item = f64;

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Self::Item> + 'a> {
        Box::new(
            self.scalar_fields
                .values()
                .flat_map(|section| section.all_data().iter().map(|scalar| &scalar.0)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Self::Item> + 'a> {
        Box::new(
            self.scalar_fields
                .values_mut()
                .flat_map(|section| section.all_data_mut().iter_mut().map(|scalar| &mut scalar.0)),
        )
    }
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

    pub fn update_from_fluxes(&mut self, _fluxes: &Fluxes) {
        // Implement logic to update derivative fields from fluxes
    }
}

impl UpdateState for Fields {
    fn update_state(&mut self, derivative: &Fields, dt: f64) {
        for (key, section) in &derivative.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            }
        }
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
