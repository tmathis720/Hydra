use crate::{domain::Section, MeshEntity};

pub struct Fields<FieldType> {
    pub velocity_field: Section<[f64; 3]>,
    pub pressure_field: Section<f64>,
    pub velocity_gradient: Section<[[f64; 3]; 3]>,
    pub temperature_field: Section<f64>,
    pub temperature_gradient: Section<[f64; 3]>,
    pub k_field: Section<f64>,
    pub epsilon_field: Section<f64>,
    pub gradient: Section<FieldType>,
    pub field: Section<FieldType>, // Fixed placeholder `_`
}

impl<T> Fields<T> {
    pub fn new() -> Self {
        Self {
            velocity_field: Section::new(),
            pressure_field: Section::new(),
            velocity_gradient: Section::new(),
            temperature_field: Section::new(),
            temperature_gradient: Section::new(),
            k_field: Section::new(),
            epsilon_field: Section::new(),
            gradient: Section::new(),
            field: Section::new(), // Initialized correctly
        }
    }


    pub fn get_velocity(&self, entity: &MeshEntity) -> Option<[f64; 3]> {
        self.velocity_field.restrict(entity)
    }

    pub fn get_pressure(&self, entity: &MeshEntity) -> Option<f64> {
        self.pressure_field.restrict(entity)
    }

    pub fn get_velocity_gradient(&self, entity: &MeshEntity) -> Option<[[f64; 3]; 3]> {
        self.velocity_gradient.restrict(entity)
    }

    pub fn set_velocity(&mut self, entity: MeshEntity, value: [f64; 3]) {
        self.velocity_field.set_data(entity, value);
    }

    pub fn set_pressure(&mut self, entity: MeshEntity, value: f64) {
        self.pressure_field.set_data(entity, value);
    }

    pub fn set_velocity_gradient(&mut self, entity: MeshEntity, value: [[f64; 3]; 3]) {
        self.velocity_gradient.set_data(entity, value);
    }
}

pub struct Fluxes {
    pub momentum_fluxes: Section<[f64; 3]>,
    pub energy_fluxes: Section<f64>,
    pub turbulence_fluxes: Section<[f64; 2]>,
}

impl Fluxes {
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: [f64; 3]) {
        if let Some(mut current) = self.momentum_fluxes.restrict(&entity) {
            for i in 0..3 {
                current[i] += value[i];
            }
            self.momentum_fluxes.set_data(entity, current);
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: f64) {
        if let Some(mut current) = self.energy_fluxes.restrict(&entity) {
            current += value;
            self.energy_fluxes.set_data(entity, current);
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

    pub fn add_turbulence_flux(&mut self, entity: MeshEntity, value: [f64; 2]) {
        if let Some(mut current) = self.turbulence_fluxes.restrict(&entity) {
            for i in 0..2 {
                current[i] += value[i];
            }
            self.turbulence_fluxes.set_data(entity, current);
        } else {
            self.turbulence_fluxes.set_data(entity, value);
        }
    }
}
