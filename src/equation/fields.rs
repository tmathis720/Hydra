// src/equation/fields.rs

use crate::domain::Section;

pub struct Fields {
    pub field: Section<f64>, // For primary variables like pressure
    pub gradient: Section<[f64; 3]>,
    pub velocity_field: Section<[f64; 3]>,

    // Additional fields for energy and turbulence
    pub temperature_field: Section<f64>,
    pub temperature_gradient: Section<[f64; 3]>,
    pub k_field: Section<f64>,         // Turbulent kinetic energy
    pub epsilon_field: Section<f64>,   // Turbulent dissipation rate
}

pub struct Fluxes {
    pub momentum_fluxes: Section<f64>,
    pub energy_fluxes: Section<f64>,
    pub turbulence_fluxes: Section<f64>,
}
