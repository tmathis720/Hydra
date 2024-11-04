use fields::{Fields, Fluxes};

use crate::{boundary::bc_handler::BoundaryConditionHandler, Mesh, Section};

pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;

pub mod fields;
pub mod manager;
pub mod energy_equation;
pub mod turbulence_models;
pub mod momentum_equation;

// src/equation/mod.rs

pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    );
}
