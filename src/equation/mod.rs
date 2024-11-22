use fields::{Fields, Fluxes};

use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    Mesh,
};

pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;

pub mod fields;
pub mod manager;
pub mod energy_equation;
pub mod momentum_equation;

pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

#[cfg(test)]
pub mod tests;
