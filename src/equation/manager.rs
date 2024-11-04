// src/equation/manager.rs

use super::{PhysicalEquation, Fields, Fluxes};
use crate::domain::mesh::Mesh;
use crate::boundary::bc_handler::BoundaryConditionHandler;

pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
}

impl EquationManager {
    pub fn new() -> Self {
        EquationManager {
            equations: Vec::new(),
        }
    }

    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        for equation in &self.equations {
            equation.assemble(domain, fields, fluxes, boundary_handler);
        }
    }
}
