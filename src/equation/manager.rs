use crate::{boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh};
use crate::time_stepping::TimeStepper;
use super::{Fields, Fluxes, PhysicalEquation};

pub struct EquationManager<FieldType, TStepper> {
    equations: Vec<Box<dyn PhysicalEquation<FieldType>>>, // Single `Box`
    time_stepper: TStepper,
}

impl<FieldType, TStepper> EquationManager<FieldType, TStepper>
where
    TStepper: TimeStepper<Box<dyn PhysicalEquation<FieldType>>>, // Reflect the correct bound
{
    pub fn new(time_stepper: TStepper) -> Self {
        Self {
            equations: Vec::new(),
            time_stepper,
        }
    }

    pub fn add_equation<E: PhysicalEquation<FieldType> + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        domain: &Mesh,
        fields: &Fields<FieldType>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        let current_time = self.time_stepper.current_time();
        for equation in &self.equations {
            equation.assemble(domain, fields, fluxes, boundary_handler, current_time);
        }
    }

    pub fn step(&mut self, fields: &mut Vec<f64>) {
        let current_time = self.time_stepper.current_time();
        self.time_stepper
            .step(&self.equations, self.time_stepper.get_time_step(), current_time, fields)
            .expect("Time-stepping failed");
    }
}
