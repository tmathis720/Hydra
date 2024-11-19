use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    domain::mesh::Mesh,
    time_stepping::{TimeDependentProblem, TimeStepper, TimeSteppingError},
    Matrix,
};
use super::{Fields, Fluxes, PhysicalEquation};
use std::sync::{Arc, RwLock};

pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
    time_stepper: Box<dyn TimeStepper<Self>>,
    domain: Arc<RwLock<Mesh>>,
    boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
}

impl EquationManager {
    pub fn new(
        time_stepper: Box<dyn TimeStepper<Self>>,
        domain: Arc<RwLock<Mesh>>,
        boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
    ) -> Self {
        Self {
            equations: Vec::new(),
            time_stepper,
            domain,
            boundary_handler,
        }
    }

    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        fields: &Fields,
        fluxes: &mut Fluxes,
    ) {
        let current_time = self.time_stepper.current_time();
        let domain = self.domain.read().unwrap();
        let boundary_handler = self.boundary_handler.read().unwrap();
        for equation in &self.equations {
            equation.assemble(&domain, fields, fluxes, &boundary_handler, current_time);
        }
    }

    pub fn step(&mut self, fields: &mut Fields) {
        let mut time_stepper = std::mem::replace(&mut self.time_stepper, Box::new(NoOpStepper));
        let current_time = time_stepper.current_time();
        let time_step = time_stepper.get_time_step();

        time_stepper
            .step(self, time_step, current_time, fields)
            .expect("Time-stepping failed");

        self.time_stepper = time_stepper;
    }
}

impl TimeDependentProblem for EquationManager {
    type State = Fields;
    type Time = f64;

    fn compute_rhs(
        &self,
        _time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        let mut fluxes = Fluxes::new();

        self.assemble_all(state, &mut fluxes);
        derivative.update_from_fluxes(&fluxes);

        Ok(())
    }

    fn initial_state(&self) -> Self::State {
        Fields::new()
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
        None
    }

    fn solve_linear_system(
        &self,
        _matrix: &mut dyn Matrix<Scalar = f64>,
        _state: &mut Self::State,
        _rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        Ok(())
    }
}

// Temporary no-op stepper for handling ownership.
struct NoOpStepper;

impl<P> TimeStepper<P> for NoOpStepper
where
    P: TimeDependentProblem,
{
    fn current_time(&self) -> P::Time {
        P::Time::from(0.0)
    }

    fn set_current_time(&mut self, _time: P::Time) {}

    fn step(
        &mut self,
        _problem: &P,
        _dt: P::Time,
        _current_time: P::Time,
        _state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
        _tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        Err(TimeSteppingError::InvalidStep)
    }

    fn set_time_interval(&mut self, _start_time: P::Time, _end_time: P::Time) {}

    fn set_time_step(&mut self, _dt: P::Time) {}

    fn get_time_step(&self) -> P::Time {
        P::Time::from(0.0)
    }
}
