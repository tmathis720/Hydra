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
pub struct NoOpStepper;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::BoundaryConditionHandler,
        domain::{mesh::Mesh, section::Scalar},
        MeshEntity,
    };
    use std::sync::{Arc, RwLock};

    struct MockEquation;
    impl PhysicalEquation for MockEquation {
        fn assemble(
            &self,
            _domain: &Mesh,
            _fields: &Fields,
            _fluxes: &mut Fluxes,
            _boundary_handler: &BoundaryConditionHandler,
            _current_time: f64,
        ) {
            // Mock behavior: adds a dummy flux
            _fluxes.add_energy_flux(MeshEntity::Face(0), Scalar(1.0));
        }
    }

    fn setup_environment() -> (Arc<RwLock<Mesh>>, Arc<RwLock<BoundaryConditionHandler>>) {
        let domain = Arc::new(RwLock::new(Mesh::new()));
        let boundary_handler = Arc::new(RwLock::new(BoundaryConditionHandler::new()));
        (domain, boundary_handler)
    }

    #[test]
    fn test_add_equation() {
        let (domain, boundary_handler) = setup_environment();
        let mut manager = EquationManager::new(Box::new(NoOpStepper), domain, boundary_handler);

        manager.add_equation(MockEquation {});
        assert_eq!(manager.equations.len(), 1);
    }

    #[test]
    fn test_assemble_all() {
        let (domain, boundary_handler) = setup_environment();
        let mut manager = EquationManager::new(Box::new(NoOpStepper), domain, boundary_handler);

        manager.add_equation(MockEquation {});
        let fields = Fields::new();
        let mut fluxes = Fluxes::new();

        manager.assemble_all(&fields, &mut fluxes);

        // Verify flux was added
        assert_eq!(fluxes.energy_fluxes.data.len(), 1);
    }

    #[test]
    fn test_step() {
        let (domain, boundary_handler) = setup_environment();
        let mut manager = EquationManager::new(Box::new(NoOpStepper), domain, boundary_handler);

        let mut fields = Fields::new();
        manager.step(&mut fields);

        // Verify step completes without error
    }

    #[test]
    fn test_compute_rhs() {
        let (domain, boundary_handler) = setup_environment();
        let manager = EquationManager::new(Box::new(NoOpStepper), domain, boundary_handler);

        let fields = Fields::new();
        let mut derivative = Fields::new();
        let result = manager.compute_rhs(0.0, &fields, &mut derivative);

        assert!(result.is_ok());
    }

    #[test]
    fn test_initial_state() {
        let (domain, boundary_handler) = setup_environment();
        let manager = EquationManager::new(Box::new(NoOpStepper), domain, boundary_handler);

        let state = manager.initial_state();
        assert_eq!(state.scalar_fields.len(), 0);
    }
}
