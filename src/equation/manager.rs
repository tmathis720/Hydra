use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh, linalg::matrix::sparse_matrix::SparseMatrix, solver::KSP, time_stepping::{TimeDependentProblem, TimeStepper, TimeSteppingError}, Matrix
};
use super::{Fields, Fluxes, PhysicalEquation};
use std::sync::{Arc, RwLock};

/// The `EquationManager` struct manages a collection of equations
/// governing the physical processes on a mesh. It integrates the equations
/// using a specified time-stepping scheme and handles domain and boundary conditions.
pub struct EquationManager {
    /// A collection of equations that implement the `PhysicalEquation` trait.
    equations: Vec<Box<dyn PhysicalEquation>>,
    /// The time-stepping scheme to evolve the equations.
    time_stepper: Box<dyn TimeStepper<Self>>,
    /// The computational domain represented by a mesh.
    domain: Arc<RwLock<Mesh>>,
    /// Handler for boundary conditions associated with the domain.
    boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
}

impl EquationManager {
    /// Constructs a new `EquationManager` with the given time stepper, domain, and boundary handler.
    ///
    /// # Arguments
    /// - `time_stepper`: A boxed implementation of a `TimeStepper`.
    /// - `domain`: A shared, thread-safe reference to the computational mesh.
    /// - `boundary_handler`: A shared, thread-safe reference to the boundary condition handler.
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

    /// Adds a new equation to the manager.
    ///
    /// The equation must implement the `PhysicalEquation` trait.
    ///
    /// # Type Parameters
    /// - `E`: The type of the equation.
    ///
    /// # Arguments
    /// - `equation`: The equation to add.
    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    /// Assembles all equations in the manager, updating fluxes based on the current fields.
    ///
    /// # Arguments
    /// - `fields`: The current field data.
    /// - `fluxes`: A mutable reference to the flux data to be updated.
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

    /// Advances the solution by a single time step using the associated time stepper.
    ///
    /// # Arguments
    /// - `fields`: The field data to be updated.
    pub fn step(&mut self, fields: &mut Fields) -> Result<(), TimeSteppingError> {
        // Replace the current time stepper temporarily to ensure exclusive ownership during the step.
        let mut time_stepper = std::mem::replace(&mut self.time_stepper, 
            Box::new(NoOpStepper));
        let current_time = time_stepper.current_time();
        let time_step = time_stepper.get_time_step();

        // Perform the time-stepping operation.
        let result = time_stepper.step(self, time_step, current_time, fields);


        // Restore the time stepper after the step.
        self.time_stepper = time_stepper;

        result
    }
}

impl TimeDependentProblem for EquationManager {
    type State = Fields;
    type Time = f64;

    /// Computes the right-hand side (RHS) for the time-stepping scheme.
    ///
    /// # Arguments
    /// - `_time`: The current time (unused here but required for the trait).
    /// - `state`: The current state of the fields.
    /// - `derivative`: The output structure to hold the computed derivatives.
    ///
    /// # Returns
    /// - `Ok(())` on successful computation of the RHS.
    /// - `Err(TimeSteppingError)` if an error occurs.
    fn compute_rhs(
        &self,
        _time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        let mut fluxes = Fluxes::new();

        // Assemble all fluxes from the equations.
        self.assemble_all(state, &mut fluxes);

        // Update the derivative fields based on the computed fluxes.
        derivative.update_from_fluxes(&fluxes);

        Ok(())
    }

    /// Provides the initial state for the time-dependent problem.
    ///
    /// # Returns
    /// - A new `Fields` instance with default-initialized fields.
    fn initial_state(&self) -> Self::State {
        Fields::new()
    }

    /// Returns the system matrix, if applicable (not used in this implementation).
    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
        Some(Box::new(SparseMatrix::new(0, 0)))
    }

    /// Solves the linear system, if applicable (not used in this implementation).
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(TimeSteppingError)` if an error occurs.
    fn solve_linear_system(
        &self,
        _matrix: &mut dyn Matrix<Scalar = f64>,
        _state: &mut Self::State,
        _rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        Ok(())
    }
}

/// A temporary no-operation time stepper used to handle ownership of the time stepper during operations.
pub struct NoOpStepper;

impl<P> TimeStepper<P> for NoOpStepper
where
    P: TimeDependentProblem,
{
    fn get_solver(&mut self) -> &mut dyn KSP {
        todo!()
    }
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
        domain::{mesh::Mesh, section::scalar::Scalar},
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
