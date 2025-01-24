use crate::{
    equation::fields::UpdateState,
    linalg::Matrix,
    solver::{ksp::SolverManager, KSP},
};
use std::ops::Add;

/// Represents errors that can occur during time-stepping.
#[derive(Debug)]
pub enum TimeSteppingError {
    /// Indicates an invalid time step.
    InvalidStep,
    /// Represents an error related to the solver, with a specific message.
    SolverError(String),
}

/// Trait for problems that evolve over time and can be solved using a time-stepper.
pub trait TimeDependentProblem {
    /// Represents the state of the system (e.g., fields, variables).
    type State: Clone + UpdateState;
    /// Represents the time variable with arithmetic and conversion capabilities.
    type Time: Copy + PartialOrd + Add<Output = Self::Time> + From<f64> + Into<f64>;

    /// Computes the right-hand side (RHS) of the governing equations.
    ///
    /// # Parameters
    /// - `time`: The current simulation time.
    /// - `state`: The current state of the system.
    /// - `derivative`: The computed RHS to be updated.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` on failure.
    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    /// Provides the initial state of the problem.
    ///
    /// # Returns
    /// - The initial state of the system.
    fn initial_state(&self) -> Self::State;

    /// Provides the matrix representation of the system, if available.
    ///
    /// # Returns
    /// - An optional boxed matrix.
    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    /// Provides the configuration for sub-iterations, if applicable.
    ///
    /// # Returns
    /// - `None` by default; override to specify a number of sub-iterations.
    fn get_sub_iteration_config(&self) -> Option<usize> {
        None
    }

    /// Solves the linear system for the current time step.
    ///
    /// # Parameters
    /// - `matrix`: The system matrix.
    /// - `state`: The state to be updated.
    /// - `rhs`: The right-hand side vector.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` on failure.
    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}

/// Trait for implementing time-stepping algorithms.
///
/// # Type Parameters
/// - `P`: The problem type, which must implement `TimeDependentProblem`.
pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    /// Gets the current time of the time-stepper.
    fn current_time(&self) -> P::Time;

    /// Sets the current time of the time-stepper.
    fn set_current_time(&mut self, time: P::Time);

    /// Advances the solution by one time step.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `dt`: The time step size.
    /// - `current_time`: The current simulation time.
    /// - `state`: The current state of the system.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` on failure.
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    /// Attempts an adaptive time step based on error tolerance.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `state`: The current state of the system.
    /// - `tol`: The error tolerance for adaptivity.
    ///
    /// # Returns
    /// - The size of the next time step on success, or a `TimeSteppingError`.
    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError>;

    /// Sets the time interval for the simulation.
    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    /// Sets the size of the time step.
    fn set_time_step(&mut self, dt: P::Time);

    /// Gets the current size of the time step.
    fn get_time_step(&self) -> P::Time;

    /// Provides access to the underlying solver.
    fn get_solver(&mut self) -> &mut dyn KSP;
}

/// Fixed time-stepper implementation.
///
/// This implementation uses a constant time step for the entire simulation.
///
/// # Type Parameters
/// - `P`: The problem type, which must implement `TimeDependentProblem`.
pub struct FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    /// The current simulation time.
    current_time: P::Time,
    /// The fixed time step size.
    time_step: P::Time,
    /// The start time of the simulation.
    start_time: P::Time,
    /// The end time of the simulation.
    end_time: P::Time,
    /// Manager for the solver used in time-stepping.
    solver_manager: SolverManager,
}

impl<P> FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    /// Constructs a new fixed time-stepper.
    ///
    /// # Parameters
    /// - `start_time`: The start time of the simulation.
    /// - `end_time`: The end time of the simulation.
    /// - `time_step`: The fixed size of the time step.
    /// - `solver`: The solver to use for solving linear systems.
    ///
    /// # Returns
    /// - A new instance of `FixedTimeStepper`.
    pub fn new(
        start_time: P::Time,
        end_time: P::Time,
        time_step: P::Time,
        solver: Box<dyn KSP>,
    ) -> Self {
        FixedTimeStepper {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
            solver_manager: SolverManager::new(solver),
        }
    }

    /// Performs sub-iterations for specialized time-stepping methods (e.g., PISO).
    ///
    /// # Parameters
    /// - `problem`: The governing equations to solve.
    /// - `state`: The current state of the system.
    /// - `max_iterations`: Maximum number of sub-iterations allowed.
    /// - `tolerance`: Convergence tolerance for sub-iterations.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` if sub-iterations fail to converge.
    pub fn sub_iterate(
        &mut self,
        problem: &P,
        state: &mut P::State,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<(), TimeSteppingError> {
        for iteration in 0..max_iterations {
            let mut rhs = state.clone();
            problem.compute_rhs(self.current_time, state, &mut rhs)?;

            let mut matrix = problem
                .get_matrix()
                .ok_or_else(|| TimeSteppingError::SolverError("Matrix not provided by problem.".into()))?;

            problem.solve_linear_system(&mut *matrix, state, &rhs)?;

            let residual = state.compute_residual(&rhs);
            if residual < tolerance {
                break;
            }

            if iteration == max_iterations - 1 {
                return Err(TimeSteppingError::SolverError(
                    "Sub-iterations did not converge.".into(),
                ));
            }
        }
        Ok(())
    }

    /// Logs progress for debugging or monitoring purposes.
    ///
    /// # Parameters
    /// - `iteration`: The current iteration count.
    /// - `residual`: The residual value at the current iteration.
    fn _log_progress(&self, iteration: usize, residual: f64) {
        println!(
            "[TimeStepper] Iteration: {}, Time: {:.3}, Residual: {:.6}",
            iteration, self.current_time.into(), residual
        );
    }
}

impl<P> TimeStepper<P> for FixedTimeStepper<P>
where
    P: TimeDependentProblem,
{
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut derivative = state.clone();
        problem.compute_rhs(current_time, state, &mut derivative)?;

        state.update_state(&derivative, dt.into());

        // Perform sub-iterations if required by the problem
        if let Some(max_iterations) = problem.get_sub_iteration_config() {
            self.sub_iterate(problem, state, max_iterations, 1e-6)?; // Example tolerance
        }

        self.current_time = self.current_time + dt;

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

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time) {
        self.start_time = start_time;
        self.end_time = end_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    fn get_time_step(&self) -> P::Time {
        self.time_step
    }

    fn get_solver(&mut self) -> &mut dyn KSP {
        &mut *self.solver_manager.solver
    }
}
