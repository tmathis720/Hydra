//! Time Stepping module
//!
//! This module defines traits and structures used for time stepping in solving
//! time-dependent problems, such as PDEs or ODEs. The `TimeStepper` trait defines
//! an interface for different time stepping methods, while the `TimeDependentProblem`
//! trait defines an interface for problems that can be solved using time stepping.

use crate::linalg::Matrix;
use crate::linalg::Vector;  // Import the Vector trait

/// Error type for time-stepping operations.
#[derive(Debug)]
pub struct TimeSteppingError;

/// Trait representing a time-dependent problem, such as a system of ODEs or PDEs.
///
/// Types implementing this trait must define how to compute the right-hand side (RHS) 
/// of the system, provide the initial state, and define how to interact with the matrix
/// and solver for implicit time-stepping schemes.
///
/// # Associated Types:
/// - `State`: The type representing the state of the system, which must implement `Vector`.
/// - `Time`: The type representing time (typically `f64` for real-valued time).
pub trait TimeDependentProblem {
    // Ensure that `State` implements `Vector`.
    type State: Vector;  // `State` must implement the `Vector` trait.
    type Time;

    /// Computes the right-hand side (RHS) of the system at a given time.
    ///
    /// # Parameters
    /// - `time`: The current time in the simulation.
    /// - `state`: The current state of the system.
    /// - `derivative`: The output where the derivative (RHS) is stored.
    ///
    /// # Returns
    /// - `Result<(), TimeSteppingError>`: Ok if the computation was successful, otherwise an error.
    fn compute_rhs(&self, time: Self::Time, state: &Self::State, derivative: &mut Self::State) -> Result<(), TimeSteppingError>;

    /// Returns the initial state of the system at the beginning of the simulation.
    ///
    /// # Returns
    /// - `State`: The initial state of the system.
    fn initial_state(&self) -> Self::State;

    /// Converts time to the scalar type of the vector.
    ///
    /// This allows for using time values in vector operations like axpy.
    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar;

    /// Returns a matrix representation of the system if applicable.
    ///
    /// This function is used in implicit time-stepping methods to retrieve the matrix
    /// for the linear system that needs to be solved at each time step.
    ///
    /// # Returns
    /// - `Option<dyn Matrix<Scalar = f64>>`: Some matrix if available, or None if not applicable.
    fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>>;

    /// Solves the linear system `A * x = b` for implicit time-stepping methods.
    ///
    /// # Parameters
    /// - `matrix`: The matrix `A` in the system.
    /// - `state`: The state vector `x` that is being solved for.
    /// - `rhs`: The right-hand side vector `b` in the system.
    ///
    /// # Returns
    /// - `Result<(), TimeSteppingError>`: Ok if the system was solved successfully, otherwise an error.
    fn solve_linear_system(&self, matrix: &mut dyn Matrix<Scalar = f64>, state: &mut Self::State, rhs: &Self::State) -> Result<(), TimeSteppingError>;
}

/// Trait for time-stepping methods.
///
/// This trait defines the interface for time-stepping methods, such as explicit methods
/// like Forward Euler, or implicit methods like Backward Euler. The primary function is
/// `step()`, which advances the solution by one time step.
///
/// # Associated Types:
/// - `P`: The type representing the time-dependent problem to be solved.
pub trait TimeStepper<P: TimeDependentProblem> {
    /// Performs a single time step.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem being solved.
    /// - `time`: The current time in the simulation.
    /// - `dt`: The time step size.
    /// - `state`: The current state of the system, which will be updated in place.
    ///
    /// # Returns
    /// - `Result<(), TimeSteppingError>`: Ok if the time step was successful, otherwise an error.
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    /// Performs an adaptive time step, if applicable.
    ///
    /// Some time-stepping methods support adaptive time-stepping, where the time step size
    /// is dynamically adjusted based on error estimates or other criteria.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem being solved.
    /// - `time`: The current time in the simulation.
    /// - `state`: The current state of the system, which will be updated in place.
    ///
    /// # Returns
    /// - `Result<(), TimeSteppingError>`: Ok if the time step was successful, otherwise an error.
    fn adaptive_step(
        &mut self,
        problem: &P,
        time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    /// Sets the time interval for the simulation.
    ///
    /// This function is used to specify the start and end times of the simulation.
    ///
    /// # Parameters
    /// - `start_time`: The start time of the simulation.
    /// - `end_time`: The end time of the simulation.
    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    /// Sets the time step size for the simulation.
    ///
    /// This function is used to specify the size of the time step in explicit or implicit
    /// time-stepping methods.
    ///
    /// # Parameters
    /// - `dt`: The time step size.
    fn set_time_step(&mut self, dt: P::Time);
}
