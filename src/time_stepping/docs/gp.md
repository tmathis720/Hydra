Generate a detailed users guide for the `Time Stepping` module for Hydra. I am going to provide the code for all of the parts of the `Time Stepping` module below, and you can analyze and build the detailed outline based on this version of the source code.

Much of the module is yet to be implemented, so here is the source tree to help fill in some of the missing details. However, please be sure to point out what isn't implemented yet.

```bash
C:.
│   mod.rs
│   ts.rs
│
├───adaptivity
│       error_estimate.rs
│       mod.rs
│       step_size_control.rs
│
└───methods
        backward_euler.rs
        crank_nicolson.rs
        euler.rs
        mod.rs
        runge_kutta.rs

```

---

`src/time_stepping/mod.rs`

```rust
pub mod ts;
pub mod methods;
pub mod adaptivity;

pub use ts::{TimeStepper, TimeSteppingError, TimeDependentProblem};
pub use methods::backward_euler::BackwardEuler;
pub use methods::euler::ForwardEuler;
```

---

`src/time_stepping/ts.rs`

```rust
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
    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,  // Change this to use the trait
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
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
```

---

`src/time_stepping/methods/mod.rs`

```rust
pub mod euler;
pub mod backward_euler;
```

---

`src/time_stepping/methods/backward_euler.rs`

```rust

use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct BackwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        _dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        // Use Box to store the dynamically sized Matrix on the heap
        let mut matrix = problem.get_matrix();  // No need for Box::new here
        let mut rhs = problem.initial_state();

        problem.compute_rhs(time, state, &mut rhs)?;
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _time: P::Time,
        _state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        // Adaptive step logic implementation
        Ok(())
    }

    fn set_time_interval(&mut self, _start_time: P::Time, _end_time: P::Time) {
        // Implement setting time interval if needed
    }

    fn set_time_step(&mut self, _dt: P::Time) {
        // Implement setting time step if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat; // Import matrix types from faer
    use crate::linalg::Matrix;
    use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

    // Mock struct to represent a simple linear system for testing
    struct MockProblem {
        matrix: Mat<f64>,  // Use Mat from faer
        initial_state: Vec<f64>,
    }

    // Implement the TimeDependentProblem trait for MockProblem
    impl TimeDependentProblem for MockProblem {
        type State = Vec<f64>;
        type Time = f64;
    
        // Returns the initial state of the problem
        fn initial_state(&self) -> Self::State {
            self.initial_state.clone()
        }
    
        // Computes the right-hand side of the equation (RHS)
        fn compute_rhs(
            &self,
            _time: Self::Time,
            _state: &Self::State,
            rhs: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            rhs[0] = 1.0;
            rhs[1] = 1.0;
            Ok(())
        }
    
        // Correct return type for get_matrix
        fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>> {
            Box::new(self.matrix.clone())  // Clone matrix into a boxed trait object
        }
    
        // Correct type for the matrix parameter
        fn solve_linear_system(
            &self,
            _matrix: &mut dyn Matrix<Scalar = f64>,  // Use concrete Mat type
            state: &mut Self::State,
            rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            state[0] = rhs[0];
            state[1] = rhs[1];
            Ok(())
        }
    
        fn time_to_scalar(&self, _time: Self::Time) -> <Self::State as crate::Vector>::Scalar {
            todo!()
        }
    }

    #[test]
    fn test_backward_euler_step() {
        // Create a simple 2x2 matrix for testing using faer::Mat
        let test_matrix = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });

        let initial_state = vec![0.0, 0.0];

        // Create the mock problem
        let problem = MockProblem {
            matrix: test_matrix,
            initial_state,
        };

        // Initialize BackwardEuler stepper
        let mut stepper = BackwardEuler;

        // Define time and timestep
        let time = 0.0;
        let dt = 0.1;
        let mut state = problem.initial_state();

        // Perform the step using BackwardEuler
        let result = stepper.step(&problem, time, dt, &mut state);

        // Ensure the step was successful
        assert!(result.is_ok());

        // Ensure that the state has been updated correctly
        assert_eq!(state, vec![1.0, 1.0]);
    }
}

```

---

`src/time_stepping/methods/euler.rs`

```rust

use crate::linalg::Vector;
use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct ForwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for ForwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut rhs = problem.initial_state();

        problem.compute_rhs(time, state, &mut rhs)?;

        // Convert dt (time) to the scalar type used by the vector.
        let scalar_dt = problem.time_to_scalar(dt);

        // Now call axpy with the scalar converted from dt.
        state.axpy(scalar_dt, &rhs);

        Ok(())
    }

    fn set_time_interval(&mut self, _start_time: P::Time, _end_time: P::Time) {
        // Implement setting time interval if needed
    }

    fn set_time_step(&mut self, _dt: P::Time) {
        // Implement setting time step if needed
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _time: P::Time,
        _state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        // Adaptive step logic implementation
        Ok(())
    }
}

```

---

