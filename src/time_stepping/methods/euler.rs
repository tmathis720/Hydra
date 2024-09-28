use crate::time_stepping::ts::{TimeStepper, TimeDependentProblem, ProblemError};
use crate::solver::Matrix;
/// The Forward Euler method for time-stepping.
pub struct ForwardEuler {
    dt: f64, // Time step size
}

impl ForwardEuler {
    /// Creates a new ForwardEuler time stepper with the given time step size.
    pub fn new(dt: f64) -> Self {
        ForwardEuler { dt }
    }

    /// Updates the time step size.
    pub fn set_time_step(&mut self, dt: f64) {
        self.dt = dt;
    }
}

impl<P> TimeStepper<P> for ForwardEuler
where
    P: TimeDependentProblem<State = Vec<f64>, Time = f64>, // Assuming the state is a vector of f64
{
    /// Performs one time step using the Forward Euler method.
    ///
    /// The state is updated as: `u_new = u_old + dt * f(t, u)`
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        state: &mut P::State,
    ) -> Result<(), ProblemError> {
        let mut rhs = state.clone(); // Create a copy of the state to store the result
        
        // Compute the right-hand side (RHS) of the ODE: f(t, u)
        problem.compute_rhs(time, state, &mut rhs)?;

        // Update state: u_new = u_old + dt * f(t, u)
        for (i, val) in state.iter_mut().enumerate() {
            *val += self.dt * rhs[i];
        }

        Ok(())
    }

    fn set_time_interval(&mut self, _start_time: P::Time, _end_time: P::Time) {
        // For Forward Euler, time interval management is handled externally.
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.dt = dt;
    }
}

/// Unit tests for ForwardEuler
#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_stepping::ts::{TimeDependentProblem, ProblemError, TimeStepper};

    struct MockProblem;

    impl TimeDependentProblem for MockProblem {
        type State = Vec<f64>;
        type Time = f64;

        fn compute_rhs(
            &self,
            _time: Self::Time,
            state: &Self::State,
            derivative: &mut Self::State,
        ) -> Result<(), ProblemError> {
            // Simple linear ODE: du/dt = -u
            for (i, val) in state.iter().enumerate() {
                derivative[i] = -val;
            }
            Ok(())
        }

        fn initial_condition(&self, _position: &[f64]) -> Self::State {
            vec![1.0] // Initial condition: u(0) = 1
        }

        fn boundary_condition(
            &self,
            _time: Self::Time,
            _position: &[f64],
        ) -> Option<Self::State> {
            None // No boundary conditions in this mock problem
        }

        fn source_term(
            &self,
            _time: Self::Time,
            _position: &[f64],
        ) -> Self::State {
            vec![0.0] // No source term in this mock problem
        }

        fn coefficient(&self, _position: &[f64]) -> f64 {
            1.0 // Constant coefficient
        }

        fn compute_jacobian(&self, _time: Self::Time, _state: &Self::State) -> Option<dyn Matrix<Scalar = f64>> {
            // Default to no Jacobian for explicit methods
            unimplemented!()
        }

        fn mass_matrix(&self, _time: Self::Time, _matrix: &mut dyn Matrix<Scalar = f64>) -> Result<(), ProblemError> {
            // Default to the identity matrix for ODEs
            unimplemented!()
        }

        fn set_initial_conditions(&self, _state: &mut Self::State) {
            unimplemented!()
        }

        fn apply_boundary_conditions(&self, _time: Self::Time, _state: &mut Self::State) {
            unimplemented!()
        }
    }

    #[test]
    fn test_forward_euler() {
        let problem = MockProblem;
        let mut state = problem.initial_condition(&[]);
        let time = 0.0;
        let dt = 0.1; // Time step size: dt = 0.1
        let mut euler = ForwardEuler { dt }; // Fixed initialization

        // Perform one step: u_new = u_old + dt * f(t, u)
        euler.step(&problem, time, dt, &mut state).unwrap();

        // Expected result: u(0.1) = 1.0 - 0.1 * 1.0 = 0.9
        assert!((state[0] - 0.9).abs() < 1e-6);
    }
}

