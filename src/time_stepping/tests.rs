#[cfg(test)]
mod tests {
    use crate::equation::fields::UpdateState;
    use crate::time_stepping::adaptivity::error_estimate::estimate_error;
    use crate::time_stepping::adaptivity::step_size_control::adjust_step_size;
    use crate::time_stepping::methods::euler::ExplicitEuler;
    use crate::time_stepping::{TimeDependentProblem, TimeStepper, TimeSteppingError};
    use crate::Matrix;

    /// Represents a simple state with a scalar value
    #[derive(Clone, Debug)]
    struct MockState {
        value: f64,
    }

    /// Trait implementation for updating and calculating differences in state
    impl UpdateState for MockState {
        fn compute_residual(&self, rhs: &Self) -> f64 {
            (self.value - rhs.value).abs()
        }
        fn update_state(&mut self, derivative: &Self, dt: f64) {
            self.value += derivative.value * dt;
        }

        fn difference(&self, other: &Self) -> Self {
            MockState {
                value: self.value - other.value,
            }
        }

        fn norm(&self) -> f64 {
            self.value.abs() // L1 norm for simplicity
        }
    }

    /// Represents a simple problem with exponential decay dynamics
    struct MockProblem;

    impl TimeDependentProblem for MockProblem {
        type State = MockState;
        type Time = f64;

        /// Computes the derivative (RHS) of the state
        fn compute_rhs(
            &self,
            _time: Self::Time,
            state: &Self::State,
            derivative: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            derivative.value = -state.value; // Exponential decay
            Ok(())
        }

        /// Provides the initial state
        fn initial_state(&self) -> Self::State {
            MockState { value: 1.0 } // Initial value for the problem
        }

        /// Mock implementation returns no matrix
        fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
            None
        }

        /// Mock implementation of solving a linear system (not used)
        fn solve_linear_system(
            &self,
            _matrix: &mut dyn Matrix<Scalar = f64>,
            _state: &mut Self::State,
            _rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            Err(TimeSteppingError::SolverError(
                "No solver implemented for MockProblem".into(),
            ))
        }
    }

    #[test]
    fn test_adaptive_step() {
        let mut solver = ExplicitEuler::new(0.1, 0.0, 1.0);
        let problem = MockProblem;
        let mut state = problem.initial_state();

        let tol = 1e-3;
        let result = solver.adaptive_step(&problem, &mut state, tol);
        assert!(result.is_ok());
        assert!(
            solver.get_time_step() < 0.1,
            "Step size should decrease if error is high."
        );
    }

    #[test]
    fn test_error_estimation() {
        let problem = MockProblem;
        let state = problem.initial_state();
        let error = estimate_error(&problem, &state, 0.1).unwrap();
        assert!(error > 0.0, "Error should be positive.");
    }

    #[test]
    fn test_step_size_control() {
        let current_dt = 0.1;
        let error = 0.01;
        let tol = 1e-3;
        let safety_factor = 0.9;
        let growth_factor = 2.0;
        let new_dt = adjust_step_size(current_dt, error, tol, safety_factor, growth_factor);
        assert!(new_dt < current_dt, "Step size should decrease when error is high.");
    }

    #[test]
    fn test_step_size_adjustment() {
        let new_dt = adjust_step_size(0.1, 0.01, 1e-3, 0.9, 2.0);
        assert!(new_dt < 0.1, "Step size should decrease when error is above tolerance.");
    }
}
