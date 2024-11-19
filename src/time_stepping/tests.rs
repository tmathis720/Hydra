#[cfg(test)]
mod tests {
    use crate::equation::fields::UpdateState;
    use crate::time_stepping::adaptivity::error_estimate::estimate_error;
    use crate::time_stepping::adaptivity::step_size_control::adjust_step_size;
    use crate::time_stepping::methods::euler::ExplicitEuler;
    use crate::time_stepping::{TimeDependentProblem, TimeStepper, TimeSteppingError};
    use crate::Matrix;

    #[derive(Clone)]
    struct MockState {
        value: f64,
    }

    impl UpdateState for MockState {
        fn update_state(&mut self, derivative: &Self, dt: f64) {
            self.value += derivative.value * dt;
        }
        
        fn difference(&self, _other: &Self) -> Self {
            todo!()
        }
        
        fn norm(&self) -> f64 {
            todo!()
        }
    }

    struct MockProblem;

    impl TimeDependentProblem for MockProblem {
        type State = MockState;
        type Time = f64;

        fn compute_rhs(
            &self,
            _time: Self::Time,
            state: &Self::State,
            derivative: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            derivative.value = -state.value; // Simple exponential decay
            Ok(())
        }

        fn initial_state(&self) -> Self::State {
            MockState { value: 1.0 }
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
            Err(TimeSteppingError::SolverError("No solver in mock".into()))
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
        assert!(solver.get_time_step() < 0.1, "Step size should decrease if error is high.");
    }

    #[test]
    fn test_error_estimation() {
        let problem = MockProblem;
        let state = problem.initial_state();
        let error = estimate_error(&problem, &state, 0.1).unwrap();
        assert!(error > 0.0, "Error should be positive.");
    }

    #[test]
    fn test_step_size_adjustment() {
        let new_dt = adjust_step_size(0.1, 0.01, 1e-3, 0.9, 2.0);
        assert!(new_dt < 0.1, "Step size should decrease when error is above tolerance.");
    }
}
