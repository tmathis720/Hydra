use crate::linalg::Matrix;
use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct BackwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
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
        problem: &P,
        time: P::Time,
        state: &mut P::State,
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
    use faer::{Mat, MatRef}; // Import matrix types from faer
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
            matrix: &mut dyn Matrix<Scalar = f64>,  // Use concrete Mat type
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

