use crate::linalg::Vector;
use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct ForwardEuler {
    current_time: f64,
    time_step: f64,
}

impl ForwardEuler {
    pub fn new(start_time: f64, time_step: f64) -> Self {
        Self {
            current_time: start_time,
            time_step,
        }
    }
}

impl<P: TimeDependentProblem<Time = f64>> TimeStepper<P> for ForwardEuler {
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problems: &[P],
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        self.time_step = dt;

        // Iterate over problems and apply Forward Euler method
        for problem in problems {
            let mut rhs = state.clone();
            problem.compute_rhs(current_time, state, &mut rhs)?;

            let scalar_dt = problem.time_to_scalar(dt);
            state.axpy(scalar_dt, &rhs);
        }

        // Update the current time
        self.current_time += dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError> {
        // Adaptive step logic (placeholder)
        Ok(self.time_step)
    }

    fn set_time_interval(&mut self, start_time: P::Time, _end_time: P::Time) {
        self.current_time = start_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }
    
    fn get_time_step(&self) -> <P as TimeDependentProblem>::Time {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{matrix, Vector};
    use crate::time_stepping::{TimeDependentProblem, TimeSteppingError};
    use faer::Mat;

    struct MockProblem {
        initial_state: Vec<f64>,
    }

    impl TimeDependentProblem for MockProblem {
        type State = Vec<f64>;
        type Time = f64;

        fn initial_state(&self) -> Self::State {
            self.initial_state.clone()
        }

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

        fn get_matrix(&self) -> Option<std::boxed::Box<(dyn matrix::traits::Matrix<Scalar = f64> + 'static)>> { // Corrected to use `faer::Mat`
            None
        }

        fn solve_linear_system(
            &self,
            _matrix: &mut dyn matrix::traits::Matrix<Scalar = f64>, // Corrected to use `faer::Mat`
            _state: &mut Self::State,
            _rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            Ok(())
        }

        fn time_to_scalar(&self, time: Self::Time) -> f64 {
            time
        }
    }

    #[test]
    fn test_forward_euler_step() {
        let problem = MockProblem {
            initial_state: vec![0.0, 0.0],
        };

        let mut stepper = ForwardEuler::new(0.0, 0.1);
        let mut state = problem.initial_state();

        let problems = vec![problem];

        let result = stepper.step(&problems, 0.1, 0.0, &mut state);

        assert!(result.is_ok());
        assert_eq!(state, vec![0.1, 0.1]);
        assert_eq!(stepper.current_time(), 0.1);
    }

    #[test]
    fn test_set_current_time() {
        let mut stepper = ForwardEuler::new(0.0, 0.1);
        stepper.set_current_time(1.0);
        assert_eq!(stepper.current_time(), 1.0);
    }

    #[test]
    fn test_set_time_step() {
        let mut stepper = ForwardEuler::new(0.0, 0.1);
        stepper.set_time_step(0.2);
        assert_eq!(stepper.time_step, 0.2);
    }
}