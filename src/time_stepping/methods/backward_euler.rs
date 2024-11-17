use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct BackwardEuler {
    current_time: f64,
    time_step: f64,
}

impl BackwardEuler {
    pub fn new(start_time: f64, time_step: f64) -> Self {
        Self {
            current_time: start_time,
            time_step,
        }
    }
}

impl<P> TimeStepper<P> for BackwardEuler
where
    P: TimeDependentProblem,
    P::Time: From<f64> + Into<f64>,
{
    fn current_time(&self) -> P::Time {
        P::Time::from(self.current_time)
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time.into();
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let dt_f64: f64 = dt.into();
        self.time_step = dt_f64;

        let mut matrix = problem
            .get_matrix()
            .ok_or(TimeSteppingError::SolverError("Matrix is required for Backward Euler.".into()))?;
        let mut rhs = state.clone();

        problem.compute_rhs(current_time, state, &mut rhs)?;
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;

        // Update the current time
        self.current_time += dt_f64;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError> {
        // Adaptive step logic (placeholder)
        Ok(self.time_step.into())
    }

    fn set_time_interval(&mut self, start_time: P::Time, _end_time: P::Time) {
        self.current_time = start_time.into();
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt.into();
    }
    
    fn get_time_step(&self) -> P::Time {
        self.time_step.into()
    }
}
