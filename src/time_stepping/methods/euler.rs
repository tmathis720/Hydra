use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
use crate::equation::fields::UpdateState;

pub struct ExplicitEuler<P: TimeDependentProblem> {
    current_time: P::Time,
    time_step: P::Time,
    start_time: P::Time,
    end_time: P::Time,
}

impl<P: TimeDependentProblem> ExplicitEuler<P> {
    pub fn new(time_step: P::Time, start_time: P::Time, end_time: P::Time) -> Self {
        Self {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
        }
    }
}

impl<P> TimeStepper<P> for ExplicitEuler<P>
where
    P: TimeDependentProblem,
    P::State: UpdateState,
    P::Time: From<f64> + Into<f64>,
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
        let mut derivative = problem.initial_state(); // Initialize derivative
        problem.compute_rhs(current_time, state, &mut derivative)?;

        // Update the state: state = state + dt * derivative
        let dt_f64: f64 = dt.into();
        state.update_state(&derivative, dt_f64);

        self.current_time = current_time + dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError> {
        // For simplicity, not implemented
        unimplemented!()
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
}
