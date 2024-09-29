use crate::linalg::Matrix;
use crate::time_stepping::TimeStepper;
use crate::time_stepping::TimeSteppingError;
use crate::time_stepping::TimeDependentProblem;

pub struct ForwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for ForwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut rhs = problem.initial_condition();

        problem.compute_rhs(time, state, &mut rhs)?;
        problem.axpy(dt, &rhs, state)?;

        Ok(())
    }

    fn set_time_interval(&mut self, _start_time: P::Time, _end_time: P::Time) {
        // Implement setting time interval if needed
    }

    fn set_time_step(&mut self, _dt: P::Time) {
        // Implement setting time step if needed
    }
}
