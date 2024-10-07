
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

