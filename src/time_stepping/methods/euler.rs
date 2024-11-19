use crate::time_stepping::adaptivity::error_estimate::estimate_error;
use crate::time_stepping::adaptivity::step_size_control::adjust_step_size;
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
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        let mut error = f64::INFINITY;
        let mut dt = self.time_step.into();
        while error > tol {
            // Compute high-order step
            let mut temp_state = state.clone();
            let mid_dt = P::Time::from(0.5 * dt);
            self.step(problem, mid_dt, self.current_time, &mut temp_state)?;

            // Compute full step for comparison
            let mut high_order_state = temp_state.clone();
            self.step(problem, mid_dt, self.current_time + mid_dt, &mut high_order_state)?;

            error = estimate_error(problem, state, P::Time::from(dt))?;
            dt = adjust_step_size(dt, error, tol, 0.9, 2.0);
        }
        self.set_time_step(P::Time::from(dt));
        Ok(P::Time::from(dt))
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
