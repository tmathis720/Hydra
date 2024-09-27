use crate::time_stepping::ts::TimeStepper;
use crate::time_stepping::ts::TimeDependentProblem;
use crate::time_stepping::ts::ProblemError;

pub struct ForwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for ForwardEuler {
    fn step(&mut self, problem: &P, time: P::Time, dt: P::Time, state: &mut P::State) -> Result<(), ProblemError> {
        let mut rhs = state.clone(); // Create a copy of the state to store the result
        problem.compute_rhs(time, state, &mut rhs)?;
        
        // Update state: u_new = u_old + dt * f(t, u)
        for i in 0..state.len() {
            state[i] += dt * rhs[i];
        }
        Ok(())
    }
}
