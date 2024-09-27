pub struct ForwardEuler;

impl TimeStepper for ForwardEuler {
    type State = Vec<f64>; // or a custom state type
    type Time = f64;

    fn step(
        &mut self,
        problem: &dyn TimeDependentProblem<State = Self::State, Time = Self::Time>,
        current_time: Self::Time,
        dt: Self::Time,
        state: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        let mut derivative = state.clone();
        problem.compute_rhs(current_time, state, &mut derivative)?;
        for i in 0..state.len() {
            state[i] += dt * derivative[i];
        }
        Ok(())
    }
}
