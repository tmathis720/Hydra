use crate::{equation::fields::UpdateState, time_stepping::{TimeDependentProblem, TimeSteppingError}};

pub fn estimate_error<P>(
    problem: &P,
    state: &P::State,
    dt: P::Time,
) -> Result<f64, TimeSteppingError>
where
    P: TimeDependentProblem,
    P::State: Clone,
    P::Time: Into<f64> + From<f64> + Copy,
{
    // Clone the initial state to preserve the original for comparison
    let mut single_step_state = state.clone();
    let mut two_half_steps_state = state.clone();

    // Perform a single full step
    let mut rhs = state.clone();
    problem.compute_rhs(
        P::Time::from(0.0), // Assume starting time is 0 for simplicity
        state,
        &mut rhs,
    )?;
    let single_rhs = rhs.clone(); // Save RHS to avoid re-borrowing
    single_step_state.update_state(&single_rhs, dt.into());

    // Perform two half-steps
    let half_dt = P::Time::from(dt.into() * 0.5);
    let mut rhs_half = state.clone();
    problem.compute_rhs(
        P::Time::from(0.0),
        state,
        &mut rhs_half,
    )?;
    let first_half_rhs = rhs_half.clone();
    two_half_steps_state.update_state(&first_half_rhs, half_dt.into());

    let mut second_rhs_half = two_half_steps_state.clone();
    problem.compute_rhs(
        P::Time::from(half_dt.into()),
        &two_half_steps_state,
        &mut second_rhs_half,
    )?;
    two_half_steps_state.update_state(&second_rhs_half, half_dt.into());

    // Compute the error as the difference between the two states
    let error = compute_state_difference(&single_step_state, &two_half_steps_state)?;

    Ok(error)
}

// Helper function to compute the difference between two states
fn compute_state_difference<S>(state1: &S, state2: &S) -> Result<f64, TimeSteppingError>
where
    S: Clone + UpdateState,
{
    // This assumes that the state has a method to calculate norm or difference.
    // Replace with appropriate norm calculation for the state type.
    let difference_norm = state1
        .clone()
        .difference(state2)
        .norm(); // You need to implement difference and norm in the `UpdateState` trait

    Ok(difference_norm)
}
