use hydra::solver::{KSP, ConjugateGradient};

impl<State, Time> TimeStepper for BackwardEuler<State, Time>
where
    State: Clone + Default,
    Time: Copy,
{
    fn step(
        &mut self,
        problem: &dyn TimeDependentProblem<State = State, Time = Time>,
        current_time: Time,
        dt: Time,
        state: &mut State,
    ) -> Result<(), TimeSteppingError> {
        // Assemble the system: (I - dt * A) * x_new = x_old + dt * b
        // Use the solver to solve for x_new
        let mut solver = ConjugateGradient::new(1000, 1e-6);
        solver.set_preconditioner(Box::new(Jacobi::new(&a_matrix)));
        let result = solver.solve(&system_matrix, &rhs_vector, &mut x_new);

        if result.converged {
            *state = x_new;
            Ok(())
        } else {
            Err(TimeSteppingError::SolverError("Solver did not converge".into()))
        }
    }
}