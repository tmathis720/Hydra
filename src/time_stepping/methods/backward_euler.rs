use crate::time_stepping::ts::{TimeStepper, TimeDependentProblem, ProblemError};
use crate::solver::ksp::KSP;
use crate::solver::ksp::KSPError;

pub struct BackwardEuler {
    solver: Box<dyn KSP>, // Use dynamic dispatch for the KSP solver
}

impl BackwardEuler {
    pub fn new(solver: Box<dyn KSP>) -> Self {
        Self { solver }
    }
}

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(&mut self, problem: &P, time: P::Time, dt: P::Time, state: &mut P::State) -> Result<(), ProblemError> {
        // Create a copy of the current state to hold the right-hand side (rhs)
        let mut rhs = state.clone();
        
        // Compute the right-hand side: f(t + dt, u)
        problem.compute_rhs(time + dt, state, &mut rhs)?;
        
        // Compute the Jacobian matrix at the new time level (for implicit solve)
        let mut system_matrix = problem
            .compute_jacobian(time + dt, state)
            .ok_or(ProblemError::MissingJacobian)?;

        // Solve the linear system: A * u_new = rhs
        self.solver
            .solve(&mut system_matrix, &rhs, state)
            .map_err(|err| ProblemError::SolverError(format!("KSP solver error: {}", err)))?;
        
        Ok(())
    }

    fn set_tolerances(&mut self, _rel_tol: f64, _abs_tol: f64) {
        // If needed, set tolerances for the KSP solver
        self.solver.set_tolerances(_rel_tol, _abs_tol);
    }

    fn adaptive_step(&mut self, _problem: &P, _time: P::Time, _state: &mut P::State) -> Result<(), ProblemError> {
        // Adaptive stepping is not implemented for Backward Euler
        unimplemented!();
    }
}
