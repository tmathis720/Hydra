use crate::time_stepping::ts::TimeStepper;
use crate::time_stepping::ts::TimeDependentProblem;
use crate::time_stepping::ts::ProblemError;
use crate::solver::ksp::KSP;

pub struct BackwardEuler {
    solver: dyn KSP,
}

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(&mut self, problem: &P, time: P::Time, dt: P::Time, state: &mut P::State) -> Result<(), ProblemError> {
        let mut rhs = state.clone();
        problem.compute_rhs(time, state, &mut rhs)?;
        
        let mut system_matrix = problem.compute_jacobian(time + dt, state).ok_or(ProblemError::MissingJacobian)?;
        
        // Solve the linear system A * u_new = rhs
        self.solver.solve(&mut system_matrix, &rhs, state);
        Ok(())
    }
}
