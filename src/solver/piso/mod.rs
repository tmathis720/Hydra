pub mod predictor;
pub mod pressure_correction;
pub mod velocity_correction;
pub mod nonlinear_loop;
pub mod boundary;

use crate::{
    domain::mesh::Mesh,
    time_stepping::{TimeDependentProblem, TimeStepper},
};

/// Errors specific to the PISO solver.
#[derive(Debug)]
pub enum PISOError {
    ConvergenceFailure(String),
    MatrixError(String),
    TimeSteppingError(String),
}

/// Configuration options for the PISO solver.
pub struct PISOConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub relaxation_factor: f64,
}

/// The main struct for the PISO solver.
pub struct PISOSolver<P>
where
    P: TimeDependentProblem,
{
    mesh: Mesh,
    time_stepper: Box<dyn TimeStepper<P>>,
    config: PISOConfig,
}

impl<P> PISOSolver<P>
where
    P: TimeDependentProblem,
{
    /// Creates a new instance of the PISO solver.
    ///
    /// # Parameters
    /// - `mesh`: The computational mesh.
    /// - `time_stepper`: The time-stepping method to use.
    /// - `config`: Configuration options for the solver.
    pub fn new(
        mesh: Mesh,
        time_stepper: Box<dyn TimeStepper<P>>,
        config: PISOConfig,
    ) -> Self {
        Self {
            mesh,
            time_stepper,
            config,
        }
    }

    /// Executes the PISO algorithm for a single time step.
    ///
    /// # Parameters
    /// - `problem`: The problem defining the governing equations.
    /// - `state`: The current state of the problem.
    ///
    /// # Returns
    /// Result<(), PISOError> indicating success or failure.
    pub fn solve(
        &mut self,
        problem: &P,
        state: &mut P::State,
    ) -> Result<(), PISOError> {
        let current_time = self.time_stepper.current_time();
        let dt = self.time_stepper.get_time_step();

        // Step 1: Predictor
        predictor::predict_velocity(&self.mesh, problem, state)?;

        // Step 2: Pressure Correction Loop
        for iteration in 0..self.config.max_iterations {
            let pressure_correction_result = pressure_correction::solve_pressure_poisson(
                &self.mesh,
                problem,
                state,
            )?;

            velocity_correction::correct_velocity(
                &self.mesh,
                problem,
                state,
                &pressure_correction_result,
            )?;

            if pressure_correction_result.residual < self.config.tolerance {
                break;
            }

            if iteration == self.config.max_iterations - 1 {
                return Err(PISOError::ConvergenceFailure(
                    "Pressure correction did not converge.".to_string(),
                ));
            }
        }

        // Step 3: Time Integration
        self.time_stepper.step(problem, dt, current_time, state)
            .map_err(|e| PISOError::TimeSteppingError(format!("{:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::mesh::Mesh, equation::fields::Fields, time_stepping::{TimeDependentProblem, TimeStepper}, Matrix
    };

    /// Dummy problem for testing the PISO solver.
    struct DummyProblem;

    impl TimeDependentProblem for DummyProblem {
        type State = Fields;
        type Time = f64;

        fn compute_rhs(
            &self,
            _time: Self::Time,
            _state: &Self::State,
            _derivative: &mut Self::State,
        ) -> Result<(), crate::time_stepping::TimeSteppingError> {
            Ok(())
        }

        fn initial_state(&self) -> Self::State {
            Fields::default()
        }

        fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
            None
        }

        fn solve_linear_system(
            &self,
            _matrix: &mut dyn Matrix<Scalar = f64>,
            _state: &mut Self::State,
            _rhs: &Self::State,
        ) -> Result<(), crate::time_stepping::TimeSteppingError> {
            Ok(())
        }
    }

    #[test]
    fn test_piso_solver() {
        let mesh = Mesh::new();
        let time_stepper = Box::new(TimeStepper::new(0.0, 1.0, 0.01));
        let config = PISOConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            relaxation_factor: 0.7,
        };

        let mut solver = PISOSolver::new(mesh, time_stepper, config);
        let problem = DummyProblem;
        let mut state = problem.initial_state();

        assert!(solver.solve(&problem, &mut state).is_ok());
    }
}
