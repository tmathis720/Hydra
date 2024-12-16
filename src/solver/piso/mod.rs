pub mod predictor;
pub mod pressure_correction;
pub mod velocity_correction;
pub mod nonlinear_loop;
pub mod boundary;

use crate::{
    domain::mesh::Mesh,
    time_stepping::{TimeDependentProblem, TimeStepper},
    equation::{
        fields::{Fields, Fluxes},
        momentum_equation::MomentumEquation,
    },
    boundary::bc_handler::BoundaryConditionHandler,
};

/// Errors specific to the PISO solver.
#[derive(Debug)]
pub enum PISOError {
    ConvergenceFailure(String),
    MatrixError(String),
    TimeSteppingError(String),
}

impl From<String> for PISOError {
    fn from(err: String) -> Self {
        PISOError::ConvergenceFailure(err)
    }
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

        // Extract required fields
        let mut fields = Fields::new();
        let mut fluxes = Fluxes::new();
        let mut boundary_handler = BoundaryConditionHandler::new();

        // Step 1: Predictor
        let momentum_equation = MomentumEquation::new(); // Initialize momentum equation
        predictor::predict_velocity(
            &self.mesh,
            &mut fields,
            &mut fluxes,
            &boundary_handler,
            &momentum_equation,
            current_time,
        )
        .map_err(|e| PISOError::MatrixError(format!("Predictor step failed: {}", e)))?;

        // Step 2: Pressure Correction Loop
        for iteration in 0..self.config.max_iterations {
            let pressure_correction_result = pressure_correction::solve_pressure_poisson(
                &self.mesh,
                &mut fields,
                &fluxes,
                &boundary_handler,
                &mut *self.time_stepper.linear_solver(), // Obtain the linear solver
            )
            .map_err(|e| PISOError::MatrixError(format!("Pressure correction step failed: {}", e)))?;

            velocity_correction::correct_velocity(
                &self.mesh,
                &mut fields,
                &pressure_correction_result.pressure_correction,
                &boundary_handler,
            )
            .map_err(|e| PISOError::MatrixError(format!("Velocity correction step failed: {}", e)))?;

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
        self.time_stepper
            .step(problem, dt, current_time, state)
            .map_err(|e| PISOError::TimeSteppingError(format!("{:?}", e)))?;

        Ok(())
    }
}
