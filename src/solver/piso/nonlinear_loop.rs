use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh, equation::fields::Fields, solver::{piso::{
        predictor::predict_velocity,
        pressure_correction::solve_pressure_poisson,
        velocity_correction::correct_velocity,
    }, KSP}
};

/// Configuration options for the nonlinear loop.
pub struct NonlinearLoopConfig {
    pub max_iterations: usize, // Maximum number of iterations in the loop
    pub tolerance: f64,        // Convergence tolerance for residuals
}

/// Executes the nonlinear loop of the PISO algorithm.
///
/// This loop iterates between the predictor, pressure correction, and velocity correction
/// steps until the solution converges or the maximum number of iterations is reached.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: The current state of the physical fields.
/// - `boundary_handler`: Handles boundary conditions for the domain.
/// - `linear_solver`: A sparse linear solver for solving the pressure Poisson equation.
/// - `config`: Configuration options for the nonlinear loop.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` if the loop converges, or an error message if it fails.
pub fn solve_nonlinear_system(
    mesh: &Mesh,
    fields: &mut Fields,
    boundary_handler: &BoundaryConditionHandler,
    linear_solver: &mut dyn KSP,
    config: &NonlinearLoopConfig,
) -> Result<(), String> {
    let mut fluxes = crate::equation::fields::Fluxes::new(); // Initialize flux container

    for iteration in 0..config.max_iterations {
        // Step 1: Predictor
        predict_velocity(mesh, fields, &mut fluxes, boundary_handler, &crate::equation::momentum_equation::MomentumEquation { 
            params: crate::equation::momentum_equation::MomentumParameters {
                density: 1.0, // Default fluid density (adjustable based on setup)
                viscosity: 0.001, // Default dynamic viscosity
            }
        }, 0.0).map_err(|e| format!("Predictor step failed: {}", e))?;

        // Step 2: Pressure Correction
        let pressure_correction_result = solve_pressure_poisson(
            mesh,
            fields,
            &fluxes,
            boundary_handler,
            linear_solver,
        ).map_err(|e| format!("Pressure correction step failed: {}", e))?;

        // Step 3: Velocity Correction
        correct_velocity(mesh, fields, &fields.scalar_fields["pressure"], boundary_handler)
            .map_err(|e| format!("Velocity correction step failed: {}", e))?;

        // Check convergence based on residual
        if pressure_correction_result.residual < config.tolerance {
            println!(
                "Nonlinear loop converged after {} iterations with residual {}.",
                iteration + 1,
                pressure_correction_result.residual
            );
            return Ok(());
        }

        println!(
            "Iteration {}: residual = {}",
            iteration + 1,
            pressure_correction_result.residual
        );
    }

    Err(format!(
        "Nonlinear loop did not converge after {} iterations.",
        config.max_iterations
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::BoundaryConditionHandler,
        domain::mesh::Mesh,
        equation::fields::Fields, Section,
    };

    #[test]
    fn test_solve_nonlinear_system() {
        // 1. Setup: Create a mock mesh, boundary handler, and fields
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut fields = Fields::new();

        // Initialize fields
        fields.scalar_fields.insert("pressure".to_string(), Section::new());
        fields.vector_fields.insert("velocity".to_string(), Section::new());

        // Create a mock linear solver
        let mut linear_solver = KSP::new();

        // Configure the nonlinear loop
        let config = NonlinearLoopConfig {
            max_iterations: 10,
            tolerance: 1e-6,
        };

        // 2. Execute: Run the nonlinear loop
        let result = solve_nonlinear_system(&mesh, &mut fields, &boundary_handler, &mut linear_solver, &config);

        // 3. Verify: Ensure the function completes successfully
        assert!(result.is_ok());
    }
}
