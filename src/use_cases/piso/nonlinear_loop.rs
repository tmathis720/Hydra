use crate::{
    boundary::bc_handler::BoundaryConditionHandler, solver::KSP, domain::mesh::Mesh, equation::fields::Fields, use_cases::piso::{
        predictor::predict_velocity,
        pressure_correction::solve_pressure_poisson,
        velocity_correction::correct_velocity,
    }
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
    println!("Starting nonlinear solver loop with max_iterations: {} and tolerance: {}", config.max_iterations, config.tolerance);

    for iteration in 0..config.max_iterations {
        println!("Iteration {}: Starting predictor step.", iteration + 1);

        // Step 1: Predictor
        predict_velocity(
            mesh,
            fields,
            &mut fluxes,
            boundary_handler,
            &crate::equation::momentum_equation::MomentumEquation {
                params: crate::equation::momentum_equation::MomentumParameters {
                    density: 1.0, // Default fluid density (adjustable based on setup)
                    viscosity: 0.001, // Default dynamic viscosity
                },
            },
            0.0,
        )
        .map_err(|e| {
            println!("Iteration {}: Predictor step failed. Error: {}", iteration + 1, e);
            format!("Predictor step failed: {}", e)
        })?;

        println!("Iteration {}: Predictor step completed successfully.", iteration + 1);

        // Step 2: Pressure Correction
        println!("Iteration {}: Starting pressure correction step.", iteration + 1);
        let pressure_correction_result = solve_pressure_poisson(
            mesh,
            fields,
            &fluxes,
            boundary_handler,
            linear_solver,
        )
        .map_err(|e| {
            println!("Iteration {}: Pressure correction step failed. Error: {}", iteration + 1, e);
            format!("Pressure correction step failed: {}", e)
        })?;

        println!(
            "Iteration {}: Pressure correction step completed. Residual: {}",
            iteration + 1, pressure_correction_result.residual
        );

        // Step 3: Velocity Correction
        println!("Iteration {}: Starting velocity correction step.", iteration + 1);
        let pressure_field = fields
            .scalar_fields
            .get("pressure")
            .cloned()
            .ok_or_else(|| {
                let err_msg = format!("Pressure field missing in fields at iteration {}.", iteration + 1);
                println!("{}", err_msg);
                err_msg
            })?;
        
        correct_velocity(mesh, fields, &pressure_field, boundary_handler)
            .map_err(|e| {
                println!("Iteration {}: Velocity correction step failed. Error: {}", iteration + 1, e);
                format!("Velocity correction step failed: {}", e)
            })?;

        println!("Iteration {}: Velocity correction step completed successfully.", iteration + 1);

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
            "Iteration {}: Residual = {}, continuing to next iteration.",
            iteration + 1,
            pressure_correction_result.residual
        );
    }

    println!(
        "Nonlinear loop did not converge after {} iterations.",
        config.max_iterations
    );

    Err(format!(
        "Nonlinear loop did not converge after {} iterations.",
        config.max_iterations
    ))
}
