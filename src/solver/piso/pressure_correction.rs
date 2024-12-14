use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{
        fields::{Fields, Fluxes},
        gradient::{Gradient, GradientCalculationMethod},
    }, solver::KSP, Matrix, MeshEntity, Section
};

/// Results from the pressure correction step.
pub struct PressureCorrectionResult {
    pub residual: f64, // Residual of the pressure correction
}

/// Solves the pressure Poisson equation to compute the pressure correction and updates the velocity field.
///
/// # Parameters
/// - `mesh`: Reference to the computational mesh.
/// - `fields`: The current state of physical fields, including velocity and pressure.
/// - `fluxes`: Container for fluxes computed during the predictor step.
/// - `boundary_handler`: Handler for boundary conditions.
/// - `linear_solver`: The solver for the sparse linear system (e.g., CG or GMRES).
///
/// # Returns
/// - `Result<PressureCorrectionResult, String>`: Returns a `PressureCorrectionResult` on success or an error message on failure.
pub fn solve_pressure_poisson(
    mesh: &Mesh,
    fields: &mut Fields,
    fluxes: &Fluxes,
    boundary_handler: &BoundaryConditionHandler,
    linear_solver: &mut dyn KSP,
) -> Result<PressureCorrectionResult, String> {
    // 1. Assemble the pressure Poisson equation.
    let mut pressure_matrix = Matrix::new();
    let mut rhs = Section::<Scalar>::new(); // Right-hand side for pressure correction

    assemble_pressure_poisson(mesh, fields, fluxes, boundary_handler, &mut pressure_matrix, &mut rhs)?;

    // 2. Solve the sparse linear system for pressure correction.
    let mut pressure_correction = Section::<Scalar>::new();
    linear_solver
        .solve(&mut pressure_matrix, &mut pressure_correction, &rhs)
        .map_err(|e| format!("Pressure Poisson solver failed: {}", e))?;

    // 3. Update the pressure field using the correction.
    update_pressure_field(fields, &pressure_correction)?;

    // 4. Correct the velocity field to ensure divergence-free flow.
    correct_velocity_field(mesh, fields, &pressure_correction, boundary_handler)?;

    // 5. Compute the residual of the pressure correction (for convergence check).
    let residual = compute_residual(&pressure_matrix, &pressure_correction, &rhs);

    Ok(PressureCorrectionResult { residual })
}

/// Assembles the pressure Poisson equation matrix and RHS.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: Current physical fields, including velocity and pressure.
/// - `fluxes`: Fluxes from the predictor step.
/// - `boundary_handler`: Handles boundary conditions.
/// - `matrix`: The sparse matrix for the pressure Poisson equation.
/// - `rhs`: The right-hand side vector for pressure correction.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message on failure.
fn assemble_pressure_poisson<T: Matrix>(
    mesh: &Mesh,
    fields: &Fields,
    fluxes: &Fluxes,
    boundary_handler: &BoundaryConditionHandler,
    matrix: &mut T,
    rhs: &mut Section<Scalar>,
) -> Result<(), String> {
    // Iterate over cells to assemble the pressure Poisson matrix and RHS
    for cell in mesh.get_cells() {
        let divergence = fluxes.momentum_fluxes.restrict(&cell).map_or(0.0, |flux| flux.magnitude());
        rhs.set_data(cell, Scalar(divergence)); // Use the momentum divergence as the RHS term

        // Add pressure matrix coefficients based on cell neighbors
        for neighbor in mesh.get_cell_neighbors(&cell) {
            let coefficient = compute_pressure_coefficient(mesh, fields, &cell, &neighbor)?;
            matrix.add_entry(cell, neighbor, coefficient);
        }

        // Diagonal entry
        let diagonal_coefficient = compute_pressure_diagonal_coefficient(mesh, fields, &cell)?;
        matrix.add_entry(cell, cell, diagonal_coefficient);
    }

    // Apply boundary conditions
    boundary_handler.apply_pressure_poisson_bc(mesh, matrix, rhs)?;

    Ok(())
}

/// Updates the pressure field using the pressure correction.
///
/// # Parameters
/// - `fields`: Physical fields, including pressure.
/// - `pressure_correction`: The computed pressure correction.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message on failure.
fn update_pressure_field(fields: &mut Fields, pressure_correction: &Section<Scalar>) -> Result<(), String> {
    match fields.scalar_fields.get_mut("pressure") {
        Some(pressure_field) => {
            pressure_field.update_with_derivative(pressure_correction, 1.0); // Apply correction
            Ok(())
        }
        None => Err("Pressure field not found in the fields structure.".to_string()),
    }
}

/// Corrects the velocity field using the pressure correction.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: The current state of physical fields, including velocity.
/// - `pressure_correction`: The computed pressure correction.
/// - `boundary_handler`: Handles boundary conditions.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message on failure.
fn correct_velocity_field(
    mesh: &Mesh,
    fields: &mut Fields,
    pressure_correction: &Section<Scalar>,
    boundary_handler: &BoundaryConditionHandler,
) -> Result<(), String> {
    let mut velocity_field = match fields.vector_fields.get_mut("velocity") {
        Some(field) => field,
        None => return Err("Velocity field not found in the fields structure.".to_string()),
    };

    let gradient_method = GradientCalculationMethod::FiniteVolume;
    let mut gradient_calculator = Gradient::new(mesh, boundary_handler, gradient_method);

    // Compute the gradient of the pressure correction
    let mut pressure_gradient = Section::<Vector3>::new();
    gradient_calculator.compute_gradient(pressure_correction, &mut pressure_gradient, 0.0)?;

    // Subtract the pressure gradient from the velocity field
    velocity_field.update_with_derivative(&pressure_gradient, -1.0); // Correct the velocity field

    Ok(())
}

/// Computes the residual of the pressure correction system.
///
/// # Parameters
/// - `matrix`: The sparse matrix of the pressure Poisson equation.
/// - `pressure_correction`: The computed pressure correction.
/// - `rhs`: The right-hand side vector.
///
/// # Returns
/// - `f64`: The computed residual value.
fn compute_residual<T: Matrix>(
    matrix: &T,
    pressure_correction: &Section<Scalar>,
    rhs: &Section<Scalar>,
) -> f64 {
    let mut residual = 0.0;
    for (cell, &value) in rhs.data.iter() {
        let lhs = matrix.apply_to_vector(&pressure_correction, cell);
        let diff = lhs - value.0;
        residual += diff * diff;
    }
    residual.sqrt()
}

/// Computes the coefficient for a pressure neighbor in the pressure Poisson equation.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: Physical fields, including velocity and pressure.
/// - `cell`: The current cell.
/// - `neighbor`: The neighbor cell.
///
/// # Returns
/// - `Result<f64, String>`: The computed coefficient or an error message.
fn compute_pressure_coefficient(
    mesh: &Mesh,
    fields: &Fields,
    cell: &MeshEntity,
    neighbor: &MeshEntity,
) -> Result<f64, String> {
    // Compute pressure coefficient (e.g., based on face area and distance between cells)
    let distance = mesh.compute_cell_distance(cell, neighbor)?;
    let area = mesh.compute_face_area_between_cells(cell, neighbor)?;
    let density = fields.scalar_fields.get("density").and_then(|f| f.restrict(cell)).map_or(1.0, |d| d.0);
    Ok(area / (density * distance)) // Coefficient = Area / (Density * Distance)
}

/// Computes the diagonal coefficient for a cell in the pressure Poisson equation.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: Physical fields, including velocity and pressure.
/// - `cell`: The current cell.
///
/// # Returns
/// - `Result<f64, String>`: The computed diagonal coefficient or an error message.
fn compute_pressure_diagonal_coefficient(
    mesh: &Mesh,
    fields: &Fields,
    cell: &MeshEntity,
) -> Result<f64, String> {
    // Sum coefficients for all neighbors
    let mut coefficient_sum = 0.0;
    for neighbor in mesh.get_cell_neighbors(cell) {
        coefficient_sum += compute_pressure_coefficient(mesh, fields, cell, &neighbor)?;
    }
    Ok(-coefficient_sum) // Diagonal entry is negative of sum of neighbor coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::BoundaryConditionHandler,
        domain::mesh::Mesh,
        equation::fields::{Fields, Fluxes},
    };

    #[test]
    fn test_solve_pressure_poisson() {
        // Create a mock mesh, fields, and boundary handler
        let mesh = Mesh::new();
        let mut fields = Fields::new();
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // Add dummy data to fields and fluxes
        fields.scalar_fields.insert("pressure".to_string(), Section::new());
        fields.vector_fields.insert("velocity".to_string(), Section::new());
        fluxes.momentum_fluxes = Section::new();

        // Create a mock linear solver
        let mut linear_solver = KSP::new();

        // Run the pressure correction step
        let result = solve_pressure_poisson(&mesh, &mut fields, &fluxes, &boundary_handler, &mut linear_solver);

        // Check that the function executes successfully
        assert!(result.is_ok());
        assert!(result.unwrap().residual >= 0.0);
    }
}
