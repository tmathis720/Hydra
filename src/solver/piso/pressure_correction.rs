use faer::Mat;

use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{
        fields::{Fields, Fluxes},
        gradient::{Gradient, GradientCalculationMethod},
    }, solver::KSP, MeshEntity, Section,
    interface_adapters::section_matvec_adapter::SectionMatVecAdapter,
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
    let num_cells = mesh.count_entities(&MeshEntity::Cell(0));
    let mut pressure_matrix = Mat::<f64>::zeros(num_cells, num_cells);
    let mut rhs = Section::<Scalar>::new();

    assemble_pressure_poisson(mesh, fields, fluxes, boundary_handler, &mut pressure_matrix, &mut rhs)?;

    let mut pressure_correction = Section::<Scalar>::new();
    let mut rhs_adapter = SectionMatVecAdapter::new(&rhs);
    linear_solver
        .solve(&mut pressure_matrix, &mut pressure_correction, &mut rhs_adapter)
        .map_err(|e| format!("Pressure Poisson solver failed: {}", e))?;

    update_pressure_field(fields, &pressure_correction)?;
    correct_velocity_field(mesh, fields, &pressure_correction, boundary_handler)?;

    let residual = compute_residual(&mesh, &pressure_matrix, &pressure_correction, &rhs);

    Ok(PressureCorrectionResult { residual })
}

/// Assembles the pressure Poisson equation matrix and RHS.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: Current physical fields, including velocity and pressure.
/// - `fluxes`: Fluxes from the predictor step.
/// - `boundary_handler`: Handles boundary conditions.
/// - `matrix`: The dense matrix for the pressure Poisson equation.
/// - `rhs`: The right-hand side vector for pressure correction.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message on failure.
fn assemble_pressure_poisson(
    mesh: &Mesh,
    fields: &Fields,
    fluxes: &Fluxes,
    boundary_handler: &BoundaryConditionHandler,
    matrix: &mut Mat<f64>,
    rhs: &mut Section<Scalar>,
) -> Result<(), String> {
    let num_cells = mesh.count_entities(&MeshEntity::Cell(0));

    // Ensure the matrix is the correct size
    assert_eq!(matrix.nrows(), num_cells, "Matrix row count mismatch.");
    assert_eq!(matrix.ncols(), num_cells, "Matrix column count mismatch.");

    // Assemble the matrix and RHS
    for cell in mesh.get_cells() {
        let cell_id = cell.get_id();
        let divergence = fluxes
            .momentum_fluxes
            .restrict(&cell)
            .map_or(0.0, |flux| flux.magnitude());
        rhs.set_data(cell, Scalar(divergence));

        for neighbor in mesh.get_ordered_neighbors(&cell) {
            let neighbor_id = neighbor.get_id();
            let coefficient = compute_pressure_coefficient(mesh, fields, &cell, &neighbor)?;
            matrix.write(cell_id, neighbor_id, coefficient); // Update the off-diagonal entry
        }

        let diagonal_coefficient = compute_pressure_diagonal_coefficient(mesh, fields, &cell)?;
        matrix.write(cell_id, cell_id, diagonal_coefficient); // Update the diagonal entry
    }

    // Apply boundary conditions
    boundary_handler.apply_bc(
        &mut matrix.as_mut(),
        rhs,
        &boundary_handler.get_boundary_faces(),
        &mesh.entity_to_index_map().into(),
        0.0, // Pass time as needed
    );

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
    let velocity_field = match fields.vector_fields.get_mut("velocity") {
        Some(field) => field,
        None => return Err("Velocity field not found in the fields structure.".to_string()),
    };

    let gradient_method = GradientCalculationMethod::FiniteVolume;
    let mut gradient_calculator = Gradient::new(mesh, boundary_handler, gradient_method);

    // Compute the gradient of the pressure correction
    let mut pressure_gradient = Section::<Vector3>::new();
    gradient_calculator.compute_gradient(pressure_correction, &mut pressure_gradient, 0.0)
    .map_err(|e| format!("Gradient computation failed: {:?}", e))?;

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
fn compute_residual(
    mesh: &Mesh,
    matrix: &Mat<f64>,
    pressure_correction: &Section<Scalar>,
    rhs: &Section<Scalar>,
) -> f64 {
    let mut residual: f64 = 0.0;

    // Convert `pressure_correction` to a dense vector representation
    let mut pressure_vec = vec![0.0; matrix.ncols()]; // Use matrix.ncols() instead of pressure_correction.len()
    for entry in pressure_correction.data.iter() {
        let (key, scalar) = entry.pair();
        let index = mesh.entity_to_index_map().get(key).ok_or("Invalid key")?;
        pressure_vec[*index] = scalar.0;
    }

    // Allocate a result vector for the matrix-vector multiplication
    let mut lhs_vec = vec![0.0; matrix.nrows()];

    // Perform the matrix-vector multiplication
    for i in 0..matrix.nrows() {
        lhs_vec[i] = (0..matrix.ncols())
            .map(|j| matrix.read(i, j) * pressure_vec[j])
            .sum::<f64>();
    }

    // Compute the residual as the L2 norm of the difference
    for entry in rhs.data.iter() {
        let (key, value) = entry.pair();
        let diff = lhs_vec[*key] - value.0;
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
    // Step 1: Compute the distance between the centroids of the two cells
    let distance = mesh.get_distance_between_cells(cell, neighbor);

    // Step 2: Identify the shared face between the two cells
    let shared_faces = mesh.get_faces_of_cell(cell)
        .ok_or("Failed to retrieve faces of the cell")?;
    let shared_face = shared_faces.iter()
        .find(|face| {
            let cells_sharing_face = mesh.get_cells_sharing_face(&face.key());
            cells_sharing_face.contains_key(neighbor)
        })
        .map(|face| face.key().clone()) // Clone the key to resolve the lifetime issue
        .ok_or("No shared face found between the cell and its neighbor")?;


    // Step 3: Compute the area of the shared face
    let area = mesh.get_face_area(&shared_face)
        .ok_or("Failed to compute the area of the shared face")?;

    // Step 4: Get the density from the fields structure
    let density = fields.scalar_fields.get("density")
        .and_then(|f| f.restrict(cell))
        .map_or(1.0, |d| d.0); // Default density is 1.0 if not found

    // Step 5: Compute and return the coefficient
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
    for neighbor in mesh.get_ordered_neighbors(cell) {
        coefficient_sum += compute_pressure_coefficient(mesh, fields, cell, &neighbor)?;
    }
    Ok(-coefficient_sum) // Diagonal entry is negative of sum of neighbor coefficients
}
