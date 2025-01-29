use faer::Mat;

use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{
        fields::{Fields, Fluxes},
        gradient::{Gradient, GradientCalculationMethod},
    }, solver::KSP, MeshEntity, Section,
};

/// Results from the pressure correction step.
#[derive(Debug)]
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
    let _entity_to_index_map = mesh.entity_to_index_map();
    let num_cells = mesh.get_cells().len();

    // Initialize pressure matrix and RHS
    let mut pressure_matrix = Mat::<f64>::zeros(num_cells, num_cells);
    let mut rhs = Section::<Scalar>::new();

    println!("Starting assembly of pressure Poisson matrix and RHS.");
    assemble_pressure_poisson(mesh, fields, fluxes, boundary_handler, &mut pressure_matrix, &mut rhs)
        .map_err(|e| format!("Error in assembling pressure Poisson matrix: {}", e))?;

    // Validate RHS initialization
    assert_eq!(
        rhs.data.len(),
        num_cells,
        "RHS vector is incomplete. Expected {} entries, found {}.",
        num_cells,
        rhs.data.len()
    );

    // Initialize pressure correction vector
    let mut pressure_correction = Section::<Scalar>::new();
    for cell in mesh.get_cells() {
        pressure_correction.set_data(cell.clone(), Scalar(0.0));
    }

    // Solve using the linear solver
    println!("Solving the pressure Poisson equation.");
    linear_solver
        .solve(&mut pressure_matrix, &mut pressure_correction, &mut rhs)
        .map_err(|e| format!("Pressure Poisson solver failed: {}", e))?;

    // Validate correction size
    assert_eq!(
        pressure_correction.data.len(),
        num_cells,
        "Pressure correction vector is incomplete."
    );

    println!("Residual of the pressure Poisson system: {}", compute_residual(
        mesh, &pressure_matrix, &pressure_correction, &rhs
    ));

    // Update the pressure and velocity fields
    update_pressure_field(fields, &pressure_correction)
        .map_err(|e| format!("Error updating pressure field: {}", e))?;
    correct_velocity_field(mesh, fields, &pressure_correction, boundary_handler)
        .map_err(|e| format!("Error correcting velocity field: {}", e))?;

    Ok(PressureCorrectionResult {
        residual: compute_residual(mesh, &pressure_matrix, &pressure_correction, &rhs),
    })
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
    _boundary_handler: &BoundaryConditionHandler,
    matrix: &mut Mat<f64>,
    rhs: &mut Section<Scalar>,
) -> Result<(), String> {
    let entity_to_index_map = mesh.entity_to_index_map();

    // Iterate over cells to populate matrix and RHS
    for cell in mesh.get_cells() {
        rhs.set_data(cell.clone(), Scalar(0.0)); // Initialize with zero
        let cell_index = *entity_to_index_map
            .get(&cell)
            .ok_or_else(|| format!("Cell {:?} not found in entity_to_index_map", cell))?;

        // Calculate divergence for RHS
        let divergence = fluxes
            .momentum_fluxes
            .restrict(&cell)
            .map_or(0.0, |flux| flux.magnitude());
        rhs.set_data(cell.clone(), Scalar(divergence));

        // Assemble coefficients for neighbors
        if let Ok(neighbor) = mesh.get_ordered_neighbors(&cell) {
            //let mut coefficient = 0.0;
            for neighbor in neighbor.iter() {
                let neighbor_index = *entity_to_index_map
                    .get(neighbor)
                    .ok_or_else(|| format!("Neighbor {:?} not found in entity_to_index_map", neighbor))?;
                let coefficient = compute_pressure_coefficient(mesh, fields, &cell, neighbor)?;
                matrix[(cell_index, neighbor_index)] = coefficient;
            }
        }

        // Assemble diagonal coefficient
        let diagonal_coefficient = compute_pressure_diagonal_coefficient(mesh, fields, &cell)?;
        matrix[(cell_index, cell_index)] = diagonal_coefficient;
        if diagonal_coefficient.abs() < 1e-12 {
            return Err(format!(
                "Diagonal coefficient for cell {:?} is near zero, indicating a disconnected system.",
                cell
            ));
        }
    }

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
            pressure_field.update_with_derivative(pressure_correction, 1.0).unwrap(); // Apply correction
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
    velocity_field.update_with_derivative(&pressure_gradient, -1.0).unwrap(); // Correct the velocity field

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
        let entity_to_index_map = mesh.entity_to_index_map();
        let index = match entity_to_index_map.get(key) {
            Some(idx) => idx,
            None => return f64::NAN, // or any other appropriate error handling
        };
        pressure_vec[*index] = scalar.0;
    }

    // Allocate a result vector for the matrix-vector multiplication
    let mut lhs_vec = vec![0.0; matrix.nrows()];

    // Perform the matrix-vector multiplication
    for i in 0..matrix.nrows() {
        lhs_vec[i] = (0..matrix.ncols())
            .map(|j| matrix[(i, j)] * pressure_vec[j])
            .sum::<f64>();
    }

    // Compute the residual as the L2 norm of the difference
    for entry in rhs.data.iter() {
        let (key, value) = entry.pair();
        let diff = lhs_vec[key.get_id()] - value.0;
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
    let distance = mesh.get_distance_between_cells(cell, neighbor).map_err(|e| e.to_string())?;

    if distance <= 0.0 {
        return Err(format!(
            "Invalid distance ({:?}) between cell {:?} and neighbor {:?}",
            distance, cell, neighbor
        ));
    }

    // Step 2: Identify the shared face between the two cells
    let shared_faces = mesh
        .get_faces_of_cell(cell)
        .map_err(|e| format!("Failed to retrieve faces for cell {:?}: {}", cell, e))?; // Handle Result correctly

    let shared_face = shared_faces
        .iter()
        .find_map(|face| {
            mesh.get_cells_sharing_face(&face.key())
                .ok() // Convert Result to Option
                .and_then(|cells_sharing_face| {
                    if cells_sharing_face.contains_key(neighbor) {
                        Some(face.key().clone())
                    } else {
                        None
                    }
                })
        })
        .ok_or_else(|| format!("No shared face found between cell {:?} and neighbor {:?}", cell, neighbor))?;

    // Step 3: Compute the area of the shared face
    let area = mesh
        .get_face_area(&shared_face)
        .map_err(|e| format!("Failed to retrieve area for shared face {:?}: {}", shared_face, e))?;

    if area <= 0.0 {
        return Err(format!(
            "Invalid face area ({:?}) for face {:?} between cell {:?} and neighbor {:?}",
            area, shared_face, cell, neighbor
        ));
    }

    // Step 4: Get the density from the fields structure
    let density = fields
        .scalar_fields
        .get("density")
        .and_then(|f| f.restrict(cell).ok())
        .map_or(1.0, |d| d.0); // Default density is 1.0 if not found

    // Step 5: Compute and return the coefficient
    let coefficient = area / (density * distance);
    println!(
        "Cell {:?}, Neighbor {:?}, Shared Face {:?}, Distance: {:?}, Area: {:?}, Coefficient: {:?}",
        cell, neighbor, shared_face, distance, area, coefficient
    );
    log::debug!(
        "Cell {:?}, Neighbor {:?}, Shared Face {:?}, Distance: {:?}, Area: {:?}, Coefficient: {:?}",
        cell, neighbor, shared_face, distance, area, coefficient
    );

    // Return the computed coefficient
    Ok(coefficient)
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
    if let Ok(neighbor) = mesh.get_ordered_neighbors(cell) {
        for neighbor in neighbor.iter() {
            coefficient_sum += compute_pressure_coefficient(mesh, fields, cell, neighbor)?;
        }
    }
    Ok(-coefficient_sum) // Diagonal entry is negative of sum of neighbor coefficients
}

#[cfg(test)]
mod pressure_correction_tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::mesh::Mesh,
        domain::section::{Scalar, Vector3},
        equation::fields::{Fields, Fluxes},
        interface_adapters::domain_adapter::DomainBuilder,
        solver::cg::ConjugateGradient,
    };

    /// Sets up a simple tetrahedral mesh for testing.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();
    
        // Add vertices to form a simple tetrahedron-based mesh
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 1");
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 2");
        assert!(builder.add_vertex(3, [0.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 3");
        assert!(builder.add_vertex(4, [0.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 4");
        assert!(builder.add_vertex(5, [1.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 5");
    
        // Add tetrahedron cells
        assert!(
            builder.add_tetrahedron_cell(vec![1, 2, 3, 4]).is_ok(),
            "Failed to add first tetrahedron cell"
        );
        assert!(
            builder.add_tetrahedron_cell(vec![2, 3, 5, 4]).is_ok(),
            "Failed to add second tetrahedron cell"
        );
    
        builder.build()
    }
    

    /// Sets up fields, including velocity and pressure.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();
        for cell in mesh.get_cells() {
            fields.set_vector_field_value("velocity", cell.clone(), Vector3([1.0, 1.0, 1.0]));
            fields.set_scalar_field_value("pressure", cell.clone(), Scalar(100.0));
        }
        fields
    }

    /// Sets up fluxes for testing.
    fn setup_fluxes(mesh: &Mesh) -> Fluxes {
        let fluxes = Fluxes::new();
        for cell in mesh.get_cells() {
            fluxes
                .momentum_fluxes
                .set_data(cell.clone(), Vector3([1.0, -1.0, 0.0]));
        }
        fluxes
    }

    /// Sets up boundary conditions.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let boundary_handler = BoundaryConditionHandler::new();
        for face in mesh.get_faces() {
            boundary_handler.set_bc(face.clone(), BoundaryCondition::Dirichlet(0.0));
        }
        boundary_handler
    }

    #[test]
    fn test_update_pressure_field_success() {
        // Create a mock `Fields` structure with a "pressure" scalar field
        let mut fields = Fields::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);

        // Initialize pressure field with some values
        let pressure_section = Section::new();
        pressure_section.set_data(cell1.clone(), Scalar(100.0));
        pressure_section.set_data(cell2.clone(), Scalar(120.0));
        fields.scalar_fields.insert("pressure".to_string(), pressure_section);

        // Create a pressure correction Section with some updates
        let pressure_correction = Section::new();
        pressure_correction.set_data(cell1.clone(), Scalar(-10.0));
        pressure_correction.set_data(cell2.clone(), Scalar(20.0));

        // Update pressure field using the correction
        let result = update_pressure_field(&mut fields, &pressure_correction);

        // Assertions
        assert!(result.is_ok(), "Expected update to succeed.");
        let updated_pressure_field = fields.scalar_fields.get("pressure").unwrap();

        // Verify the corrected pressure values
        assert_eq!(
            updated_pressure_field.restrict(&cell1).unwrap().0,
            90.0,
            "Cell 1 pressure should be updated."
        );
        assert_eq!(
            updated_pressure_field.restrict(&cell2).unwrap().0,
            140.0,
            "Cell 2 pressure should be updated."
        );
    }

    #[test]
    fn test_update_pressure_field_failure() {
        // Create a mock `Fields` structure without a "pressure" field
        let mut fields = Fields::new();

        // Create a pressure correction Section
        let pressure_correction = Section::new();
        let cell = MeshEntity::Cell(1);
        pressure_correction.set_data(cell, Scalar(-10.0));

        // Attempt to update pressure field
        let result = update_pressure_field(&mut fields, &pressure_correction);

        // Assertions
        assert!(result.is_err(), "Expected update to fail.");
        assert_eq!(
            result.unwrap_err(),
            "Pressure field not found in the fields structure.",
            "Error message should match."
        );
    }

    #[test]
    fn test_solve_pressure_poisson_success() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let fluxes = setup_fluxes(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Initialize the conjugate gradient solver
        let mut cg_solver = ConjugateGradient::new(1000, 1e-6);

        let result = solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &boundary_handler,
            &mut cg_solver,
        );

        assert!(result.is_ok(), "Pressure Poisson solver should succeed.");
        let result = result.unwrap();
        println!("Residual: {}", result.residual);
        assert!(result.residual < 1e-6, "Residual should be within tolerance.");
    }

    #[test]
    fn test_solve_pressure_poisson_missing_pressure_field() {
        let mesh = setup_simple_mesh();
        let mut fields = Fields::new(); // Missing pressure field
        let fluxes = setup_fluxes(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        let mut cg_solver = ConjugateGradient::new(1000, 1e-6);

        let result = solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &boundary_handler,
            &mut cg_solver,
        );

        assert!(result.is_err(), "Should fail due to missing pressure field.");
        assert!(result
            .unwrap_err()
            .contains("Pressure field not found in the fields structure."));
    }

    #[test]
    fn test_solve_pressure_poisson_solver_divergence() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let fluxes = setup_fluxes(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Initialize the solver with an excessively high tolerance to simulate divergence
        let mut cg_solver = ConjugateGradient::new(5, 1e-12); // Low max iterations to force divergence

        let result = solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &boundary_handler,
            &mut cg_solver,
        );

        assert!(result.is_err(), "Should fail due to solver divergence.");
        assert!(result
            .unwrap_err()
            .contains("Pressure Poisson solver failed"));
    }

    #[test]
    fn test_boundary_condition_enforcement() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let fluxes = setup_fluxes(&mesh);

        // Apply Neumann BC instead of Dirichlet
        let boundary_handler = BoundaryConditionHandler::new();
        for face in mesh.get_faces() {
            boundary_handler.set_bc(face.clone(), BoundaryCondition::Neumann(0.0));
        }

        let mut cg_solver = ConjugateGradient::new(1000, 1e-6);

        let result = solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &boundary_handler,
            &mut cg_solver,
        );

        assert!(
            result.is_ok(),
            "Solver should handle Neumann boundary conditions correctly."
        );
    }

    #[test]
    fn test_matrix_rhs_initialization() {
        let mesh = setup_simple_mesh();
        let entity_to_index_map = mesh.entity_to_index_map();
        let num_cells = mesh.get_cells().len();

        // Ensure all entities are mapped
        assert_eq!(entity_to_index_map.len(), num_cells);

        // Initialize matrix and RHS
        let _matrix = Mat::<f64>::zeros(num_cells, num_cells);
        let rhs = Section::<Scalar>::new();
        for cell in mesh.get_cells() {
            rhs.set_data(cell.clone(), Scalar(0.0));
        }

        assert_eq!(
            rhs.data.len(),
            num_cells,
            "RHS vector does not cover all cells."
        );
    }
}

#[cfg(test)]
mod correct_velocity_tests {
    use super::*;
    use crate::{
        domain::{mesh::Mesh, section::{Scalar, Vector3}},
        equation::fields::Fields,
        boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition},
        interface_adapters::domain_adapter::DomainBuilder,
    };

    /// Helper function to set up a simple mesh for testing.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(3, [0.0, 1.0, 0.0]).is_ok());
        assert!(builder.add_vertex(4, [0.0, 0.0, 1.0]).is_ok());
        assert!(builder.add_tetrahedron_cell(vec![1, 2, 3, 4]).is_ok());

        builder.build()
    }

    /// Helper function to set up fields with velocity and pressure.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();
        for cell in mesh.get_cells() {
            fields.set_vector_field_value("velocity", cell.clone(), Vector3([1.0, 1.0, 1.0]));
            fields.set_scalar_field_value("pressure", cell.clone(), Scalar(100.0));
        }
        fields
    }

    /// Helper function to set up a pressure correction section.
    fn setup_pressure_correction(mesh: &Mesh) -> Section<Scalar> {
        let pressure_correction = Section::new();
        for cell in mesh.get_cells() {
            pressure_correction.set_data(cell.clone(), Scalar(10.0));
        }
        pressure_correction
    }

    /// Helper function to set up boundary conditions.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let boundary_handler = BoundaryConditionHandler::new();
        for face in mesh.get_faces() {
            boundary_handler.set_bc(face.clone(), BoundaryCondition::Neumann(0.0));
        }
        boundary_handler
    }

    #[test]
    fn test_correct_velocity_field_success() {
        // Set up test components
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Correct velocity field
        let result = correct_velocity_field(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        // Assertions
        assert!(result.is_ok(), "Velocity correction should succeed.");

        // Verify updated velocity field values
        let velocity_field = fields.vector_fields.get("velocity").unwrap();
        for cell in mesh.get_cells() {
            let updated_velocity = velocity_field.restrict(&cell).unwrap();
            // Velocity should decrease due to pressure gradient correction
            assert!(updated_velocity.0[0] < 1.0);
            assert!(updated_velocity.0[1] < 1.0);
            assert!(updated_velocity.0[2] < 1.0);
        }
    }

    #[test]
    fn test_correct_velocity_field_missing_velocity_field() {
        // Set up test components without a velocity field
        let mesh = setup_simple_mesh();
        let mut fields = Fields::new(); // Empty fields, no velocity field
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Attempt to correct velocity field
        let result = correct_velocity_field(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        // Assertions
        assert!(result.is_err(), "Expected failure due to missing velocity field.");
        assert_eq!(
            result.unwrap_err(),
            "Velocity field not found in the fields structure.",
            "Error message should match."
        );
    }

    #[test]
    fn test_correct_velocity_field_gradient_failure() {
        // Set up test components
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Introduce invalid data to trigger gradient computation failure
        pressure_correction.set_data(MeshEntity::Cell(999), Scalar(10.0)); // Non-existent cell

        // Attempt to correct velocity field
        let result = correct_velocity_field(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        // Assertions
        assert!(result.is_err(), "Expected failure due to gradient computation issue.");
        assert!(result.unwrap_err().contains("Gradient computation failed"), "Error should indicate gradient failure.");
    }
}