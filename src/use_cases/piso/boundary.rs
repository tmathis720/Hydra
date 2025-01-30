use crate::{
    boundary::bc_handler::{BoundaryConditionApply, 
        BoundaryConditionHandler}, 
        domain::{mesh::Mesh, section::Scalar}, 
        interface_adapters::section_matvec_adapter::SectionMatVecAdapter, 
        Section
};

/// Applies boundary conditions to the pressure Poisson equation.
///
/// This function modifies the matrix and RHS vector to enforce boundary conditions
/// during the pressure correction step of the PISO algorithm.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `boundary_handler`: Handles the boundary conditions for the domain.
/// - `matrix`: The sparse matrix representing the pressure Poisson system.
/// - `rhs`: The right-hand side vector for the pressure correction system.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message if boundary conditions cannot be applied.
pub fn apply_pressure_poisson_bc(
    mesh: &Mesh,
    boundary_handler: &BoundaryConditionHandler,
    matrix: &mut faer::MatMut<'_, f64>,
    rhs: &mut Section<Scalar>,
) -> Result<(), String> {
    println!("Starting application of boundary conditions for the pressure Poisson equation.");

    // Convert RHS Section into MatMut
    println!("Converting RHS Section into MatMut.");
    let mut rhs_mat = SectionMatVecAdapter::section_to_matmut(&rhs, &mesh.entity_to_index_map(), matrix.nrows());
    assert_eq!(
        rhs_mat.nrows(),
        rhs.data.len(),
        "RHS matrix dimensions do not match section size."
    );
    println!("RHS conversion complete. Number of rows: {}", rhs_mat.nrows());

    // Loop through boundary faces and apply conditions
    println!("Applying boundary conditions to {} faces.", boundary_handler.get_boundary_faces().len());
    for face in boundary_handler.get_boundary_faces() {
        println!("Processing boundary face: {:?}", face);

        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            let entity_to_index_map = mesh.entity_to_index_map();
            let face_index = entity_to_index_map.get(&face)
                .ok_or_else(|| {
                    let msg = format!("Face {:?} not found in entity-to-index map", face);
                    println!("{}", msg);
                    msg
                })?;

            println!(
                "Face index for boundary face {:?}: {:?}. Matrix rows: {}",
                face, face_index, matrix.nrows()
            );

            if *face_index >= matrix.nrows() {
                let error_msg = format!(
                    "Error: Face index {:?} exceeds matrix dimensions (nrows = {}).",
                    face_index, matrix.nrows()
                );
                println!("{}", error_msg);
                return Err(error_msg);
            }

            println!(
                "Applying boundary condition to face {:?} with index {:?}.",
                face, face_index
            );
            boundary_condition.apply(
                &face,
                &mut rhs_mat.as_mut(),
                matrix,
                &mesh.entity_to_index_map(),
                0.0,
            )
            .map_err(|err| {
                let msg = format!("Error applying boundary condition to face {:?}: {:?}", face, err);
                println!("{}", msg);
                msg
            })?;
            println!("Boundary condition applied successfully to face {:?}.", face);
        } else {
            println!("No boundary condition found for face {:?}.", face);
        }
    }

    println!("Boundary conditions applied successfully to all faces.");
    Ok(())
}
