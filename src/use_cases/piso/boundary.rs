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

    // Convert RHS Section into MatMut
    let mut rhs_mat = SectionMatVecAdapter::section_to_matmut(&rhs, &mesh.entity_to_index_map(), matrix.nrows());
    assert_eq!(
        rhs_mat.nrows(),
        rhs.data.len(),
        "RHS matrix dimensions do not match section size"
    );

    for face in boundary_handler.get_boundary_faces() {
        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            /* println!("Applying boundary condition to face {:?}", face); */
            let entity_to_index_map = mesh.entity_to_index_map();
            let face_index = entity_to_index_map.get(&face)
                .ok_or_else(|| format!("Face {:?} not found in entity-to-index map", face))?;

            if *face_index >= matrix.nrows() {
                return Err(format!(
                    "Face index {:?} exceeds matrix dimensions (nrows = {})",
                    face_index, matrix.nrows()
                ));
            }

            boundary_condition.apply(
                &face,
                &mut rhs_mat.as_mut(),
                matrix,
                &mesh.entity_to_index_map(),
                0.0,
            );
        }
    }
    

    Ok(())
}
