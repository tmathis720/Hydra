use crate::{
    boundary::bc_handler::{BoundaryConditionApply, BoundaryConditionHandler}, domain::{mesh::Mesh, section::Scalar}, interface_adapters::section_matvec_adapter::SectionMatVecAdapter, Section
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
    let boundary_faces = boundary_handler.get_boundary_faces();
    let entity_to_index = mesh.entity_to_index_map();

    // Convert rhs Section<Scalar> into MatMut<f64> for faer compatibility
    let mut rhs_mat = SectionMatVecAdapter::section_to_matmut(rhs);

    for face in boundary_faces {
        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            boundary_condition.apply(&face, &mut rhs_mat.as_mut(), matrix, &entity_to_index, 0.0);
        }
    }

    // Update back the values into Section<Scalar> after processing
    //SectionMatVecAdapter::matmut_to_section(&rhs_mat, rhs);

    Ok(())
}
