use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    domain::{mesh::Mesh, section::Scalar},
    interface_adapters::section_matvec_adapter::SectionMatVecAdapter,
    Section,
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
    let mut rhs_mat =
        SectionMatVecAdapter::section_to_matmut(rhs, &mesh.entity_to_index_map(), matrix.nrows());

    for face in boundary_handler.get_boundary_faces() {
        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            let entity_to_index_map = mesh.entity_to_index_map();
            let face_index = entity_to_index_map.get(&face).ok_or_else(|| {
                format!(
                    "Face {:?} not found in entity-to-index map.",
                    face
                )
            })?;

            if *face_index >= matrix.nrows() {
                return Err(format!(
                    "Face index {:?} exceeds matrix dimensions (nrows = {}).",
                    face_index, matrix.nrows()
                ));
            }

            match boundary_condition {
                BoundaryCondition::Dirichlet(value) => {
                    matrix.write(*face_index, *face_index, 1.0);
                    for col in 0..matrix.ncols() {
                        if col != *face_index {
                            matrix.write(*face_index, col, 0.0);
                        }
                    }
                    rhs_mat.write(*face_index, 0, value);
                }
                BoundaryCondition::Neumann(value) => {
                    let current_value = rhs_mat.read(*face_index, 0);
                    rhs_mat.write(*face_index, 0, current_value + value);
                }
                _ => return Err("Unsupported boundary condition type.".to_string()),
            }
        }
    }

    Ok(())
}