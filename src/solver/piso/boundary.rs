use crate::{
    boundary::bc_handler::{BoundaryConditionApply, BoundaryConditionHandler},
    domain::{mesh::Mesh, section::Scalar},
    linalg::matrix::Matrix,
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
pub fn apply_pressure_poisson_bc<T: Matrix>(
    mesh: &Mesh,
    boundary_handler: &BoundaryConditionHandler,
    matrix: &mut T,
    rhs: &mut Section<Scalar>,
) -> Result<(), String> {
    let boundary_faces = boundary_handler.get_boundary_faces();

    // Map mesh entities to indices for use in matrix/RHS operations
    let entity_to_index = mesh.entity_to_index_map();

    for face in boundary_faces {
        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            // Use the existing `BoundaryConditionApply` trait for all supported conditions
            boundary_condition.apply(&face, rhs, matrix, &entity_to_index, 0.0);
        }
    }

    Ok(())
}
