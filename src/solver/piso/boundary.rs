use crate::{
    boundary::{
        bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        dirichlet::DirichletBC,
        neumann::NeumannBC,
        robin::RobinBC,
    }, domain::{mesh::Mesh, section::{Scalar, Vector3}}, linalg::matrix::Matrix, MeshEntity, Section
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

    for face in boundary_faces {
        if let Some(bc) = boundary_handler.get_bc(&face) {
            match bc {
                BoundaryCondition::Dirichlet(value) => {
                    apply_dirichlet_bc(mesh, &face, matrix, rhs, value)?;
                }
                BoundaryCondition::Neumann(flux) => {
                    apply_neumann_bc(mesh, &face, rhs, flux)?;
                }
                BoundaryCondition::Robin { alpha, beta } => {
                    apply_robin_bc(mesh, &face, matrix, rhs, alpha, beta)?;
                }
                _ => {
                    return Err(format!(
                        "Unsupported boundary condition type for pressure Poisson equation: {:?}",
                        bc
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Applies Dirichlet boundary conditions.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `face`: The mesh entity corresponding to the boundary face.
/// - `matrix`: The sparse matrix of the pressure Poisson system.
/// - `rhs`: The right-hand side vector.
/// - `value`: The fixed Dirichlet value.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message if it fails.
fn apply_dirichlet_bc<T: Matrix>(
    mesh: &Mesh,
    face: &MeshEntity,
    matrix: &mut T,
    rhs: &mut Section<Scalar>,
    value: f64,
) -> Result<(), String> {
    let dirichlet_bc = DirichletBC::new();
    dirichlet_bc.apply_to_matrix_and_rhs(mesh, face, matrix, rhs, value)
}

/// Applies Neumann boundary conditions.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `face`: The mesh entity corresponding to the boundary face.
/// - `rhs`: The right-hand side vector.
/// - `flux`: The fixed Neumann flux value.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message if it fails.
fn apply_neumann_bc(
    mesh: &Mesh,
    face: &MeshEntity,
    rhs: &mut Section<Scalar>,
    flux: f64,
) -> Result<(), String> {
    let neumann_bc = NeumannBC::new();
    neumann_bc.apply_to_rhs(mesh, face, rhs, flux)
}

/// Applies Robin boundary conditions.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `face`: The mesh entity corresponding to the boundary face.
/// - `matrix`: The sparse matrix of the pressure Poisson system.
/// - `rhs`: The right-hand side vector.
/// - `alpha`: The Robin coefficient for the Dirichlet-like term.
/// - `beta`: The Robin coefficient for the Neumann-like term.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` on success or an error message if it fails.
fn apply_robin_bc<T: Matrix>(
    mesh: &Mesh,
    face: &MeshEntity,
    matrix: &mut T,
    rhs: &mut Section<Scalar>,
    alpha: f64,
    beta: f64,
) -> Result<(), String> {
    let robin_bc = RobinBC::new();
    robin_bc.apply_to_matrix_and_rhs(mesh, face, matrix, rhs, alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::{mesh::Mesh, mesh_entity::MeshEntity},
        linalg::matrix::Matrix,
        domain::section::Section,
    };

    #[test]
    fn test_apply_pressure_poisson_bc_dirichlet() {
        // Setup
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let face = MeshEntity::Face(1);
        boundary_handler.set_bc(face.clone(), BoundaryCondition::Dirichlet(10.0));

        let mut matrix = Matrix::new();
        let mut rhs = Section::<Scalar>::new();

        // Apply boundary condition
        let result = apply_pressure_poisson_bc(&mesh, &boundary_handler, &mut matrix, &mut rhs);

        // Validate
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_pressure_poisson_bc_neumann() {
        // Setup
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let face = MeshEntity::Face(1);
        boundary_handler.set_bc(face.clone(), BoundaryCondition::Neumann(5.0));

        let mut matrix = <dyn Matrix>::new();
        let mut rhs = Section::<Scalar>::new();

        // Apply boundary condition
        let result = apply_pressure_poisson_bc(&mesh, &boundary_handler, &mut matrix, &mut rhs);

        // Validate
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_pressure_poisson_bc_robin() {
        // Setup
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let face = MeshEntity::Face(1);
        boundary_handler.set_bc(
            face.clone(),
            BoundaryCondition::Robin {
                alpha: 2.0,
                beta: 3.0,
            },
        );

        let mut matrix = Matrix::new();
        let mut rhs = Section::<Scalar>::new();

        // Apply boundary condition
        let result = apply_pressure_poisson_bc(&mesh, &boundary_handler, &mut matrix, &mut rhs);

        // Validate
        assert!(result.is_ok());
    }
}
