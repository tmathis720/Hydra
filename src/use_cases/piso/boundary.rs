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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::{mesh::Mesh, section::Scalar},
        geometry::Geometry,
        interface_adapters::domain_adapter::DomainBuilder,
        FaceShape, Section,
    };
    use faer::Mat;

    /// Creates a simple 2x2x1 hexahedral grid mesh.
    fn setup_hexahedral_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();
        for z in 0..=1 {
            for y in 0..=2 {
                for x in 0..=2 {
                    builder.add_vertex(
                        (z * 9 + y * 3 + x) as usize,
                        [x as f64, y as f64, z as f64],
                    );
                }
            }
        }

        for z in 0..1 {
            for y in 0..2 {
                for x in 0..2 {
                    let base = z * 9 + y * 3 + x;
                    builder.add_hexahedron_cell(vec![
                        base as usize,
                        (base + 1) as usize,
                        (base + 4) as usize,
                        (base + 3) as usize,
                        (base + 9) as usize,
                        (base + 10) as usize,
                        (base + 13) as usize,
                        (base + 12) as usize,
                    ]);
                }
            }
        }

        builder.build()
    }

    /// Sets up boundary conditions for the 2x2x1 hexahedral mesh.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();
        let geometry = Geometry::new();

        for face in mesh.get_faces() {
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };

            let centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
            if centroid[0] == 0.0 {
                bc_handler.set_bc(face, BoundaryCondition::Dirichlet(10.0));
            } else if centroid[0] == 2.0 {
                bc_handler.set_bc(face, BoundaryCondition::Neumann(5.0));
            }
        }

        bc_handler
    }

    /// Initializes the system matrix and RHS for the test.
    fn setup_matrix_and_rhs(mesh: &Mesh) -> (Mat<f64>, Section<Scalar>) {
        let total_entities = mesh.get_faces().len();
        let matrix = Mat::<f64>::zeros(total_entities, total_entities);

        let rhs = Section::new();
        for entity in mesh.get_faces().iter() {
            rhs.set_data(*entity, Scalar(0.0));
        }

        (matrix, rhs)
    }

    #[test]
    fn test_apply_pressure_poisson_bc() {
        let mesh = setup_hexahedral_mesh();
        let bc_handler = setup_boundary_conditions(&mesh);
        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(result.is_ok(), "Boundary condition application failed.");

        let geometry = Geometry::new();
        for face in mesh.get_faces() {
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };

            let centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
            if centroid[0] == 0.0 {
                let entity_to_index_map = mesh.entity_to_index_map();
                let face_index = entity_to_index_map.get(&face).unwrap();
                assert_eq!(matrix_mut.read(*face_index, *face_index), 1.0);
                assert_eq!(rhs.data.get(&face).unwrap().0, 10.0);
            } else if centroid[0] == 2.0 {
                let entity_to_index_map = mesh.entity_to_index_map();
                let _face_index = entity_to_index_map.get(&face).unwrap();
                let rhs_value = rhs.data.get(&face).unwrap().0;
                assert_eq!(rhs_value, 5.0);
            }
        }
    }
}
