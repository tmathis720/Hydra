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
<<<<<<< HEAD
}
=======
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler}, 
        domain::{mesh::Mesh, section::Scalar}, 
        geometry::Geometry, 
        interface_adapters::domain_adapter::DomainBuilder, 
        FaceShape, Section
    };
    use faer::Mat;

    /// Creates a 2x2x1 hexahedral grid mesh.
    fn setup_hexahedral_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        // Generate vertices for a 2x2x1 grid
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

        // Add cells as hexahedrons (8 vertices per cell)
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

        let mesh = builder.build();

        // Validate that we have 4 cells
        assert_eq!(mesh.get_cells().len(), 4, "Should have 4 hexahedral cells");
        assert!(mesh.get_faces().len() > 0, "Mesh should have faces for boundary handling");
        mesh
    }

    /// Sets up boundary conditions for the 2x2x1 hexahedral mesh.
    fn setup_boundary_conditions(mesh: &Mesh, geometry: &mut Geometry) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();

        // Apply Dirichlet to all faces on x=0 and Neumann on x=2
        for face in mesh.get_faces() {
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                2 => FaceShape::Edge,
                3 => FaceShape::Triangle,
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

    /// Initializes the system matrix and RHS for the 2x2x1 hexahedral grid.
    fn setup_matrix_and_rhs(mesh: &Mesh) -> (Mat<f64>, Section<Scalar>) {
        let total_entities = mesh.get_cells().len() + mesh.get_faces().len();

        // Check total entities is non-zero
        assert!(
            total_entities > 0,
            "Mesh must have cells and faces to initialize the matrix."
        );

        // Create a square matrix with size equal to total entities
        let matrix = Mat::<f64>::zeros(total_entities, total_entities);

        // Create an RHS section with zero values
        let rhs = Section::new();
        for entity in mesh.get_cells().iter().chain(mesh.get_faces().iter()) {
            rhs.set_data(*entity, Scalar(0.0));
        }

        (matrix, rhs)
    }

    #[test]
    fn test_apply_pressure_poisson_bc_small_mesh() {
        let mesh = setup_hexahedral_mesh();
        let mut geometry = Geometry::new();

        let bc_handler = setup_boundary_conditions(&mesh, &mut geometry);

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let entity_to_index_map = mesh.entity_to_index_map();
        for entry in entity_to_index_map.iter() {
            let (entity, index) = entry.pair();
            println!("Entity {:?} mapped to index {}", entity, index);
        }
        let mut matrix_mut = matrix.as_mut();

        // Apply boundary conditions
        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(result.is_ok(), "Expected successful boundary condition application.");

        for entry in rhs.data.iter() {
            let (entity, scalar) = entry.pair();
            println!("Entity {:?}: RHS value {}", entity, scalar.0);
        }
        // Verify Dirichlet conditions
        for face in mesh.get_faces() {
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                2 => FaceShape::Edge,
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };

            let centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
            if centroid[0] == 0.0 {
                let index = mesh.entity_to_index_map().get(&face)
                .expect("Face index not found in entity-to-index map")
                .clone();
                assert_eq!(matrix_mut.read(index, index), 1.0, "Dirichlet diagonal should be 1.0");
                assert_eq!(
                    rhs.data.get(&face).unwrap().0,
                    10.0,
                    "Dirichlet RHS value should match boundary condition"
                );
            }
        }

        // Verify Neumann conditions
        for face in mesh.get_faces() {
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                2 => FaceShape::Edge,
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };

            let centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
            if centroid[0] == 2.0 {
                let rhs_val = rhs.data.get(&face).unwrap().0;
                assert_eq!(rhs_val, 5.0, "Neumann boundary should increment RHS correctly");
            }
        }
    }
}
>>>>>>> parent of 45edab1 (Enhance section_to_matmut function with index validation and add unit tests for conversion and error handling)
