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
    // 1) Gather the boundary faces from the BC handler:
    let boundary_faces = boundary_handler.get_boundary_faces();

    // 2) Get the complete entity-to-index map (cells + faces):
    let entity_to_index = mesh.entity_to_index_map();

    // 3) Convert the Section<Scalar> into a Faer matrix row-vector:
    //    Instead of only using cells, we gather all cells + faces. 
    let all_entities = mesh
        .get_cells()
        .into_iter()
        .chain(mesh.get_faces())
        .collect::<Vec<_>>();
    
    // Convert the Section into a (total_entities x 1) Faer matrix:
    let mut rhs_mat = SectionMatVecAdapter::section_to_matmut(rhs);

    // 4) Apply boundary conditions:
    for face in boundary_faces {
        if let Some(boundary_condition) = boundary_handler.get_bc(&face) {
            boundary_condition.apply(
                &face,
                &mut rhs_mat.as_mut(),
                matrix,
                &entity_to_index,
                0.0,
            );
        }
    }

    // 5) Convert the updated Faer vector back to a Section<Scalar>.
    //    We must pass the same list of entities (cells + faces).
    let updated_section = SectionMatVecAdapter::matmut_to_section(&rhs_mat, &all_entities);

    // 6) Overwrite the original RHS:
    *rhs = updated_section;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::{mesh::Mesh, section::Scalar},
        interface_adapters::domain_adapter::DomainBuilder,
        Section,
    };
    use faer::Mat;

    /// Builds a single tetrahedron: 1 cell, 4 faces.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 0.0, 1.0]);

        builder.add_tetrahedron_cell(vec![1, 2, 3, 4]);
        let mesh = builder.build();

        // Validate we have 1 cell + 4 faces => total 5 entities
        assert_eq!(mesh.get_cells().len(), 1, "Should have 1 tetrahedron cell");
        assert_eq!(mesh.get_faces().len(), 4, "Should have 4 triangular faces");
        mesh
    }

    /// Attaches a Dirichlet BC to the first face, Neumann to the second (if any).
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();

        // Face[0] => Dirichlet(10.0)
        bc_handler.set_bc(faces[0], BoundaryCondition::Dirichlet(10.0));

        // Face[1] => Neumann(5.0) if there's a second face
        if faces.len() > 1 {
            bc_handler.set_bc(faces[1], BoundaryCondition::Neumann(5.0));
        }

        bc_handler
    }

    /// Sets up the system matrix (5x5) and an RHS Section with 5 entries (1 cell + 4 faces).
    fn setup_matrix_and_rhs(mesh: &Mesh) -> (Mat<f64>, Section<Scalar>) {
        let num_cells = mesh.get_cells().len();
        let num_faces = mesh.get_faces().len();
        let total_entities = num_cells + num_faces;
        assert_eq!(total_entities, 5, "Tetrahedron => 5 total (1 cell + 4 faces)");

        // 1) Create a 5x5 zero matrix
        let matrix = Mat::<f64>::zeros(total_entities, total_entities);

        // 2) Create a Section with 5 zero entries
        let rhs = Section::new();
        for entity in mesh.get_cells() {
            rhs.set_data(entity, Scalar(0.0));
        }
        for entity in mesh.get_faces() {
            rhs.set_data(entity, Scalar(0.0));
        }

        (matrix, rhs)
    }

    // ================ TESTS ================= //

    #[test]
    fn test_apply_pressure_poisson_bc_dirichlet() {
        let mesh = setup_simple_mesh();
        let bc_handler = setup_boundary_conditions(&mesh);

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        // Apply BC
        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(result.is_ok(), "Expected OK applying Dirichlet BC.");

        // Face[0] => Dirichlet(10.0)
        let face0 = mesh.get_faces()[0];
        let e2i = mesh.entity_to_index_map();
        let row = *e2i.get(&face0).unwrap(); // Index for face0

        // Check matrix diagonal => 1.0
        assert_eq!(matrix_mut.read(row, row), 1.0, "Dirichlet sets diagonal to 1.0.");

        // Check RHS => 10.0
        let rhs_val = rhs.data.get(&face0).unwrap().0;
        assert_eq!(rhs_val, 10.0, "Dirichlet sets RHS to boundary value (10.0).");
    }

    #[test]
    fn test_apply_pressure_poisson_bc_neumann() {
        let mesh = setup_simple_mesh();
        let bc_handler = BoundaryConditionHandler::new();
        // Attach Neumann(5.0) to the second face
        let faces = mesh.get_faces();
        assert!(faces.len() > 1, "Tetrahedron => at least 4 faces");
        bc_handler.set_bc(faces[1], BoundaryCondition::Neumann(5.0));

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(result.is_ok(), "Expected OK applying Neumann BC.");

        // Face[1] => Neumann(5.0)
        let face1 = faces[1];
        let e2i = mesh.entity_to_index_map();
        let _row = *e2i.get(&face1).unwrap();

        // For a typical Neumann BC, we expect the matrix row to remain mostly unchanged
        // but the RHS to be incremented by 5.0
        let rhs_val = rhs.data.get(&face1).unwrap().0;
        assert_eq!(rhs_val, 5.0, "Neumann increments RHS by 5.0.");
    }

    #[test]
    fn test_apply_pressure_poisson_bc_multiple_conditions() {
        let mesh = setup_simple_mesh();
        let bc_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();

        // Face[0] => Dirichlet(15.0)
        // Face[1] => Neumann(3.0)
        assert!(faces.len() > 2, "Tetrahedron => 4 faces => must be >2");
        bc_handler.set_bc(faces[0], BoundaryCondition::Dirichlet(15.0));
        bc_handler.set_bc(faces[1], BoundaryCondition::Neumann(3.0));

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(result.is_ok(), "Expected OK applying multiple BCs.");

        // Check Dirichlet(15.0) on face[0]
        let face0 = faces[0];
        let e2i = mesh.entity_to_index_map();
        let i0 = *e2i.get(&face0).unwrap();
        assert_eq!(matrix_mut.read(i0, i0), 1.0, "Dirichlet diagonal => 1.0");
        assert_eq!(rhs.data.get(&face0).unwrap().0, 15.0, "Dirichlet => RHS=15.0");

        // Check Neumann(3.0) on face[1]
        let face1 = faces[1];
        let rhs_val1 = rhs.data.get(&face1).unwrap().0;
        assert_eq!(rhs_val1, 3.0, "Neumann => RHS += 3.0");
    }

    #[test]
    fn test_invalid_boundary_conditions() {
        let mesh = setup_simple_mesh();
        let bc_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();

        // Suppose alpha=0, beta=0 is invalid for Robin
        bc_handler.set_bc(
            faces[0],
            BoundaryCondition::Robin { alpha: 0.0, beta: 0.0 },
        );

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        // Expect an error:
        let result = apply_pressure_poisson_bc(&mesh, &bc_handler, &mut matrix_mut, &mut rhs);
        assert!(
            result.is_err(),
            "Expected error for invalid Robin(0.0,0.0) boundary condition"
        );
    }
}
