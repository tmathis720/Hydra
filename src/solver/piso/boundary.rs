use crate::{
    boundary::bc_handler::{BoundaryConditionApply, BoundaryConditionHandler}, domain::{mesh::Mesh, section::Scalar}, interface_adapters::section_matvec_adapter::SectionMatVecAdapter, MeshEntity, Section
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

    // Extract entities from the mesh (or use a pre-computed list of entities)
    let entities: Vec<MeshEntity> = mesh.get_cells(); // Replace this with the correct function for entity extraction

    // Convert the matrix back to a section
    let updated_section = SectionMatVecAdapter::matmut_to_section(&rhs_mat, &entities);

    // Update the rhs section with the converted data
    *rhs = updated_section;

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::mesh::Mesh,
        domain::section::Scalar,
        interface_adapters::domain_adapter::DomainBuilder,
        Section,
    };
    use faer::Mat;

    /// Helper function to create a simple mesh for testing.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        // Define a simple tetrahedron mesh
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 0.0, 1.0]);

        builder.add_tetrahedron_cell(vec![1, 2, 3, 4]);

        builder.build()
    }

    /// Helper function to set up boundary conditions.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let handler = BoundaryConditionHandler::new();

        let faces = mesh.get_faces();
        assert!(
            !faces.is_empty(),
            "Mesh should contain at least one face for boundary conditions."
        );

        // Apply Dirichlet boundary conditions to the first face
        handler.set_bc(faces[0].clone(), BoundaryCondition::Dirichlet(10.0));

        // Apply Neumann boundary conditions to the second face (if exists)
        if faces.len() > 1 {
            handler.set_bc(faces[1].clone(), BoundaryCondition::Neumann(5.0));
        }

        handler
    }

    /// Initializes the matrix and RHS for testing.
    fn setup_matrix_and_rhs(mesh: &Mesh) -> (Mat<f64>, Section<Scalar>) {
        let total_entities = mesh.get_cells().len() + mesh.get_faces().len();
        let matrix = Mat::<f64>::zeros(total_entities, total_entities);

        let rhs = Section::new();
        let all_entities = mesh
            .get_cells()
            .into_iter()
            .chain(mesh.get_faces())
            .collect::<Vec<_>>();
        for entity in &all_entities {
            rhs.set_data(*entity, Scalar(0.0)); // Initialize RHS to zero for all entities
        }

        (matrix, rhs)
    }

    #[test]
    fn test_apply_pressure_poisson_bc_dirichlet() {
        let mesh = setup_simple_mesh();
        let boundary_handler = setup_boundary_conditions(&mesh);

        // Initialize matrix and RHS
        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        // Apply boundary conditions
        let result = apply_pressure_poisson_bc(
            &mesh,
            &boundary_handler,
            &mut matrix_mut,
            &mut rhs,
        );

        // Assert no errors occurred
        assert!(result.is_ok(), "Failed to apply boundary conditions");

        // Verify that Dirichlet boundary condition was applied
        let face = mesh.get_faces()[0];
        let entity_to_index = mesh.entity_to_index_map();
        let index = entity_to_index.get(&face).unwrap();
        assert_eq!(matrix_mut.read(*index, *index), 1.0, "Diagonal should be 1.0");
        assert_eq!(rhs.data.get(&face).unwrap().0, 10.0, "RHS should be set to boundary value");
    }

    #[test]
    fn test_apply_pressure_poisson_bc_neumann() {
        let mesh = setup_simple_mesh();
        let boundary_handler = BoundaryConditionHandler::new();

        let faces = mesh.get_faces();
        assert!(
            faces.len() > 1,
            "Mesh should contain at least two faces for Neumann BC test."
        );

        // Apply Neumann boundary condition to the second face
        boundary_handler.set_bc(faces[1].clone(), BoundaryCondition::Neumann(5.0));

        // Initialize matrix and RHS
        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        // Apply boundary conditions
        let result = apply_pressure_poisson_bc(
            &mesh,
            &boundary_handler,
            &mut matrix_mut,
            &mut rhs,
        );

        // Assert no errors occurred
        assert!(result.is_ok(), "Failed to apply boundary conditions");

        // Verify that Neumann boundary condition was applied
        let face = faces[1].clone();
        let rhs_value = rhs.data.get(&face).unwrap().0;
        assert_eq!(
            rhs_value, 5.0,
            "RHS should be incremented by Neumann value."
        );
    }

    #[test]
    fn test_apply_pressure_poisson_bc_multiple_conditions() {
        let mesh = setup_simple_mesh();
        let boundary_handler = BoundaryConditionHandler::new();

        let faces = mesh.get_faces();
        assert!(
            faces.len() > 2,
            "Mesh should contain at least three faces for multiple BC test."
        );

        // Apply Dirichlet to the first face
        boundary_handler.set_bc(faces[0].clone(), BoundaryCondition::Dirichlet(15.0));
        // Apply Neumann to the second face
        boundary_handler.set_bc(faces[1].clone(), BoundaryCondition::Neumann(3.0));

        // Initialize matrix and RHS
        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        // Apply boundary conditions
        let result = apply_pressure_poisson_bc(
            &mesh,
            &boundary_handler,
            &mut matrix_mut,
            &mut rhs,
        );

        // Assert no errors occurred
        assert!(result.is_ok(), "Failed to apply boundary conditions");

        // Verify Dirichlet condition
        let face0 = faces[0].clone();
        let entity_to_index = mesh.entity_to_index_map();
        let index0 = entity_to_index.get(&face0).unwrap();
        assert_eq!(matrix_mut.read(*index0, *index0), 1.0);
        assert_eq!(rhs.data.get(&face0).unwrap().0, 15.0);

        // Verify Neumann condition
        let face1 = faces[1].clone();
        let rhs_value = rhs.data.get(&face1).unwrap().0;
        assert_eq!(rhs_value, 3.0);
    }

    #[test]
    fn test_invalid_boundary_conditions() {
        let mesh = setup_simple_mesh();
        let boundary_handler = BoundaryConditionHandler::new();

        let faces = mesh.get_faces();
        boundary_handler.set_bc(
            faces[0].clone(),
            BoundaryCondition::Robin { alpha: 0.0, beta: 0.0 },
        );

        let (mut matrix, mut rhs) = setup_matrix_and_rhs(&mesh);
        let mut matrix_mut = matrix.as_mut();

        let result = apply_pressure_poisson_bc(
            &mesh,
            &boundary_handler,
            &mut matrix_mut,
            &mut rhs,
        );

        assert!(result.is_err(), "Expected failure for invalid BC parameters");
    }
}
