use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;
use crate::extrusion::use_cases::extrude_mesh::ExtrudeMeshUseCase;
use crate::domain::mesh::Mesh;

/// `ExtrusionService` serves as the main interface for extruding a 2D mesh into a 3D mesh.
/// It supports both quadrilateral and triangular meshes, leveraging the mesh's type to determine
/// the appropriate extrusion method (hexahedral or prismatic).
pub struct ExtrusionService;

impl ExtrusionService {
    /// Extrudes a 2D mesh into a 3D mesh, determining the mesh type (quad or triangle) and
    /// extruding it accordingly.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a 2D mesh that implements the `ExtrudableMesh` trait, indicating
    ///   the mesh supports extrusion operations.
    /// - `depth`: The extrusion depth, specifying the total height of the extruded 3D mesh.
    /// - `layers`: The number of layers into which the extrusion is divided.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok` with the extruded 3D `Mesh` on success, or an
    ///   error message `String` if extrusion fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the mesh type is unsupported or the downcasting to a specific mesh type fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let quad_mesh = QuadrilateralMesh::new(...);
    /// let depth = 5.0;
    /// let layers = 3;
    /// let result = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);
    /// ```
    pub fn extrude_mesh(mesh: &dyn ExtrudableMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if mesh.is_quad_mesh() {
            let quad_mesh = mesh.as_quad().ok_or("Failed to downcast to QuadrilateralMesh")?;
            ExtrudeMeshUseCase::extrude_to_hexahedron(quad_mesh, depth, layers)
        } else if mesh.is_tri_mesh() {
            let tri_mesh = mesh.as_tri().ok_or("Failed to downcast to TriangularMesh")?;
            ExtrudeMeshUseCase::extrude_to_prism(tri_mesh, depth, layers)
        } else {
            Err("Unsupported mesh type".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExtrusionService;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Validates the extrusion of a quadrilateral mesh into a hexahedral mesh.
    fn test_extrude_quad_mesh_to_hexahedron() {
        // Create a simple quadrilateral mesh
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        let depth = 5.0;
        let layers = 3;

        // Perform extrusion
        let extruded_result = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);

        // Check that extrusion is successful and returns a valid Mesh
        assert!(extruded_result.is_ok(), "Extrusion should succeed for quadrilateral mesh");
        let extruded_mesh = extruded_result.unwrap();
        assert!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)) > 0, 
            "Extruded mesh should contain hexahedral cells");
    }

    #[test]
    /// Validates the extrusion of a triangular mesh into a prismatic mesh.
    fn test_extrude_tri_mesh_to_prism() {
        // Create a simple triangular mesh
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        let depth = 4.0;
        let layers = 2;

        // Perform extrusion
        let extruded_result = ExtrusionService::extrude_mesh(&tri_mesh, depth, layers);

        // Check that extrusion is successful and returns a valid Mesh
        assert!(extruded_result.is_ok(), "Extrusion should succeed for triangular mesh");
        let extruded_mesh = extruded_result.unwrap();
        assert!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)) > 0, 
            "Extruded mesh should contain prismatic cells");
    }

    #[test]
    /// Tests that attempting to extrude an unsupported mesh type returns an error.
    fn test_unsupported_mesh_type() {
        #[derive(Debug)]
        struct UnsupportedMesh;
        impl ExtrudableMesh for UnsupportedMesh {
            fn is_valid_for_extrusion(&self) -> bool { false }
            fn get_vertices(&self) -> Vec<[f64; 3]> { vec![] }
            fn get_cells(&self) -> Vec<Vec<usize>> { vec![] }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }

        let unsupported_mesh = UnsupportedMesh;
        let depth = 5.0;
        let layers = 3;

        // Attempt extrusion and expect an error
        let extruded_result = ExtrusionService::extrude_mesh(&unsupported_mesh, depth, layers);
        assert!(extruded_result.is_err(), "Extrusion should fail for unsupported mesh type");
        assert_eq!(extruded_result.unwrap_err(), "Unsupported mesh type", "Error message should indicate unsupported mesh type");
    }
}
