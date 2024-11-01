use super::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use std::fmt::Debug;

/// The `ExtrudableMesh` trait defines the required methods for a 2D mesh that is capable of extrusion,
/// transforming it from a 2D to a 3D representation. This trait supports handling different mesh types,
/// specifically quadrilateral and triangular meshes, and provides methods for downcasting to specific types.
///
/// Implementations of this trait should ensure that the mesh is compatible with extrusion (e.g., only quads or triangles),
/// and provide vertices and cell connectivity for the mesh in a 3D extrusion context.
pub trait ExtrudableMesh: Debug {
    /// Checks if the mesh is valid for extrusion by ensuring all cells adhere to the expected type
    /// (e.g., all cells are quads or all are triangles).
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is valid for extrusion, `false` otherwise.
    fn is_valid_for_extrusion(&self) -> bool;

    /// Returns a list of vertices in the mesh, with each vertex formatted as a 3D coordinate.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the vertices of the 2D mesh.
    fn get_vertices(&self) -> Vec<[f64; 3]>;

    /// Returns the cell-to-vertex connectivity of the mesh, where each cell is represented by indices
    /// that refer to vertices in the `get_vertices` array.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, each of which is a list of vertex indices.
    fn get_cells(&self) -> Vec<Vec<usize>>;

    /// Determines if this mesh is a quadrilateral mesh.
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is of type `QuadrilateralMesh`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::{extrudable_mesh::ExtrudableMesh, hexahedral_mesh::QuadrilateralMesh};
    ///
    /// let some_mesh = QuadrilateralMesh::new(
    ///     vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ///     vec![vec![0, 1, 2, 3]],
    /// );
    ///
    /// assert!(some_mesh.is_quad_mesh());
    /// ```
    fn is_quad_mesh(&self) -> bool {
        self.as_any().is::<QuadrilateralMesh>()
    }

    /// Determines if this mesh is a triangular mesh.
    ///
    /// # Returns
    ///
    /// - `bool`: `true` if the mesh is of type `TriangularMesh`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::{extrudable_mesh::ExtrudableMesh, prismatic_mesh::TriangularMesh};
    ///
    /// let some_mesh = TriangularMesh::new(
    ///     vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
    ///     vec![vec![0, 1, 2]],
    /// );
    ///
    /// assert!(some_mesh.is_tri_mesh());
    /// ```
    fn is_tri_mesh(&self) -> bool {
        self.as_any().is::<TriangularMesh>()
    }

    /// Attempts to cast this mesh to a `QuadrilateralMesh` reference.
    ///
    /// # Returns
    ///
    /// - `Option<&QuadrilateralMesh>`: Some reference if successful, `None` otherwise.
    fn as_quad(&self) -> Option<&QuadrilateralMesh> {
        self.as_any().downcast_ref::<QuadrilateralMesh>()
    }

    /// Attempts to cast this mesh to a `TriangularMesh` reference.
    ///
    /// # Returns
    ///
    /// - `Option<&TriangularMesh>`: Some reference if successful, `None` otherwise.
    fn as_tri(&self) -> Option<&TriangularMesh> {
        self.as_any().downcast_ref::<TriangularMesh>()
    }

    /// Provides a type-erased reference to the mesh to allow downcasting to a specific type.
    ///
    /// # Returns
    ///
    /// - `&dyn Any`: A reference to the mesh as an `Any` type.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[cfg(test)]
mod tests {
    use super::ExtrudableMesh;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};

    #[test]
    /// Tests the `is_valid_for_extrusion` method for both quadrilateral and triangular meshes.
    /// Verifies that valid meshes return `true`, while invalid meshes return `false`.
    fn test_is_valid_for_extrusion() {
        // Valid quadrilateral mesh
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.is_valid_for_extrusion(), "Valid quadrilateral mesh should return true");

        // Valid triangular mesh
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.is_valid_for_extrusion(), "Valid triangular mesh should return true");
    }

    #[test]
    /// Tests the `get_vertices` method to ensure it returns the correct vertices for both quadrilateral and triangular meshes.
    fn test_get_vertices() {
        let quad_vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let quad_mesh = QuadrilateralMesh::new(quad_vertices.clone(), vec![vec![0, 1, 2, 3]]);
        assert_eq!(quad_mesh.get_vertices(), quad_vertices, "Quadrilateral vertices should match the input");

        let tri_vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let tri_mesh = TriangularMesh::new(tri_vertices.clone(), vec![vec![0, 1, 2]]);
        assert_eq!(tri_mesh.get_vertices(), tri_vertices, "Triangular vertices should match the input");
    }

    #[test]
    /// Tests the `get_cells` method to ensure it returns the correct cell connectivity for both quadrilateral and triangular meshes.
    fn test_get_cells() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            quad_cells.clone(),
        );
        assert_eq!(quad_mesh.get_cells(), quad_cells, "Quadrilateral cells should match the input");

        let tri_cells = vec![vec![0, 1, 2]];
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            tri_cells.clone(),
        );
        assert_eq!(tri_mesh.get_cells(), tri_cells, "Triangular cells should match the input");
    }

    #[test]
    /// Tests `is_quad_mesh` and `is_tri_mesh` methods to verify correct type identification for quadrilateral and triangular meshes.
    fn test_mesh_type_identification() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.is_quad_mesh(), "Quadrilateral mesh should identify as quad mesh");
        assert!(!quad_mesh.is_tri_mesh(), "Quadrilateral mesh should not identify as tri mesh");

        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.is_tri_mesh(), "Triangular mesh should identify as tri mesh");
        assert!(!tri_mesh.is_quad_mesh(), "Triangular mesh should not identify as quad mesh");
    }

    #[test]
    /// Tests `as_quad` and `as_tri` methods to verify proper downcasting to specific mesh types.
    fn test_mesh_downcasting() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(quad_mesh.as_quad().is_some(), "Downcast to QuadrilateralMesh should succeed");
        assert!(quad_mesh.as_tri().is_none(), "Downcast to TriangularMesh should fail");

        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(tri_mesh.as_tri().is_some(), "Downcast to TriangularMesh should succeed");
        assert!(tri_mesh.as_quad().is_none(), "Downcast to QuadrilateralMesh should fail");
    }
}
