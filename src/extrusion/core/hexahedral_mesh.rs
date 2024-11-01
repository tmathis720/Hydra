use std::any::Any;
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

/// The `QuadrilateralMesh` struct represents a 2D quadrilateral mesh, containing vertices
/// in 3D space and cells defined by indices of the vertices, each representing a quadrilateral.
///
/// This mesh struct is intended for extrusion into 3D hexahedral meshes.
///
/// # Example
///
/// ```rust
/// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
/// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
/// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
/// let cells = vec![vec![0, 1, 2, 3]];
/// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
/// assert!(quad_mesh.is_valid_for_extrusion());
/// ```
#[derive(Debug)]
pub struct QuadrilateralMesh {
    /// A list of vertices represented as `[f64; 3]` coordinates in 3D space.
    vertices: Vec<[f64; 3]>,
    /// A list of cells, where each cell is a `Vec<usize>` containing indices into the `vertices` array,
    /// representing the corners of each quadrilateral cell.
    cells: Vec<Vec<usize>>,
}

impl QuadrilateralMesh {
    /// Creates a new `QuadrilateralMesh` with the specified vertices and cells.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` specifying the 3D coordinates of each vertex.
    /// - `cells`: A `Vec<Vec<usize>>` where each inner `Vec<usize>` contains 4 indices into `vertices`,
    ///   representing a quadrilateral cell.
    ///
    /// # Returns
    ///
    /// - `Self`: A new `QuadrilateralMesh` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
    /// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
    /// ```
    pub fn new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self {
        QuadrilateralMesh { vertices, cells }
    }
}

impl ExtrudableMesh for QuadrilateralMesh {
    /// Checks if the mesh is valid for extrusion.
    ///
    /// This method verifies that all cells in the mesh are quadrilateral (i.e., each cell
    /// contains exactly 4 vertices). If any cell does not contain 4 vertices, this function returns `false`.
    ///
    /// # Returns
    ///
    /// - `bool`: Returns `true` if all cells are quadrilateral; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
    /// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
    /// assert!(quad_mesh.is_valid_for_extrusion());
    /// ```
    fn is_valid_for_extrusion(&self) -> bool {
        // Ensure all cells have exactly 4 vertices
        self.cells.iter().all(|cell| cell.len() == 4)
    }

    /// Returns a clone of the vertices in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the mesh vertices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
    /// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices.clone(), cells);
    /// assert_eq!(quad_mesh.get_vertices(), vertices);
    /// ```
    fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.vertices.clone()
    }

    /// Returns a clone of the cells in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, where each cell is defined by 4 vertex indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
    /// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices, cells.clone());
    /// assert_eq!(quad_mesh.get_cells(), cells);
    /// ```
    fn get_cells(&self) -> Vec<Vec<usize>> {
        self.cells.clone()
    }

    /// Provides a type-erased reference to the current object, allowing it to be used
    /// as a generic `ExtrudableMesh` object.
    ///
    /// # Returns
    ///
    /// - `&dyn Any`: A type-erased reference to the mesh.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;
    /// use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2, 3]];
    /// let quad_mesh = QuadrilateralMesh::new(vertices, cells);
    /// let as_any = quad_mesh.as_any();
    /// ```
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::QuadrilateralMesh;
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Tests the creation of a `QuadrilateralMesh` instance.
    /// Verifies that the mesh initializes correctly with the provided vertices and cells.
    fn test_quadrilateral_mesh_creation() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let cells = vec![vec![0, 1, 2, 3]];
        
        let quad_mesh = QuadrilateralMesh::new(vertices.clone(), cells.clone());
        
        assert_eq!(quad_mesh.get_vertices(), vertices, "Vertices should match the input vertices");
        assert_eq!(quad_mesh.get_cells(), cells, "Cells should match the input cells");
    }

    #[test]
    /// Tests the `is_valid_for_extrusion` method of `QuadrilateralMesh`.
    /// Verifies that the mesh is valid only if all cells are quadrilateral.
    fn test_is_valid_for_extrusion() {
        let valid_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(valid_mesh.is_valid_for_extrusion(), "Mesh with all quadrilateral cells should be valid");

        let invalid_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(!invalid_mesh.is_valid_for_extrusion(), "Mesh with non-quadrilateral cells should be invalid");
    }

    #[test]
    /// Tests the `get_vertices` method of `QuadrilateralMesh`.
    /// Verifies that `get_vertices` returns a clone of the original vertices.
    fn test_get_vertices() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let quad_mesh = QuadrilateralMesh::new(vertices.clone(), vec![vec![0, 1, 2, 3]]);
        
        assert_eq!(quad_mesh.get_vertices(), vertices, "Vertices should match the initialized vertices");
    }

    #[test]
    /// Tests the `get_cells` method of `QuadrilateralMesh`.
    /// Verifies that `get_cells` returns a clone of the original cells.
    fn test_get_cells() {
        let cells = vec![vec![0, 1, 2, 3]];
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            cells.clone(),
        );
        
        assert_eq!(quad_mesh.get_cells(), cells, "Cells should match the initialized cells");
    }

    #[test]
    /// Tests the `as_any` method of `QuadrilateralMesh`.
    /// Verifies that the mesh can be treated as a `dyn Any` for type erasure.
    fn test_as_any() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );

        let as_any = quad_mesh.as_any();
        assert!(as_any.is::<QuadrilateralMesh>(), "as_any should identify the struct type correctly");
    }
}
