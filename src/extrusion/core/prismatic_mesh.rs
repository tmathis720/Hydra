use std::any::Any;
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

/// The `TriangularMesh` struct represents a 2D triangular mesh, consisting of vertices
/// in 3D space and cells defined by indices of the vertices, with each cell representing a triangle.
///
/// This mesh struct is designed to be extruded into 3D prismatic meshes.
///
/// # Example
///
/// ```
/// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
/// let cells = vec![vec![0, 1, 2]];
/// let tri_mesh = TriangularMesh::new(vertices, cells);
/// assert!(tri_mesh.is_valid_for_extrusion());
/// ```
#[derive(Debug)]
pub struct TriangularMesh {
    /// A list of vertices represented as `[f64; 3]` coordinates in 3D space.
    vertices: Vec<[f64; 3]>,
    /// A list of cells, where each cell is a `Vec<usize>` containing indices into the `vertices` array,
    /// representing the corners of each triangular cell.
    cells: Vec<Vec<usize>>,
}

impl TriangularMesh {
    /// Creates a new `TriangularMesh` with the specified vertices and cells.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` specifying the 3D coordinates of each vertex.
    /// - `cells`: A `Vec<Vec<usize>>` where each inner `Vec<usize>` contains 3 indices into `vertices`,
    ///   representing a triangular cell.
    ///
    /// # Returns
    ///
    /// - `Self`: A new `TriangularMesh` instance.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    /// let cells = vec![vec![0, 1, 2]];
    /// let tri_mesh = TriangularMesh::new(vertices, cells);
    /// ```
    pub fn new(vertices: Vec<[f64; 3]>, cells: Vec<Vec<usize>>) -> Self {
        TriangularMesh { vertices, cells }
    }
}

impl ExtrudableMesh for TriangularMesh {
    /// Checks if the mesh is valid for extrusion.
    ///
    /// This method verifies that all cells in the mesh are triangular (i.e., each cell
    /// contains exactly 3 vertices). If any cell does not contain 3 vertices, this function returns `false`.
    ///
    /// # Returns
    ///
    /// - `bool`: Returns `true` if all cells are triangular; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```
    /// let tri_mesh = TriangularMesh::new(...);
    /// assert!(tri_mesh.is_valid_for_extrusion());
    /// ```
    fn is_valid_for_extrusion(&self) -> bool {
        self.cells.iter().all(|cell| cell.len() == 3)
    }

    /// Returns a clone of the vertices in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<[f64; 3]>`: A vector of 3D coordinates representing the mesh vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let vertices = tri_mesh.get_vertices();
    /// ```
    fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.vertices.clone()
    }

    /// Returns a clone of the cells in the mesh.
    ///
    /// # Returns
    ///
    /// - `Vec<Vec<usize>>`: A vector of cells, where each cell is defined by 3 vertex indices.
    ///
    /// # Example
    ///
    /// ```
    /// let cells = tri_mesh.get_cells();
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
    /// ```
    /// let as_any = tri_mesh.as_any();
    /// ```
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::TriangularMesh;
    use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;

    #[test]
    /// Tests the creation of a `TriangularMesh` instance.
    /// Verifies that the mesh initializes correctly with the provided vertices and cells.
    fn test_triangular_mesh_creation() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let cells = vec![vec![0, 1, 2]];

        let tri_mesh = TriangularMesh::new(vertices.clone(), cells.clone());

        assert_eq!(tri_mesh.get_vertices(), vertices, "Vertices should match the input vertices");
        assert_eq!(tri_mesh.get_cells(), cells, "Cells should match the input cells");
    }

    #[test]
    /// Tests the `is_valid_for_extrusion` method of `TriangularMesh`.
    /// Verifies that the mesh is valid only if all cells are triangular.
    fn test_is_valid_for_extrusion() {
        let valid_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        assert!(valid_mesh.is_valid_for_extrusion(), "Mesh with all triangular cells should be valid");

        let invalid_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        assert!(!invalid_mesh.is_valid_for_extrusion(), "Mesh with non-triangular cells should be invalid");
    }

    #[test]
    /// Tests the `get_vertices` method of `TriangularMesh`.
    /// Verifies that `get_vertices` returns a clone of the original vertices.
    fn test_get_vertices() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let tri_mesh = TriangularMesh::new(vertices.clone(), vec![vec![0, 1, 2]]);
        
        assert_eq!(tri_mesh.get_vertices(), vertices, "Vertices should match the initialized vertices");
    }

    #[test]
    /// Tests the `get_cells` method of `TriangularMesh`.
    /// Verifies that `get_cells` returns a clone of the original cells.
    fn test_get_cells() {
        let cells = vec![vec![0, 1, 2]];
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            cells.clone(),
        );
        
        assert_eq!(tri_mesh.get_cells(), cells, "Cells should match the initialized cells");
    }

    #[test]
    /// Tests the `as_any` method of `TriangularMesh`.
    /// Verifies that the mesh can be treated as a `dyn Any` for type erasure.
    fn test_as_any() {
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );

        let as_any = tri_mesh.as_any();
        assert!(as_any.is::<TriangularMesh>(), "as_any should identify the struct type correctly");
    }
}
