use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use crate::extrusion::core::extrudable_mesh::ExtrudableMesh;
use crate::extrusion::use_cases::{vertex_extrusion::VertexExtrusion, cell_extrusion::CellExtrusion};
use crate::domain::mesh::Mesh;

/// The `ExtrudeMeshUseCase` struct provides methods for extruding 2D meshes (either quadrilateral or triangular)
/// into 3D volumetric meshes (hexahedrons or prisms) based on a given extrusion depth and layer count.
///
/// This struct builds upon lower-level extrusion operations for vertices and cells, assembling a fully extruded
/// 3D mesh by extruding the vertices and connecting them in new 3D cells.
///
/// # Example
///
/// ```rust,ignore
/// let mesh = QuadrilateralMesh::new(...);
/// let depth = 5.0;
/// let layers = 3;
/// let extruded_mesh = ExtrudeMeshUseCase::extrude_to_hexahedron(&mesh, depth, layers);
/// ```
pub struct ExtrudeMeshUseCase;

impl ExtrudeMeshUseCase {
    /// Extrudes a quadrilateral 2D mesh into a 3D mesh with hexahedral cells.
    ///
    /// This method first extrudes the vertices to create multiple layers, then extrudes each quadrilateral cell
    /// into a hexahedral cell for each layer. Finally, it assembles the extruded vertices and cells into a
    /// `Mesh` structure, representing the final 3D mesh.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a `QuadrilateralMesh`, which is the 2D quadrilateral mesh to be extruded.
    /// - `depth`: A `f64` specifying the total depth of extrusion along the z-axis.
    /// - `layers`: An `usize` indicating the number of layers to extrude.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok(Mesh)` with the fully extruded mesh if successful,
    ///   or an `Err(String)` if the mesh is invalid for extrusion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let quad_mesh = QuadrilateralMesh::new(...);
    /// let result = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, 10.0, 3);
    /// assert!(result.is_ok());
    /// ```
    pub fn extrude_to_hexahedron(mesh: &QuadrilateralMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if !mesh.is_valid_for_extrusion() {
            return Err("Invalid mesh: Expected a quadrilateral mesh".to_string());
        }

        // Extrude vertices
        let extruded_vertices = VertexExtrusion::extrude_vertices(mesh.get_vertices(), depth, layers);

        // Extrude quadrilateral cells to hexahedrons
        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(mesh.get_cells(), layers);

        // Build the final Mesh
        let mut extruded_mesh = Mesh::new();
        for (id, vertex) in extruded_vertices.into_iter().enumerate() {
            extruded_mesh.set_vertex_coordinates(id, vertex);
        }

        for (cell_id, vertices) in extruded_cells.into_iter().enumerate() {
            let cell = crate::domain::mesh_entity::MeshEntity::Cell(cell_id);
            extruded_mesh.add_entity(cell.clone());
            for vertex in vertices {
                extruded_mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(vertex));
            }
        }

        Ok(extruded_mesh)
    }

    /// Extrudes a triangular 2D mesh into a 3D mesh with prismatic cells.
    ///
    /// This method first extrudes the vertices to create multiple layers, then extrudes each triangular cell
    /// into a prismatic cell for each layer. Finally, it assembles the extruded vertices and cells into a
    /// `Mesh` structure, representing the final 3D mesh.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to a `TriangularMesh`, which is the 2D triangular mesh to be extruded.
    /// - `depth`: A `f64` specifying the total depth of extrusion along the z-axis.
    /// - `layers`: An `usize` indicating the number of layers to extrude.
    ///
    /// # Returns
    ///
    /// - `Result<Mesh, String>`: Returns `Ok(Mesh)` with the fully extruded mesh if successful,
    ///   or an `Err(String)` if the mesh is invalid for extrusion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tri_mesh = TriangularMesh::new(...);
    /// let result = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, 5.0, 2);
    /// assert!(result.is_ok());
    /// ```
    pub fn extrude_to_prism(mesh: &TriangularMesh, depth: f64, layers: usize) -> Result<Mesh, String> {
        if !mesh.is_valid_for_extrusion() {
            return Err("Invalid mesh: Expected a triangular mesh".to_string());
        }

        // Extrude vertices
        let extruded_vertices = VertexExtrusion::extrude_vertices(mesh.get_vertices(), depth, layers);

        // Extrude triangular cells to prisms
        let extruded_cells = CellExtrusion::extrude_triangular_cells(mesh.get_cells(), layers);

        // Build the final Mesh
        let mut extruded_mesh = Mesh::new();
        for (id, vertex) in extruded_vertices.into_iter().enumerate() {
            extruded_mesh.set_vertex_coordinates(id, vertex);
        }

        for (cell_id, vertices) in extruded_cells.into_iter().enumerate() {
            let cell = crate::domain::mesh_entity::MeshEntity::Cell(cell_id);
            extruded_mesh.add_entity(cell.clone());
            for vertex in vertices {
                extruded_mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(vertex));
            }
        }

        Ok(extruded_mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::ExtrudeMeshUseCase;
    use crate::extrusion::core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};

    #[test]
    /// Test extruding a quadrilateral mesh to a hexahedral mesh.
    /// This test checks that the extruded mesh contains the correct number of vertices and cells.
    fn test_extrude_to_hexahedron() {
        let quad_mesh = QuadrilateralMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![vec![0, 1, 2, 3]],
        );
        let depth = 3.0;
        let layers = 2;

        let result = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, depth, layers);
        assert!(result.is_ok(), "Extrusion should succeed for a valid quadrilateral mesh");

        let extruded_mesh = result.unwrap();
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Vertex(0)), 4 * (layers + 1));
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)), layers);
    }

    #[test]
    /// Test extruding a triangular mesh to a prismatic mesh.
    /// This test checks that the extruded mesh contains the expected number of vertices and cells.
    fn test_extrude_to_prism() {
        let tri_mesh = TriangularMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![vec![0, 1, 2]],
        );
        let depth = 2.0;
        let layers = 3;

        let result = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, depth, layers);
        assert!(result.is_ok(), "Extrusion should succeed for a valid triangular mesh");

        let extruded_mesh = result.unwrap();
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Vertex(0)), 3 * (layers + 1));
        assert_eq!(extruded_mesh.count_entities(&crate::domain::mesh_entity::MeshEntity::Cell(0)), layers);
    }
}
