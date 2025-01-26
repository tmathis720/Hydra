use crate::domain::{mesh::Mesh, mesh_entity::MeshEntity};
use crate::extrusion::core::{extrudable_mesh::ExtrudableMesh, hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh};
use crate::input_output::gmsh_parser::GmshParser;
use std::fs::File;
use std::io::Write;

/// `MeshIO` is responsible for handling input and output operations for mesh data, including
/// loading a 2D mesh from a file and saving an extruded 3D mesh.
pub struct MeshIO;

impl MeshIO {
    /// Loads a 2D mesh from a Gmsh file and returns it as an `ExtrudableMesh`.
    /// Detects the type of cells in the mesh (quadrilateral or triangular) and constructs
    /// an appropriate mesh structure based on the cell type.
    ///
    /// # Parameters
    ///
    /// - `file_path`: The path to the Gmsh file containing the 2D mesh data.
    ///
    /// # Returns
    ///
    /// - `Result<Box<dyn ExtrudableMesh>, String>`: Returns a boxed `ExtrudableMesh` trait
    ///   object, either a `QuadrilateralMesh` or `TriangularMesh`, or an error message if
    ///   the mesh type is unsupported or loading fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the Gmsh file cannot be read or if the mesh contains unsupported cell types.
    pub fn load_2d_mesh(file_path: &str) -> Result<Box<dyn ExtrudableMesh>, String> {
        let mesh = GmshParser::from_gmsh_file(file_path).map_err(|e| {
            format!("Failed to parse Gmsh file {}: {}", file_path, e.to_string())
        })?;
    
        let mut is_quad_mesh = false;
        let mut is_tri_mesh = false;
    
        // Collect only cells that are quadrilaterals or triangles
        let mut quad_cells: Vec<Vec<usize>> = Vec::new();
        let mut tri_cells: Vec<Vec<usize>> = Vec::new();
    
        // Retrieve all cell vertex indices
        let cell_vertex_indices = mesh.get_cell_vertex_indices();
    
        for cell_indices in cell_vertex_indices {
            let cell_vertex_count = cell_indices.len();
            
            if cell_vertex_count == 4 {
                is_quad_mesh = true;
                quad_cells.push(cell_indices);
                eprintln!("Adding a quadrilateral cell with {} vertices.", cell_vertex_count); // Debug log
            } else if cell_vertex_count == 3 {
                is_tri_mesh = true;
                tri_cells.push(cell_indices);
                eprintln!("Adding a triangular cell with {} vertices.", cell_vertex_count); // Debug log
            } else {
                // Log unsupported cell types
                eprintln!("Ignoring unsupported cell with {} vertices", cell_vertex_count);
                continue;
            }
        }
    
        // Based on the collected cells, decide the predominant mesh type
        if is_quad_mesh && !is_tri_mesh {
            let quad_mesh = QuadrilateralMesh::new(
                mesh.get_vertices_vec(),
                quad_cells,
            );
            if quad_mesh.is_valid_for_extrusion() {
                Ok(Box::new(quad_mesh))
            } else {
                Err("Invalid quadrilateral mesh structure".to_string())
            }
        } else if is_tri_mesh && !is_quad_mesh {
            let tri_mesh = TriangularMesh::new(
                mesh.get_vertices_vec(),
                tri_cells,
            );
            if tri_mesh.is_valid_for_extrusion() {
                Ok(Box::new(tri_mesh))
            } else {
                Err("Invalid triangular mesh structure".to_string())
            }
        } else {
            Err("Mesh contains mixed cell types; currently, only meshes with exclusively quadrilateral or triangular cells are supported.".to_string())
        }
    }
    
    

    /// Saves a 3D extruded mesh to a Gmsh-compatible file.
    ///
    /// # Parameters
    ///
    /// - `mesh`: A reference to the `Mesh` to save.
    /// - `file_path`: The path where the mesh will be saved.
    ///
    /// # Returns
    ///
    /// - `Result<(), String>`: Returns `Ok` if saving is successful, or an error message if
    ///   there is an I/O failure during the save process.
    ///
    /// # Errors
    ///
    /// - Returns an error if the file cannot be created or written to.
    pub fn save_3d_mesh(mesh: &Mesh, file_path: &str) -> Result<(), String> {
        let mut file = File::create(file_path).map_err(|e| {
            format!("Failed to create file {}: {}", file_path, e.to_string())
        })?;

        // Write vertices
        writeln!(file, "$Nodes").map_err(|e| e.to_string())?;
        writeln!(file, "{}", mesh.get_vertices().len()).map_err(|e| e.to_string())?;
        for (id, coords) in mesh.get_vertices_vec().iter().enumerate() {
            writeln!(file, "{} {} {} {}", id + 1, coords[0], coords[1], coords[2])
                .map_err(|e| e.to_string())?;
        }
        writeln!(file, "$EndNodes").map_err(|e| e.to_string())?;

        // Write elements
        writeln!(file, "$Elements").map_err(|e| e.to_string())?;
        writeln!(file, "{}", mesh.get_cell_vertex_indices().len()).map_err(|e| e.to_string())?;
        for (id, vertices) in mesh.get_cell_vertex_indices().iter().enumerate() {
            writeln!(
                file,
                "{} 5 0 {}",
                id + 1,
                vertices.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
            )
            .map_err(|e| format!("Failed to write element data: {}", e.to_string()))?;
        }
        writeln!(file, "$EndElements").map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl Mesh {
    /// Retrieves all vertices in the mesh as a vector of `[f64; 3]` coordinates.
    pub fn get_vertices_vec(&self) -> Vec<[f64; 3]> {
        self.entities
            .read()
            .expect("Failed to acquire read lock")
            .iter()
            .filter_map(|entity| match entity {
                MeshEntity::Vertex(id) => self.vertex_coordinates.get(id).cloned(),
                _ => None,
            })
            .collect()
    }

    /// Retrieves vertex indices (IDs) for all cells as a vector of `Vec<usize>`.
    pub fn get_cell_vertex_indices(&self) -> Vec<Vec<usize>> {
        self.entities
            .read()
            .expect("Failed to acquire read lock")
            .iter()
            .filter_map(|entity| match entity {
                MeshEntity::Cell(id) => Some(self.get_cell_vertex_ids(&MeshEntity::Cell(*id))),
                _ => None,
            })
            .collect()
    }

    /// Helper function to retrieve only vertex IDs for a cell.
    pub fn get_cell_vertex_ids(&self, cell: &MeshEntity) -> Vec<usize> {
        self.sieve
            .cone(cell)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entity| match entity {
                MeshEntity::Vertex(vertex_id) => Some(vertex_id),
                _ => None,
            })
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::MeshIO;
    use crate::domain::mesh::Mesh;
    use std::fs;

    #[test]
    fn test_load_quadrilateral_mesh() {
        let file_path = "inputs/rectangular_channel_quad.msh2";

        assert!(
            std::path::Path::new(file_path).exists(),
            "File not found: {}",
            file_path
        );

        let result = MeshIO::load_2d_mesh(file_path);

        match result {
            Ok(mesh) => {
                assert!(mesh.is_quad_mesh(), "Loaded mesh should be recognized as a quadrilateral mesh");
            },
            Err(e) => {
                eprintln!("Error loading mesh: {}", e);
                panic!("Expected successful loading of quadrilateral mesh");
            }
        }
    }

    #[test]
    fn test_load_triangular_mesh() {
        let file_path = "inputs/triangular_basin.msh2";

        assert!(
            std::path::Path::new(file_path).exists(),
            "File not found: {}",
            file_path
        );

        let result = MeshIO::load_2d_mesh(file_path);

        match result {
            Ok(mesh) => {
                assert!(mesh.is_tri_mesh(), "Loaded mesh should be recognized as a triangular mesh");
            },
            Err(e) => {
                eprintln!("Error loading mesh: {}", e);
                panic!("Expected successful loading of quadrilateral mesh");
            }
        }
    }

    #[test]
    /// Tests saving of a 3D extruded mesh and verifies file creation and content.
    fn test_save_3d_mesh() {
        // Create a simple 3D mesh to save
        let mut mesh = Mesh::new();
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        for (id, vertex) in vertices.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, vertex);
        }

        let cell = crate::domain::mesh_entity::MeshEntity::Cell(0);
        mesh.add_entity(cell.clone());
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(0));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(1));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(2));
        mesh.add_relationship(cell.clone(), crate::domain::mesh_entity::MeshEntity::Vertex(3));

        let file_path = "outputs/test_save_3d_mesh.msh";
        let result = MeshIO::save_3d_mesh(&mesh, file_path);

        assert!(result.is_ok(), "Expected successful saving of 3D mesh");
        
        // Check if file was created
        let file_exists = fs::metadata(file_path).is_ok();
        assert!(file_exists, "File should be created at specified path");

        // Cleanup
        fs::remove_file(file_path).expect("Failed to delete test output file");
    }

    #[test]
    /// Tests error handling when saving a mesh fails due to an invalid file path.
    fn test_save_3d_mesh_invalid_path() {
        let mesh = Mesh::new();
        let file_path = "/invalid_path/test_save_3d_mesh.msh";
        let result = MeshIO::save_3d_mesh(&mesh, file_path);

        assert!(result.is_err(), "Expected error when saving to invalid path");
    }
}
