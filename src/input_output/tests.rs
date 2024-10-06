#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_output::mesh_generation;
    use crate::input_output::mesh_io;
    use crate::domain::Mesh;
    use crate::input_output::mesh_io::MeshFormat;

    #[test]
    fn test_structured_2d_generation() {
        let mesh = Mesh::structured_2d(5, 5, 1.0, 1.0);
        assert_eq!(mesh.get_cells().len(), 16); // 4x4 cells in a 5x5 grid
    }

/*     #[test]
    fn test_mesh_io_import_export() {
        let mesh = Mesh::structured_2d(5, 5, 1.0, 1.0);
        let result = mesh.to_file("output.msh", MeshFormat::Gmsh);
        assert!(result.is_ok());

        let imported_mesh = Mesh::from_file("output.msh", MeshFormat::Gmsh);
        assert!(imported_mesh.is_ok());
    } */
}
