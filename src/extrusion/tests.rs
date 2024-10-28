#[cfg(test)]
mod tests {
    use crate::extrusion::{
        core::{hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh},
        infrastructure::logger::Logger,
        interface_adapters::extrusion_service::ExtrusionService,
        use_cases::extrude_mesh::ExtrudeMeshUseCase,
    };
    use crate::input_output::gmsh_parser::GmshParser;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_extrude_rectangle_to_hexahedron() {
        let temp_file_path = "inputs/rectangle_quad.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let quad_mesh = QuadrilateralMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 5.0;
        let layers = 3;
        let extruded_mesh = ExtrudeMeshUseCase::extrude_to_hexahedron(&quad_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());
        let extruded_mesh = extruded_mesh.unwrap();

        // The expected number of vertices should be (nx + 1) * (ny + 1) * (layers + 1)
        let expected_vertices = 78 * (layers + 1); // 78 vertices in base 2D mesh
        let num_vertices = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded hexahedron mesh");

        // The expected number of cells should be nx * ny * layers
        let expected_cells = 96 * layers; // 96 quadrilateral cells in base mesh
        let num_cells = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_cells, "Incorrect number of cells in extruded hexahedron mesh");
    }

    #[test]
    fn test_extrude_triangle_basin_to_prism() {
        let temp_file_path = "inputs/triangular_basin.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let tri_mesh = TriangularMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 4.0;
        let layers = 2;
        let extruded_mesh = ExtrudeMeshUseCase::extrude_to_prism(&tri_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());
        let extruded_mesh = extruded_mesh.unwrap();

        // The expected number of vertices should be num_vertices_2d * (layers + 1)
        let expected_vertices = 66 * (layers + 1); // 66 vertices in base 2D triangular mesh
        let num_vertices = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded prismatic mesh");

        // The expected number of cells should be num_cells_2d * layers
        let expected_cells = 133 * layers; // 133 triangular cells in base mesh
        let num_cells = extruded_mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_cells, "Incorrect number of cells in extruded prismatic mesh");
    }

    #[test]
    fn test_extrusion_service_for_hexahedron() {
        let temp_file_path = "inputs/rectangle_quad.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh_2d = result.unwrap();
        let quad_mesh = QuadrilateralMesh::new(mesh_2d.get_vertices(), mesh_2d.get_cell_vertex_indices());

        let depth = 3.0;
        let layers = 2;
        let extruded_mesh = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers);
        assert!(extruded_mesh.is_ok());

        // Basic assertions about the 3D mesh structure
        let extruded_mesh = extruded_mesh.unwrap();
        assert!(extruded_mesh.count_entities(&MeshEntity::Vertex(0)) > 0);
        assert!(extruded_mesh.count_entities(&MeshEntity::Cell(0)) > 0);
    }

    #[test]
    fn test_logger_functionality() {
        let mut logger = Logger::new(None).expect("Failed to initialize logger");
        logger.info("Starting extrusion test for hexahedral mesh");
        logger.warn("Mesh contains irregular vertices");
        logger.error("Extrusion failed due to invalid cell data");

        // Check if the logger runs without error; manual check of output is recommended.
    }
}
