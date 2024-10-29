#[cfg(test)]
mod tests {
    use crate::extrusion::{
        core::{extrudable_mesh::ExtrudableMesh, hexahedral_mesh::QuadrilateralMesh, prismatic_mesh::TriangularMesh},
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

        // Verify the expected number of vertices and cells in the extruded mesh
        let expected_vertices = quad_mesh.get_vertices().len() * (layers + 1);
        let num_vertices = extruded_mesh.count_entities(&MeshEntity::Vertex(0));
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded hexahedron mesh");

        let expected_cells = quad_mesh.get_cells().len() * layers;
        let num_cells = extruded_mesh.count_entities(&MeshEntity::Cell(0));
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

        let expected_vertices = tri_mesh.get_vertices().len() * (layers + 1);
        let num_vertices = extruded_mesh.count_entities(&MeshEntity::Vertex(0));
        assert_eq!(num_vertices, expected_vertices, "Incorrect number of vertices in extruded prismatic mesh");

        let expected_cells = tri_mesh.get_cells().len() * layers;
        let num_cells = extruded_mesh.count_entities(&MeshEntity::Cell(0));
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
