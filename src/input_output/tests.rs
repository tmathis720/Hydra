#[cfg(test)]
mod tests {

    use crate::input_output::mesh_generation::MeshGenerator;
    use crate::input_output::gmsh_parser::GmshParser;
    use crate::domain::MeshEntity;

    #[test]
    fn test_circle_mesh_import() {
        let temp_file_path = "inputs/circular_lake.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        // Test for basic validity
        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)

        assert!(node_count > 0, "Circle mesh node_count should not be empty");
        assert!(element_count > 0, "Circle mesh elements should not be empty");

        // Check that the number of node_count matches the expected value
        assert_eq!(node_count, 424, "Incorrect number of node_count in circle mesh");
        assert_eq!(element_count, 849, "Incorrect number of elements in circle mesh");
    }

    #[test]
    fn test_coastal_island_mesh_import() {
        let temp_file_path = "inputs/coastal_island.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)

        // Validate the mesh structure
        assert!(node_count > 0, "Coastal Island mesh node_count should not be empty");
        assert!(element_count > 0, "Coastal Island mesh elements should not be empty");

        // Add specific tests based on expected structure
        assert_eq!(node_count, 1075, "Incorrect number of node_count in Coastal Island mesh");
        assert_eq!(element_count, 2154, "Incorrect number of elements in Coastal Island mesh");
    }

    #[test]
    fn test_lagoon_mesh_import() {
        let temp_file_path = "inputs/elliptical_lagoon.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)

        // Validate the mesh structure
        assert!(node_count > 0, "Lagoon mesh node_count should not be empty");
        assert!(element_count > 0, "Lagoon mesh elements should not be empty");

        // Further checks on expected properties
        assert_eq!(node_count, 848, "Incorrect number of node_count in Lagoon mesh");
        assert_eq!(element_count, 1697, "Incorrect number of elements in Lagoon mesh");
    }

    #[test]
    fn test_meandering_river_mesh_import() {
        let temp_file_path = "inputs/meandering_river.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)

        // Validate the mesh structure
        assert!(node_count > 0, "Meandering River mesh node_count should not be empty");
        assert!(element_count > 0, "Meandering River mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(node_count, 695, "Incorrect number of node_count in Meandering River mesh");
        assert_eq!(element_count, 1386, "Incorrect number of elements in Meandering River mesh");
    }

    #[test]
    fn test_polygon_estuary_mesh_import() {
        let temp_file_path = "inputs/polygon_estuary.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)
        
        // Validate the mesh structure
        assert!(node_count > 0, "Polygon Estuary mesh node_count should not be empty");
        assert!(element_count > 0, "Polygon Estuary mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(node_count, 469, "Incorrect number of node_count in Polygon Estuary mesh");
        assert_eq!(element_count, 941, "Incorrect number of elements in Polygon Estuary mesh");
    }

    #[test]
    fn test_rectangle_mesh_import() {
        let temp_file_path = "inputs/rectangle.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)
        
        // Validate the mesh structure
        assert!(node_count > 0, "Rectangle mesh node_count should not be empty");
        assert!(element_count > 0, "Rectangle mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(node_count, 78, "Incorrect number of node_count in Rectangle mesh");
        assert_eq!(element_count, 158, "Incorrect number of elements in Rectangle mesh");
    }

    #[test]
    fn test_rectangle_channel_mesh_import() {
        let temp_file_path = "inputs/rectangular_channel.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)
        
        // Validate the mesh structure
        assert!(node_count > 0, "Rectangular Channel mesh node_count should not be empty");
        assert!(element_count > 0, "Rectangular Channel mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(node_count, 149, "Incorrect number of node_count in Rectangular Channel mesh");
        assert_eq!(element_count, 300, "Incorrect number of elements in Rectangular Channel mesh");
    }

    #[test]
    fn test_triangle_basin_mesh_import() {
        let temp_file_path = "inputs/triangular_basin.msh2";

        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();

        let node_count = mesh.count_entities(&MeshEntity::Vertex(0)); // Count vertices (node_count)
        let element_count = mesh.count_entities(&MeshEntity::Cell(0)); // Count cells (elements)
        
        // Validate the mesh structure
        assert!(node_count > 0, "Triangular Basin mesh node_count should not be empty");
        assert!(element_count > 0, "Triangular Basin mesh elements should not be empty");


        // Further checks on the structure
        assert_eq!(node_count, 66, "Incorrect number of node_count in Triangular Basin mesh");
        assert_eq!(element_count, 133, "Incorrect number of elements in Triangular Basin mesh");
    }

    #[test]
    fn test_generate_rectangle_2d() {
        let width = 10.0;
        let height = 5.0;
        let nx = 4; // Number of cells along x-axis
        let ny = 2; // Number of cells along y-axis

        let mesh = MeshGenerator::generate_rectangle_2d(width, height, nx, ny);

        // The number of vertices should be (nx + 1) * (ny + 1)
        let expected_num_vertices = (nx + 1) * (ny + 1);
        let num_vertices = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_num_vertices, "Incorrect number of vertices");

        // The number of quadrilateral cells should be nx * ny
        let expected_num_cells = nx * ny;
        let num_cells = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_num_cells, "Incorrect number of cells");
    }

    #[test]
    fn test_generate_rectangle_3d() {
        let width = 10.0;
        let height = 5.0;
        let depth = 3.0;
        let nx = 4; // Number of cells along x-axis
        let ny = 2; // Number of cells along y-axis
        let nz = 1; // Number of cells along z-axis

        let mesh = MeshGenerator::generate_rectangle_3d(width, height, depth, nx, ny, nz);

        // The number of vertices should be (nx + 1) * (ny + 1) * (nz + 1)
        let expected_num_vertices = (nx + 1) * (ny + 1) * (nz + 1);
        let num_vertices = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_num_vertices, "Incorrect number of vertices");

        // The number of hexahedral cells should be nx * ny * nz
        let expected_num_cells = nx * ny * nz;
        let num_cells = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_num_cells, "Incorrect number of cells");
    }

    #[test]
    fn test_generate_circle() {
        let radius = 5.0;
        let num_divisions = 8; // Number of divisions around the circle

        let mesh = MeshGenerator::generate_circle(radius, num_divisions);

        // The number of vertices should be num_divisions + 1 (center vertex + boundary vertices)
        let expected_num_vertices = num_divisions + 1;
        let num_vertices = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .count();
        assert_eq!(num_vertices, expected_num_vertices, "Incorrect number of vertices");

        // The number of triangular cells should be equal to num_divisions
        let expected_num_cells = num_divisions;
        let num_cells = mesh
            .entities
            .read().expect("Failed to acquire read lock")
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .count();
        assert_eq!(num_cells, expected_num_cells, "Incorrect number of cells");
    }
}
