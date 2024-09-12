#[cfg(test)]
mod tests {
    use crate::input::gmsh::GmshParser;  // Ensure this path aligns with your code structure
    use crate::domain::{Mesh, Node, Element};
    use std::io::Error;

    // Helper function to load and initialize a mesh
    fn load_mesh_from_gmsh(file_path: &str) -> Mesh {
        let (nodes, elements, faces) = GmshParser::load_mesh(file_path).expect("Failed to load mesh");
        Mesh {
            nodes,
            elements,
            faces,
            neighbors: vec![],
            face_element_relations: vec![],
        }
    }

    #[test]
    fn test_circle_mesh_import() {
        // Assuming you saved the circle mesh as `circle.msh`
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/circular_lake.msh2")
            .expect("Failed to load circle mesh");

        // Test for basic validity
        assert!(!nodes.is_empty(), "Circle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Circle mesh elements should not be empty");

        // Expected counts based on your Gmsh file (you may need to adjust these numbers)
        let expected_node_count = 424;
        let expected_element_count = 849;

        // Check that the number of nodes matches the expected value
        assert_eq!(nodes.len(), expected_node_count, "Incorrect number of nodes in circle mesh");

        // Check that the number of elements matches the expected value
        assert_eq!(elements.len(), expected_element_count, "Incorrect number of elements in circle mesh");

        // You can also add tests for face counts, but they might need to be derived if not explicitly in Gmsh
    }

    #[test]
    fn test_coastal_island_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/coastal_island.msh2")
            .expect("Failed to load Coastal Island mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Coastal Island mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Coastal Island mesh elements should not be empty");

        // Add specific tests based on expected structure
        assert_eq!(nodes.len(), 1075, "Incorrect number of nodes in Coastal Island mesh");
        assert_eq!(elements.len(), 2154, "Incorrect number of elements in Coastal Island mesh");
    }

    #[test]
    fn test_lagoon_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/elliptical_lagoon.msh2")
            .expect("Failed to load Lagoon mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Lagoon mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Lagoon mesh elements should not be empty");

        // Further checks on expected properties
        assert_eq!(nodes.len(), 848, "Incorrect number of nodes in Lagoon mesh");
        assert_eq!(elements.len(), 1697, "Incorrect number of elements in Lagoon mesh");
    }

    #[test]
    fn test_meandering_river_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/meandering_river.msh2")
            .expect("Failed to load Meandering River mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Meandering River mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Meandering River mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 695, "Incorrect number of nodes in Meandering River mesh");
        assert_eq!(elements.len(), 1386, "Incorrect number of elements in Meandering River mesh");
    }

    #[test]
    fn test_polygon_estuary_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/polygon_estuary.msh2")
            .expect("Failed to load Polygon Estuary mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Polygon Estuary mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Polygon Estuary mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 469, "Incorrect number of nodes in Polygon Estuary mesh");
        assert_eq!(elements.len(), 941, "Incorrect number of elements in Polygon Estuary mesh");
    }

    #[test]
    fn test_rectangle_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/rectangle.msh2")
            .expect("Failed to load Rectangle mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangle mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 78, "Incorrect number of nodes in Rectangle mesh");
        assert_eq!(elements.len(), 158, "Incorrect number of elements in Rectangle mesh");
    }

    #[test]
    fn test_rectangle_channel_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/rectangular_channel.msh2")
            .expect("Failed to load Rectangular Channel mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangular Channel mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangular Channel mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 149, "Incorrect number of nodes in Rectangular Channel mesh");
        assert_eq!(elements.len(), 300, "Incorrect number of elements in Rectangular Channel mesh");
    }

    #[test]
    fn test_triangle_basin_mesh_import() {
        let (nodes, elements, _faces) = GmshParser::load_mesh("inputs/triangular_basin.msh2")
            .expect("Failed to load Triangular Basin mesh");

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Triangular Basin mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Triangular Basin mesh elements should not be empty");


        // Further checks on the structure
        assert_eq!(nodes.len(), 66, "Incorrect number of nodes in Triangular Basin mesh");
        assert_eq!(elements.len(), 133, "Incorrect number of elements in Triangular Basin mesh");
    }
}
