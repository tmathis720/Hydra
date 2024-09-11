#[cfg(test)]
mod tests {
    use crate::input::gmsh::GmshParser;  // Ensure this path aligns with your code structure
    use crate::domain::mesh::Mesh;

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
        let mesh = load_mesh_from_gmsh("inputs/circular_lake.msh2");

        // Test for basic validity
        assert!(!mesh.nodes.is_empty(), "Circle mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Circle mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Circle mesh faces should not be empty");

        // Add specific tests based on your mesh structure
        assert_eq!(mesh.nodes.len(), 424, "Incorrect number of nodes in circle mesh");
        assert_eq!(mesh.elements.len(), 849, "Incorrect number of elements in circle mesh");
    }

    #[test]
    fn test_coastal_island_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/coastal_island.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Coastal Island mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Coastal Island mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Coastal Island mesh faces should not be empty");

        // Add specific tests based on expected structure
        assert_eq!(mesh.nodes.len(), 1075, "Incorrect number of nodes in Coastal Island mesh");
        assert_eq!(mesh.elements.len(), 2154, "Incorrect number of elements in Coastal Island mesh");
    }

    #[test]
    fn test_lagoon_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/elliptical_lagoon.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Lagoon mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Lagoon mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Lagoon mesh faces should not be empty");

        // Further checks on expected properties
        assert_eq!(mesh.nodes.len(), 848, "Incorrect number of nodes in lagoon mesh");
        assert_eq!(mesh.elements.len(), 1697, "Incorrect number of elements in lagoon mesh");
    }

    #[test]
    fn test_meandering_river_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/meandering_river.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Meandering river mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Meandering river mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Meandering river mesh faces should not be empty");

        // Further checks on the structure
        assert_eq!(mesh.nodes.len(), 695, "Incorrect number of nodes in meandering river mesh");
        assert_eq!(mesh.elements.len(), 1386, "Incorrect number of elements in meandering river mesh");
    }

    #[test]
    fn test_polygon_estuary_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/polygon_estuary.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Polygon estuary mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Polygon estuary mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Polygon estuary mesh faces should not be empty");

        // Further checks on the structure
        assert_eq!(mesh.nodes.len(), 469, "Incorrect number of nodes in Polygon estuary mesh");
        assert_eq!(mesh.elements.len(), 941, "Incorrect number of elements in Polygon estuary mesh");
    }

    #[test]
    fn test_rectangle_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/rectangle.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Rectangle mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Rectangle mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Rectangle mesh faces should not be empty");

        // Further checks on the structure
        assert_eq!(mesh.nodes.len(), 78, "Incorrect number of nodes in Rectangle mesh");
        assert_eq!(mesh.elements.len(), 158, "Incorrect number of elements in Rectangle mesh");
    }

    #[test]
    fn test_rectangle_channel_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/rectangular_channel.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Rectangular channel mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Rectangular channel mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Rectangular channel mesh faces should not be empty");

        // Further checks on the structure
        assert_eq!(mesh.nodes.len(), 149, "Incorrect number of nodes in Rectangular channel mesh");
        assert_eq!(mesh.elements.len(), 300, "Incorrect number of elements in Rectangular channel mesh");
    }

    #[test]
    fn test_triangle_basin_mesh_import() {
        let mesh = load_mesh_from_gmsh("inputs/triangular_basin.msh2");

        // Validate the mesh structure
        assert!(!mesh.nodes.is_empty(), "Triangular basin mesh nodes should not be empty");
        assert!(!mesh.elements.is_empty(), "Triangular basin mesh elements should not be empty");
        assert!(!mesh.faces.is_empty(), "Triangular basin mesh faces should not be empty");

        // Further checks on the structure
        assert_eq!(mesh.nodes.len(), 66, "Incorrect number of nodes in Triangular basin mesh");
        assert_eq!(mesh.elements.len(), 133, "Incorrect number of elements in Triangular basin mesh");
    }
}
