/// Integration test cases
#[cfg(test)]
mod integration_tests {
    use crate::mesh::mesh_ops::Mesh;

    #[test]
    fn test_mesh_parsing_and_area() {
        let mesh = Mesh::load_from_gmsh("test.msh2").unwrap();
        assert_eq!(mesh.nodes.len(), 6);
        assert_eq!(mesh.elements.len(), 2);

        for element in &mesh.elements {
            assert!(element.area > 0.0, "Element area should be greater than 0");
        }
    }
}