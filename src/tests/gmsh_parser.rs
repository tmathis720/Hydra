#[cfg(test)]
mod tests {
    use crate::domain::Mesh;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_gmsh_parser() {
        // Create a temporary GMSH file
        let mut file = File::create("test.msh2").unwrap();
        writeln!(file, "$Nodes\n4\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 1.0 1.0 0.0\n4 0.0 1.0 0.0\n$EndNodes").unwrap();
        
        // Adjust this to define only 1 element or expect 2 elements in the test
        writeln!(file, "$Elements\n1\n1 2 1 1 1 2 3 4\n$EndElements").unwrap();

        // Load the mesh from the file
        let mesh = Mesh::load_from_gmsh("test.msh2").unwrap();
        
        // These assertions align with the data now
        assert_eq!(mesh.nodes.len(), 4);
        assert_eq!(mesh.elements.len(), 1); // Now expecting 1 element
        assert_eq!(mesh.faces.len(), 0); // Face handling is not implemented yet
    }
}