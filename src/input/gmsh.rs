use crate::domain::{Face, Element, Node};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct GmshParser;

impl GmshParser {
    /// Load the mesh from a Gmsh file
    pub fn load_mesh(file_path: &str) -> Result<(Vec<Node>, Vec<Element>, Vec<Face>), io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut _faces = Vec::new(); // Handle face parsing if needed later

        let mut in_nodes_section = false;
        let mut in_elements_section = false;

        let mut node_count = 0;
        let mut element_count = 0;
        let mut current_node_line = 0;
        let mut current_element_line = 0;

        for line in reader.lines() {
            let line = line?;

            // Detect sections in the Gmsh file format
            if line.starts_with("$Nodes") {
                in_nodes_section = true;
                in_elements_section = false;
                current_node_line = 0;
                continue;
            } else if line.starts_with("$Elements") {
                in_nodes_section = false;
                in_elements_section = true;
                current_element_line = 0;
                continue;
            } else if line.starts_with("$EndNodes") || line.starts_with("$EndElements") {
                in_nodes_section = false;
                in_elements_section = false;
                continue;
            }

            // Parse node and element counts
            if in_nodes_section && current_node_line == 0 {
                node_count = line.parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid node count"))?;
                current_node_line += 1;
                continue;
            } else if in_elements_section && current_element_line == 0 {
                element_count = line.parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid element count"))?;
                current_element_line += 1;
                continue;
            }

            // Parse individual nodes
            if in_nodes_section && current_node_line <= node_count {
                nodes.push(Self::parse_node(&line)?);
                current_node_line += 1;
            }

            // Parse individual elements
            if in_elements_section && current_element_line <= element_count {
                elements.push(Self::parse_element(&line)?);
                current_element_line += 1;
            }
        }

        Ok((nodes, elements, _faces)) // Return the parsed data; face handling can be added later
    }

    /// Parse a single node from a line in the Gmsh file
    fn parse_node(line: &str) -> Result<Node, io::Error> {
        let mut split = line.split_whitespace();

        let id: u32 = Self::parse_next(&mut split, "Missing node ID")?;
        let x: f64 = Self::parse_next(&mut split, "Missing x coordinate")?;
        let y: f64 = Self::parse_next(&mut split, "Missing y coordinate")?;
        let z: f64 = Self::parse_next(&mut split, "Missing z coordinate")?;

        Ok(Node {
            id,
            position: nalgebra::Vector3::new(x, y, z), // Updated to use Vector3
        })
    }

    /// Parse an element from a line in the Gmsh file
    fn parse_element(line: &str) -> Result<Element, io::Error> {
        let mut split = line.split_whitespace();

        let id: u32 = Self::parse_next(&mut split, "Missing element ID")?;
        let element_type: u32 = Self::parse_next(&mut split, "Missing element type")?;

        // Skip physical and geometrical tags (not needed in this case)
        let _num_tags: u32 = Self::parse_next(&mut split, "Missing number of tags")?;
        let _tags: u32 = Self::parse_next(&mut split, "Missing tag data")?;

        // Parse node IDs for the element
        let node_ids: Vec<usize> = split
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok(Element {
            id,
            nodes: node_ids,
            element_type: element_type,
            ..Element::default()
        })
    }

    /// Utility function to parse the next value from an iterator
    fn parse_next<'a, T: std::str::FromStr, I: Iterator<Item = &'a str>>(
        iter: &mut I,
        err_msg: &str,
    ) -> Result<T, io::Error>
    where
        T::Err: std::fmt::Debug,
    {
        iter.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, err_msg))?
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, err_msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node() {
        let line = "1 0.0 1.0 2.0";
        let node = GmshParser::parse_node(line).unwrap();
        assert_eq!(node.id, 1);
        assert_eq!(node.position, nalgebra::Vector3::new(0.0, 1.0, 2.0));
    }

    #[test]
    fn test_parse_element() {
        let line = "1 3 4 9 5 6 7";
        let element = GmshParser::parse_element(line).unwrap();
        assert_eq!(element.id, 1);
        assert_eq!(element.nodes, vec![5, 6, 7]);
        assert_eq!(element.element_type, 3);
        
    }

    #[test]
    fn test_circle_mesh_import() {
        let temp_file_path = "inputs/circular_lake.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();

        // Test for basic validity
        assert!(!nodes.is_empty(), "Circle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Circle mesh elements should not be empty");

        // Check that the number of nodes matches the expected value
        assert_eq!(nodes.len(), 424, "Incorrect number of nodes in circle mesh");
        assert_eq!(elements.len(), 849, "Incorrect number of elements in circle mesh");
    }

    #[test]
    fn test_coastal_island_mesh_import() {
        let temp_file_path = "inputs/coastal_island.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Coastal Island mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Coastal Island mesh elements should not be empty");

        // Add specific tests based on expected structure
        assert_eq!(nodes.len(), 1075, "Incorrect number of nodes in Coastal Island mesh");
        assert_eq!(elements.len(), 2154, "Incorrect number of elements in Coastal Island mesh");
    }

    #[test]
    fn test_lagoon_mesh_import() {
        let temp_file_path = "inputs/elliptical_lagoon.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Lagoon mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Lagoon mesh elements should not be empty");

        // Further checks on expected properties
        assert_eq!(nodes.len(), 848, "Incorrect number of nodes in Lagoon mesh");
        assert_eq!(elements.len(), 1697, "Incorrect number of elements in Lagoon mesh");
    }

    #[test]
    fn test_meandering_river_mesh_import() {
        let temp_file_path = "inputs/meandering_river.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();

        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Meandering River mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Meandering River mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 695, "Incorrect number of nodes in Meandering River mesh");
        assert_eq!(elements.len(), 1386, "Incorrect number of elements in Meandering River mesh");
    }

    #[test]
    fn test_polygon_estuary_mesh_import() {
        let temp_file_path = "inputs/polygon_estuary.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();
        
        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Polygon Estuary mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Polygon Estuary mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 469, "Incorrect number of nodes in Polygon Estuary mesh");
        assert_eq!(elements.len(), 941, "Incorrect number of elements in Polygon Estuary mesh");
    }

    #[test]
    fn test_rectangle_mesh_import() {
        let temp_file_path = "inputs/rectangle.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();
        
        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangle mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangle mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 78, "Incorrect number of nodes in Rectangle mesh");
        assert_eq!(elements.len(), 158, "Incorrect number of elements in Rectangle mesh");
    }

    #[test]
    fn test_rectangle_channel_mesh_import() {
        let temp_file_path = "inputs/rectangular_channel.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();
        
        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Rectangular Channel mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Rectangular Channel mesh elements should not be empty");

        // Further checks on the structure
        assert_eq!(nodes.len(), 149, "Incorrect number of nodes in Rectangular Channel mesh");
        assert_eq!(elements.len(), 300, "Incorrect number of elements in Rectangular Channel mesh");
    }

    #[test]
    fn test_triangle_basin_mesh_import() {
        let temp_file_path = "inputs/triangular_basin.msh2";

        let result = GmshParser::load_mesh(temp_file_path);
        assert!(result.is_ok());

        let (nodes, elements, _) = result.unwrap();
        
        // Validate the mesh structure
        assert!(!nodes.is_empty(), "Triangular Basin mesh nodes should not be empty");
        assert!(!elements.is_empty(), "Triangular Basin mesh elements should not be empty");


        // Further checks on the structure
        assert_eq!(nodes.len(), 66, "Incorrect number of nodes in Triangular Basin mesh");
        assert_eq!(elements.len(), 133, "Incorrect number of elements in Triangular Basin mesh");
    }
}
