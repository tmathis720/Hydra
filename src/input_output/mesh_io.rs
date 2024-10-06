use crate::domain::{Mesh, MeshEntity, Sieve};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct GmshParser;

impl GmshParser {
    /// Load the mesh from a Gmsh file
    pub fn from_gmsh_file(file_path: &str) -> Result<Mesh, io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut mesh = Mesh::new();
        let mut sieve = Sieve::new();
        let mut node_coords = Vec::<f64>::new();

        let mut in_nodes_section = false;
        let mut in_elements_section = false;

        let mut node_count = 0;
        let mut element_count = 0;
        let mut current_node_line = 0;
        let mut current_element_line = 0;

        for line in reader.lines() {
            let line = line?;

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
                let (vertex_id, coords) = Self::parse_node(&line)?;
                mesh.set_vertex_coordinates(vertex_id, coords);
                current_node_line += 1;
            }

            // Parse individual elements
            if in_elements_section && current_element_line <= element_count {
                let (element_id, node_ids) = Self::parse_element(&line)?;
                let cell = MeshEntity::Cell(element_id);
                mesh.add_entity(cell.clone());

                // Add relationships between the cell and its vertices
                for node_id in node_ids {
                    let vertex = MeshEntity::Vertex(node_id);
                    mesh.add_relationship(cell.clone(), vertex);
                }
                current_element_line += 1;
            }
        }

        Ok(mesh)
    }

    /// Parse a single node from a line in the Gmsh file
    fn parse_node(line: &str) -> Result<(usize, [f64; 3]), io::Error> {
        let mut split = line.split_whitespace();
        let id: usize = Self::parse_next(&mut split, "Missing node ID")?;
        let x: f64 = Self::parse_next(&mut split, "Missing x coordinate")?;
        let y: f64 = Self::parse_next(&mut split, "Missing y coordinate")?;
        let z: f64 = Self::parse_next(&mut split, "Missing z coordinate")?;

        Ok((id, [x, y, z]))
    }

    /// Parse an element from a line in the Gmsh file
    fn parse_element(line: &str) -> Result<(usize, Vec<usize>), io::Error> {
        let mut split = line.split_whitespace();

        let id: usize = Self::parse_next(&mut split, "Missing element ID")?;
        let _element_type: u32 = Self::parse_next(&mut split, "Missing element type")?;

        // Skip physical and geometrical tags
        let _num_tags: u32 = Self::parse_next(&mut split, "Missing number of tags")?;
        let _tags: u32 = Self::parse_next(&mut split, "Missing tag data")?;

        // Parse node IDs for the element
        let node_ids: Vec<usize> = split
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok((id, node_ids))
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
    use crate::input_output::mesh_io::GmshParser;
    use crate::domain::Mesh;

    #[test]
    fn test_parse_node() {
        let line = "1 0.0 1.0 2.0";
        let (id, coords) = GmshParser::parse_node(line).unwrap();
        assert_eq!(id, 1);
        assert_eq!(coords, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_parse_element() {
        let line = "1 3 4 9 5 6 7";
        let (id, nodes) = GmshParser::parse_element(line).unwrap();
        assert_eq!(id, 1);
        assert_eq!(nodes, vec![5, 6, 7]);
    }

    #[test]
    fn test_circle_mesh_import() {
        let temp_file_path = "inputs/circular_lake.msh2";
        let result = GmshParser::from_gmsh_file(temp_file_path);
        assert!(result.is_ok());

        let mesh = result.unwrap();
        assert_eq!(mesh.get_cells().len(), 849);
    }

    // Similar tests for other meshes, e.g., rectangular_channel.msh2, etc.
}
