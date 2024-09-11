use crate::domain::Face;
use crate::domain::Element;
use crate::domain::Node;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseIntError;

pub struct GmshParser;

impl GmshParser {
    // Load the mesh from a Gmsh file
    pub fn load_mesh(file_path: &str) -> Result<(Vec<Node>, Vec<Element>, Vec<Face>), io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let faces = Vec::new();
        let mut in_nodes_section = false;
        let mut in_elements_section = false;
        let mut node_count = 0;
        let mut element_count = 0;
        let mut current_node_line = 0;
        let mut current_element_line = 0;

        for line in reader.lines() {
            let line = line?;

            // Detect sections
            if line.starts_with("$Nodes") {
                in_nodes_section = true;
                in_elements_section = false;
                current_node_line = 0; // Reset the node line counter
                continue;
            } else if line.starts_with("$Elements") {
                in_nodes_section = false;
                in_elements_section = true;
                current_element_line = 0; // Reset the element line counter
                continue;
            } else if line.starts_with("$EndNodes") || line.starts_with("$EndElements") {
                in_nodes_section = false;
                in_elements_section = false;
                continue;
            }

            // Parse the number of nodes and elements from the next line
            if in_nodes_section && current_node_line == 0 {
                node_count = line.parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid node count"))?;
                current_node_line += 1;
                continue;
            }

            if in_elements_section && current_element_line == 0 {
                element_count = line.parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid element count"))?;
                current_element_line += 1;
                continue;
            }

            // Parse nodes
            if in_nodes_section && current_node_line <= node_count {
                nodes.push(Self::parse_node(&line)?);
                current_node_line += 1;
            }

            // Parse elements (faces included here)
            if in_elements_section && current_element_line <= element_count {
                elements.push(Self::parse_element(&line)?);
                current_element_line += 1;
            }
        }

        Ok((nodes, elements, faces))
    }

    // Parse a single node line
    fn parse_node(line: &str) -> Result<Node, io::Error> {
        let mut split = line.split_whitespace();

        // Parse the node id and its coordinates
        let id: u32 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing node ID"))?
            .parse()
            .map_err(|err: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let x: f64 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing x coordinate"))?
            .parse()
            .map_err(|err: std::num::ParseFloatError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let y: f64 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing y coordinate"))?
            .parse()
            .map_err(|err: std::num::ParseFloatError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let z: f64 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing z coordinate"))?
            .parse()
            .map_err(|err: std::num::ParseFloatError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok(Node { id, position: (x, y, z) })  // Fixed to use (x, y, z)
    }

    // Parse a single element line
    fn parse_element(line: &str) -> Result<Element, io::Error> {
        let mut split = line.split_whitespace();

        // Parse the element id, type, and its node references
        let id: u32 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing element ID"))?
            .parse()
            .map_err(|err: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let element_type: u32 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing element type"))?
            .parse()
            .map_err(|err: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        // Skip physical and geometrical tags (can be extended as needed)
        let _num_tags: u32 = split.next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing tag data"))?
            .parse()
            .map_err(|err: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, err))?;

        // Parse node ids associated with the element
        let node_ids: Vec<usize> = split.map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        Ok(Element {
            id,
            nodes: node_ids,
            faces: vec![],
            pressure: 0.0,
            neighbor_ref: 0,
            mass: 0.0,
            element_type,
            momentum: 0.0,
        })
    }
}