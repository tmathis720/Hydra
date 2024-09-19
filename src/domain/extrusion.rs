// Import the necessary modules from the HYDRA project
use crate::domain::{Mesh, Element, Node};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::error::Error;

// Enum to represent different bottom profile types
pub enum BottomProfile {
    FileBased(Vec<f64>),   // Heights from a file (CSV, raster, etc.)
    FunctionBased(Box<dyn Fn(f64, f64) -> f64>),  // Function-based bottom profile
}

// Enum for different vertical coordinate systems
pub enum VerticalCoordinateSystem {
    Sigma,
    ZLevel,
    Stretched { surface_refinement: f64, bottom_refinement: f64 },
}

// Struct to hold refinement options
pub struct RefinementOptions {
    pub surface_refinement: f64,
    pub bottom_refinement: f64,
}

impl RefinementOptions {
    pub fn default() -> Self {
        RefinementOptions {
            surface_refinement: 1.0,
            bottom_refinement: 1.0,
        }
    }
}

// Function to calculate elevation based on the vertical coordinate system
fn calculate_elevation(
    vertical_coord: &VerticalCoordinateSystem,
    bottom_elevation: f64,
    layer: usize,
    num_layers: usize
) -> f64 {
    match vertical_coord {
        VerticalCoordinateSystem::Sigma => {
            bottom_elevation + (layer as f64 / num_layers as f64) * (0.0 - bottom_elevation)
        },
        VerticalCoordinateSystem::ZLevel => {
            bottom_elevation + (layer as f64 / num_layers as f64) * (0.0 - bottom_elevation)
        },
        VerticalCoordinateSystem::Stretched { surface_refinement, bottom_refinement } => {
            exponential_refinement(layer, num_layers, bottom_elevation, 0.0, *surface_refinement, *bottom_refinement)
        }
    }
}

// Exponential refinement to control layer thickness
fn exponential_refinement(
    layer: usize,
    num_layers: usize,
    bottom_elevation: f64,
    surface_elevation: f64,
    surface_refinement: f64,
    bottom_refinement: f64,
) -> f64 {
    let alpha = 2.0; // Exponential factor
    let dz = (surface_elevation - bottom_elevation) / (num_layers as f64);
    bottom_elevation + dz * (1.0 - (layer as f64 / num_layers as f64).powf(alpha))
}

// Function to parse a bottom profile from a CSV file
fn parse_bottom_profile_from_file(file_path: &str) -> Result<BottomProfile, Box<dyn Error>> {
    let mut heights = Vec::new();
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let fields: Vec<f64> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        if fields.len() == 3 {
            heights.push(fields[2]); // Assuming z-values in the 3rd column
        }
    }

    Ok(BottomProfile::FileBased(heights))
}

// Main function to extrude 2D mesh into 3D
pub fn extrude_2d_to_3d(
    mesh: &Mesh,
    bottom_profile: &BottomProfile,
    vertical_coord: &VerticalCoordinateSystem,
    num_layers: usize,
    refinement_options: &RefinementOptions
) -> Mesh {
    let mut iter_node_id = mesh.nodes.len();
    
    // Step 1: Parallel node extrusion based on bottom profile and vertical coordinate system
    let nodes_3d: Vec<Node> = mesh.nodes.par_iter().flat_map(|node| {
        let bottom_elevation = match bottom_profile {
            BottomProfile::FileBased(heights) => heights[node.id],
            BottomProfile::FunctionBased(f) => f(node.position.x, node.position.y),
        };

        // Create nodes for each layer
        (0..num_layers).map(move |layer| {
            let elevation = calculate_elevation(vertical_coord, bottom_elevation, layer, num_layers);
            let new_node = Node::new(iter_node_id, Vector3::new(node.position.x, node.position.y, elevation));
            iter_node_id += 1;
            new_node
        }).collect::<Vec<_>>()
    }).collect();
    
    // Step 2: Create 3D elements by connecting nodes between layers
    let elements_3d: Vec<Element> = mesh.elements.par_iter().flat_map(|element| {
        let mut new_elements = Vec::new();
        for layer in 0..num_layers - 1 {
            let base_layer_nodes = get_layer_nodes(&nodes_3d, element, layer);
            let top_layer_nodes = get_layer_nodes(&nodes_3d, element, layer + 1);
            new_elements.push(create_prism_element(base_layer_nodes, top_layer_nodes));
        }
        new_elements
    }).collect();

    Mesh { 
        nodes: nodes_3d, 
        elements: elements_3d,
        ..Mesh::default()
     }
}

// Helper function to get nodes in a specific layer
fn get_layer_nodes(nodes_3d: &[Node], element: &Element, layer: usize) -> Vec<Node> {
    element.node_ids.iter().map(|&node_id| {
        nodes_3d[node_id + layer * element.node_ids.len()].clone()
    }).collect()
}

// Helper function to create a 3D prism element from two layers of nodes
fn create_prism_element(base_layer: Vec<Node>, top_layer: Vec<Node>) -> Element {
    // Use your specific logic for creating a prism or 3D element from two layers
    Element::new(vec![], vec![base_layer, top_layer], ..Element::default(), ..Element::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Mesh, Node, Element};

    #[test]
    fn test_flat_bottom_extrusion() {
        // Setup a simple 2D mesh with a flat bottom profile
        let nodes = vec![Node::new(0, Vector3::new(0.0, 0.0, 0.0)), 
                         Node::new(1, Vector3::new(1.0, 0.0, 0.0)), 
                         Node::new(2, Vector3::new(0.0, 1.0, 0.0))];
        let elements = vec![Element::new(1, 
                                                       vec![0, 1, 2],
                                                       ..Element::default(),
                                                       ..Element::default())];
        let mesh_2d = Mesh { nodes, elements, ..Mesh::default() };
        
        // Flat bottom profile
        let bottom_profile = BottomProfile::FunctionBased(Box::new(|_, _| -10.0));
        let vertical_coord = VerticalCoordinateSystem::ZLevel;
        
        // Extrude the 2D mesh into 3D
        let mesh_3d = extrude_2d_to_3d(&mesh_2d, &bottom_profile, &vertical_coord, 5, &RefinementOptions::default());
        
        // Assert that nodes and elements were created correctly
        assert_eq!(mesh_3d.nodes.len(), mesh_2d.nodes.len() * 5); // 5 layers of nodes
        assert_eq!(mesh_3d.elements.len(), mesh_2d.elements.len() * 4); // 4 layers of elements
    }

    #[test]
    fn test_slope_bottom_extrusion() {
        // Setup a simple 2D mesh with a sloped bottom profile
        let nodes = vec![Node::new(0, Vector3::new(0.0, 0.0, 0.0)), 
                         Node::new(1, Vector3::new(1.0, 0.0, 0.0)), 
                         Node::new(2, Vector3::new(0.0, 1.0, 0.0))];
        let elements = vec![Element::new(1, vec![0, 1, 2], ..Element::default(), ..Element::default())];
        let mesh_2d = Mesh { nodes, elements, ..Mesh::default() };

        // Sloped bottom profile
        let bottom_profile = BottomProfile::FunctionBased(Box::new(|x, y| -10.0 + x + y));
        let vertical_coord = VerticalCoordinateSystem::Sigma;
        
        // Extrude the 2D mesh into 3D
        let mesh_3d = extrude_2d_to_3d(&mesh_2d, &bottom_profile, &vertical_coord, 5, &RefinementOptions::default());
        
        // Assert that nodes and elements were created correctly
        assert_eq!(mesh_3d.nodes.len(), mesh_2d.nodes.len() * 5);
        assert_eq!(mesh_3d.elements.len(), mesh_2d.elements.len() * 4);
    }
}
