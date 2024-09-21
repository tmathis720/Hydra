// src/domain/element.rs

use crate::domain::{Node, Mesh};
use nalgebra::Vector3;
use std::error::Error;

/// Represents an element in the simulation domain, such as a cell in a mesh.
/// Stores physical properties and provides methods to compute derived quantities.
#[derive(Clone, Debug)]
pub struct Element {
    /// Unique identifier for the element.
    pub id: u32,

    /// Indices of nodes that define the element.
    pub nodes: Vec<usize>,

    /// Centroid coordinates for the element, based on the node position
    pub centroid_coordinates: Vec<f64>,

    /// Indices of faces that bound the element.
    pub faces: Vec<u32>,

    /// Pressure at the element (units: Pascals).
    pub pressure: f64,

    /// Height or depth of the element (units: meters).
    pub height: f64,

    /// Area of the element (units: square meters for 2D elements).
    /// For 3D elements, this may represent volume.
    pub area: f64,

    /// References to neighboring elements (indices).
    pub neighbor_refs: Vec<usize>,

    /// Distance to neighboring elements.
    pub neighbor_distance: Vec<f64>,

    /// Mass of the element (units: kilograms).
    pub mass: f64,

    /// Type identifier for the element (e.g., triangle, quadrilateral).
    pub element_type: u32,

    /// 3D momentum vector of the element (units: kg·m/s).
    pub momentum: Vector3<f64>,

    /// 3D velocity vector of the element (units: m/s).
    pub velocity: Vector3<f64>,

    /// Optional element-specific viscosity (units: Pa·s).
    pub laminar_viscosity: Option<f64>,
}

impl Element {
    /// Creates a new `Element` with the given parameters.
    ///
    /// # Parameters
    /// - `id`: Unique identifier for the element.
    /// - `nodes`: Indices of nodes that define the element.
    /// - `faces`: Indices of faces that bound the element.
    /// - `element_type`: Type identifier for the element.
    ///
    /// # Returns
    /// A new `Element` instance with default physical properties.
    pub fn new(id: u32, nodes: Vec<usize>, faces: Vec<u32>, element_type: u32) -> Self {
        Self {
            id,
            nodes,
            faces,
            element_type,
            ..Element::default()
        }
    }

    /// Checks if the element contains a specific node by its index.
    ///
    /// # Parameters
    /// - `node_index`: Index of the node to check.
    ///
    /// # Returns
    /// `true` if the element contains the node, `false` otherwise.
    pub fn has_node(&self, node_index: usize) -> bool {
        self.nodes.contains(&node_index)
    }

    /// Computes the centroid of the element based on its nodes' positions.
    ///
    /// This function assumes the positions of the nodes are stored in the mesh.
    /// 
    /// # Parameters
    /// - `mesh`: A reference to the `Mesh` where node positions are stored.
    ///
    /// # Returns
    /// A `Vec<f64>` representing the centroid coordinates of the element.
    pub fn compute_centroid(&self, mesh: &Mesh) -> Vec<f64> {
        let mut centroid = vec![];  // Assuming 2D or 3D space
        
        for &node_id in &self.nodes {
            if let Some(node) = mesh.get_node_by_id(node_id.try_into().unwrap()) {
                for i in 0..centroid.len() {
                    centroid[i] += node.position[i];
                }
            }
        }
        
        // Divide by the number of nodes to compute the average (centroid)
        for i in 0..centroid.len() {
            centroid[i] /= self.nodes.len() as f64;
        }
        
        centroid
    }

    /// Computes the distances between this element and its neighbors.
    ///
    /// This function computes the Euclidean distance between the centroids of the
    /// element and each of its neighboring elements, and stores these distances in the
    /// `neighbor_distance` vector.
    ///
    /// # Parameters
    /// - `mesh`: A reference to the `Mesh` for accessing neighboring element information.
    pub fn compute_neighbor_distances(&mut self, mesh: &Mesh) {
        // First, compute the centroid of this element
        let centroid = self.compute_centroid(mesh);

        // Clear previous distance data
        self.neighbor_distance.clear();

        // Iterate over neighbor_refs and compute distance to each
        for &neighbor_id in &self.neighbor_refs {
            if let Some(neighbor) = mesh.get_element_by_id(neighbor_id.try_into().unwrap()) {
                let neighbor_centroid = neighbor.compute_centroid(mesh);

                // Compute Euclidean distance between the centroids
                let distance = Self::euclidean_distance(&centroid, &neighbor_centroid);
                self.neighbor_distance.push(distance);
            }
        }
    }

    /// Helper function to compute the Euclidean distance between two points in n-dimensional space.
    ///
    /// # Parameters
    /// - `point1`: The first point as a reference to a `Vec<f64>`.
    /// - `point2`: The second point as a reference to a `Vec<f64>`.
    ///
    /// # Returns
    /// The Euclidean distance between the two points as `f64`.
    fn euclidean_distance(point1: &Vec<f64>, point2: &Vec<f64>) -> f64 {
        point1.iter().zip(point2.iter())
            .map(|(x1, x2)| (x2 - x1).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculates the area (2D) or volume (3D) of the element based on its nodes' positions.
    ///
    /// # Parameters
    /// - `nodes`: Slice of `Node` instances representing all nodes in the mesh.
    ///
    /// # Returns
    /// `Ok(area_or_volume)` if the calculation is successful.
    /// `Err` if the element geometry is unsupported or if an error occurs.
    pub fn calculate_area(&mut self, nodes: &[Node]) -> Result<f64, Box<dyn Error>> {
        let positions: Vec<Vector3<f64>> = self
            .nodes
            .iter()
            .map(|&i| nodes.get(i).ok_or("Node index out of bounds").map(|node| node.position))
            .collect::<Result<Vec<_>, _>>()?;

        match positions.len() {
            3 => {
                // Triangular element area (2D)
                let area = 0.5
                    * (positions[1] - positions[0])
                        .cross(&(positions[2] - positions[0]))
                        .norm();

                if area > 0.0 {
                    self.area = area;
                    Ok(area)
                } else {
                    Err("Calculated area is non-positive".into())
                }
            }
            4 => {
                // Quadrilateral element area (split into two triangles)
                let area1 = 0.5
                    * (positions[1] - positions[0])
                        .cross(&(positions[2] - positions[0]))
                        .norm();
                let area2 = 0.5
                    * (positions[2] - positions[0])
                        .cross(&(positions[3] - positions[0]))
                        .norm();

                let total_area = area1 + area2;

                if total_area > 0.0 {
                    self.area = total_area;
                    Ok(total_area)
                } else {
                    Err("Calculated area is non-positive".into())
                }
            }
            _ => Err("Unsupported element geometry for area calculation".into()),
        }
    }

    /// Calculates the mass of the element based on its density and volume.
    ///
    /// # Parameters
    /// - `density`: Density of the material (units: kg/m³).
    ///
    /// # Returns
    /// Updates the element's mass and returns `Ok(mass)` if successful.
    /// Returns `Err` if volume is non-positive.
    pub fn calculate_mass(&mut self, density: f64) -> Result<f64, Box<dyn Error>> {
        let volume = self.compute_volume();

        if volume > 0.0 {
            self.mass = density * volume;
            Ok(self.mass)
        } else {
            Err("Element volume is non-positive".into())
        }
    }

    /// Computes the volume of the element.
    ///
    /// # Returns
    /// The volume of the element (units: cubic meters).
    /// For 2D elements, this may be the area times height.
    pub fn compute_volume(&self) -> f64 {
        self.area * self.height
    }

    /// Computes the density of the element.
    ///
    /// # Returns
    /// `Some(density)` if volume is positive.
    /// `None` if volume is zero or negative.
    pub fn compute_density(&self) -> Option<f64> {
        let volume = self.compute_volume();
        if volume > 0.0 {
            Some(self.mass / volume)
        } else {
            None
        }
    }

    /// Computes the velocity of the element based on its momentum and mass.
    ///
    /// # Returns
    /// `Ok(velocity)` if mass is positive.
    /// `Err` if mass is zero or negative.
    pub fn compute_velocity(&mut self) -> Result<Vector3<f64>, Box<dyn Error>> {
        if self.mass > 0.0 {
            self.velocity = self.momentum / self.mass;
            Ok(self.velocity)
        } else {
            Err("Element mass is non-positive; cannot compute velocity".into())
        }
    }

    /// Updates the momentum of the element by adding the given change (delta) in momentum.
    ///
    /// # Parameters
    /// - `delta_momentum`: Change in momentum to be added (units: kg·m/s).
    pub fn update_momentum(&mut self, delta_momentum: Vector3<f64>) {
        self.momentum += delta_momentum;
    }

    /// Updates the element's velocity based on its current momentum and mass.
    ///
    /// # Returns
    /// `Ok(velocity)` if mass is positive.
    /// `Err` if mass is zero or negative.
    pub fn update_velocity_from_momentum(&mut self) -> Result<Vector3<f64>, Box<dyn Error>> {
        self.compute_velocity()
    }

    /// Computes the kinetic energy of the element.
    ///
    /// # Returns
    /// The kinetic energy (units: Joules) of the element.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.norm_squared()
    }

    /// Updates the element's pressure.
    ///
    /// # Parameters
    /// - `new_pressure`: New pressure value to set (units: Pascals).
    pub fn update_pressure(&mut self, new_pressure: f64) {
        self.pressure = new_pressure;
    }

    /// Adds mass to the element.
    ///
    /// # Parameters
    /// - `additional_mass`: Mass to add (units: kilograms).
    pub fn add_mass(&mut self, additional_mass: f64) {
        self.mass += additional_mass;
    }

    /// Removes mass from the element, ensuring mass remains non-negative.
    ///
    /// # Parameters
    /// - `mass_to_remove`: Mass to remove (units: kilograms).
    pub fn remove_mass(&mut self, mass_to_remove: f64) {
        self.mass = (self.mass - mass_to_remove).max(0.0);
    }

    /// Adds mass and momentum to the element.
    ///
    /// # Parameters
    /// - `mass_delta`: Change in mass (units: kilograms).
    /// - `momentum_delta`: Change in momentum (units: kg·m/s).
    pub fn add_mass_and_momentum(&mut self, mass_delta: f64, momentum_delta: Vector3<f64>) {
        self.mass += mass_delta;
        self.momentum += momentum_delta;
    }

    /// Subtracts mass and momentum from the element, ensuring mass remains non-negative.
    ///
    /// # Parameters
    /// - `mass_delta`: Change in mass (units: kilograms).
    /// - `momentum_delta`: Change in momentum (units: kg·m/s).
    pub fn subtract_mass_and_momentum(&mut self, mass_delta: f64, momentum_delta: Vector3<f64>) {
        self.mass = (self.mass - mass_delta).max(0.0);
        self.momentum -= momentum_delta;
    }

    /// Sets the velocity of the element.
    ///
    /// # Parameters
    /// - `velocity`: New velocity vector (units: m/s).
    pub fn set_velocity(&mut self, velocity: Vector3<f64>) {
        self.velocity = velocity;
    }

    /// Resets the element's properties to default values.
    pub fn reset_properties(&mut self) {
        self.momentum = Vector3::zeros();
        self.velocity = Vector3::zeros();
        self.pressure = 0.0;
        self.mass = 0.0;
        self.height = 0.0;
        self.area = 0.0;
    }

    /// Applies the no-slip condition by setting momentum and velocity to zero.
    pub fn apply_no_slip(&mut self) {
        self.momentum = Vector3::zeros();
        self.velocity = Vector3::zeros();
    }

    /// Generates the faces (edges for 2D elements) of the element.
    /// Returns a vector of faces, where each face is represented by a vector of node indices.
    pub fn generate_faces(&self) -> Vec<Vec<usize>> {
        let mut faces = Vec::new();
        let num_nodes = self.nodes.len();

        for i in 0..num_nodes {
            let face_nodes = vec![
                self.nodes[i],
                self.nodes[(i + 1) % num_nodes], // Wrap around for the last node
            ];
            faces.push(face_nodes);
        }

        faces
    }
}

impl Default for Element {
    fn default() -> Self {
        Element {
            id: 0,
            nodes: vec![],
            centroid_coordinates: vec![],
            faces: vec![],
            pressure: 0.0,
            height: 0.0,
            area: 0.0,
            neighbor_refs: vec![],
            neighbor_distance: vec![],
            mass: 0.0,
            element_type: 0,
            momentum: Vector3::zeros(),
            velocity: Vector3::zeros(),
            laminar_viscosity: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Node;
    use nalgebra::Vector3;

    #[test]
    fn test_area_calculation_triangle() {
        // Create nodes forming a right triangle with area 0.5
        let nodes = vec![
            Node {
                id: 0,
                position: Vector3::new(0.0, 0.0, 0.0),
            },
            Node {
                id: 1,
                position: Vector3::new(1.0, 0.0, 0.0),
            },
            Node {
                id: 2,
                position: Vector3::new(0.0, 1.0, 0.0),
            },
        ];

        let mut element = Element::new(0, vec![0, 1, 2], vec![], 0);
        let area = element.calculate_area(&nodes).unwrap();

        assert!((area - 0.5).abs() < 1e-6, "Area should be 0.5");
    }

    #[test]
    fn test_area_calculation_quadrilateral() {
        // Create nodes forming a square with area 1.0
        let nodes = vec![
            Node {
                id: 0,
                position: Vector3::new(0.0, 0.0, 0.0),
            },
            Node {
                id: 1,
                position: Vector3::new(1.0, 0.0, 0.0),
            },
            Node {
                id: 2,
                position: Vector3::new(1.0, 1.0, 0.0),
            },
            Node {
                id: 3,
                position: Vector3::new(0.0, 1.0, 0.0),
            },
        ];

        let mut element = Element::new(0, vec![0, 1, 2, 3], vec![], 0);
        let area = element.calculate_area(&nodes).unwrap();

        assert!((area - 1.0).abs() < 1e-6, "Area should be 1.0");
    }

    #[test]
    fn test_compute_velocity() {
        let mut element = Element {
            mass: 2.0,
            momentum: Vector3::new(4.0, 0.0, 0.0),
            ..Default::default()
        };

        let velocity = element.compute_velocity().unwrap();
        assert_eq!(velocity, Vector3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_compute_velocity_zero_mass() {
        let mut element = Element {
            mass: 0.0,
            momentum: Vector3::new(4.0, 0.0, 0.0),
            ..Default::default()
        };

        let result = element.compute_velocity();
        assert!(result.is_err(), "Should return an error due to zero mass");
    }

    #[test]
    fn test_compute_density() {
        let element = Element {
            mass: 2.0,
            area: 1.0,
            height: 1.0,
            ..Default::default()
        };

        let density = element.compute_density().unwrap();
        assert_eq!(density, 2.0);
    }

    #[test]
    fn test_compute_density_zero_volume() {
        let element = Element {
            mass: 2.0,
            area: 0.0,
            height: 0.0,
            ..Default::default()
        };

        let density = element.compute_density();
        assert!(density.is_none(), "Density should be None due to zero volume");
    }
}
