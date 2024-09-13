use crate::domain::Node;
use nalgebra::Vector3;  // Use nalgebra's Vector3 for 3D vector operations

#[derive(Clone)]
pub struct Element {
    pub id: u32,
    pub nodes: Vec<usize>,     // Use usize for indexing node indices
    pub faces: Vec<u32>,       // Faces of the element, indexed by face ids
    pub pressure: f64,         // Pressure at the element
    pub height: f64,           // Height of the element (e.g., water depth for free surface flows)
    pub area: f64,             // Area of the element (can be computed from nodes)
    pub neighbor_ref: usize,   // Reference to neighboring elements
    pub mass: f64,             // Mass of the element
    pub element_type: u32,     // Type identifier for the element
    pub momentum: Vector3<f64>, // 3D momentum as Vector3<f64>
    pub velocity: Vector3<f64>, // 3D velocity as Vector3<f64>
    pub laminar_viscosity: Option<f64>, // Optional element specific viscosity
}

impl Element {
    /// Check if the element contains a specific node by its ID
    pub fn has_node(&self, node_id: u32) -> bool {
        self.faces.contains(&node_id)  // Checks if the node is part of the element's face
    }

    /// Calculate the area of the element based on its nodes' positions.
    /// Supports triangular and quadrilateral elements in 2D.
    pub fn calculate_area(&mut self, nodes: &[Node]) -> f64 {
        // Collect node positions as tuples (f64, f64, f64)
        let positions: Vec<(f64, f64, f64)> = self
            .nodes
            .iter()
            .map(|&i| {
                let pos = nodes[i].position;
                (pos.x, pos.y, pos.z)  // Convert Vector3 to tuple (f64, f64, f64)
            })
            .collect();

        // Handle different geometries based on the number of nodes
        match positions.len() {
            3 => {
                // Triangular element area (in 2D, ignore z-component)
                let (x1, y1, _z1) = positions[0];
                let (x2, y2, _z2) = positions[1];
                let (x3, y3, _z3) = positions[2];
                self.area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
                self.area
            }
            4 => {
                // Quadrilateral element area (split into two triangles)
                let (x1, y1, _z1) = positions[0];
                let (x2, y2, _z2) = positions[1];
                let (x3, y3, _z3) = positions[2];
                let (x4, y4, _z4) = positions[3];

                // Split the quad into two triangles and sum their areas
                let triangle_1_area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
                let triangle_2_area = 0.5 * ((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1)).abs();
                self.area = triangle_1_area + triangle_2_area;
                self.area
            }
            _ => {
                // Unsupported geometry (extend for more complex shapes if needed)
                0.0
            }
        }
    }

    /// Calculate the mass of the element based on material properties
    /// This can be extended for more complex mass calculations
    pub fn calculate_mass(&self) -> f64 {
        // Mass calculation logic (currently, just returns stored mass)
        self.mass
    }

    /// Compute the density of the element based on its mass and volume (area * height)
    pub fn compute_density(&self) -> f64 {
        let volume = self.area * self.height;
        if volume > 0.0 {
            self.mass / volume
        } else {
            0.0  // Handle zero or negative volume cases gracefully
        }
    }

    /// Compute the velocity of the element based on its momentum and mass
    /// Updates the element's velocity field
    pub fn compute_velocity(&mut self) {
        if self.mass > 0.0 {
            self.velocity = self.momentum / self.mass;
        } else {
            self.velocity = Vector3::new(0.0, 0.0, 0.0);  // Handle zero-mass cases by resetting velocity to zero
        }
    }

    /// Update the momentum of the element by adding the given change (delta) in momentum
    /// Element-wise update of the 3D momentum vector
    pub fn update_momentum(&mut self, delta_momentum: Vector3<f64>) {
        self.momentum += delta_momentum;
    }

    pub fn update_velocity_from_momentum(&mut self) {
        if self.mass > 0.0 {
            self.velocity = self.momentum / self.mass;
        } else {
            self.velocity = Vector3::new(0.0, 0.0, 0.0);
        }
    }

    /// Compute the kinetic energy of the element based on its velocity and mass
    /// Uses the kinetic energy formula: KE = 1/2 * m * v^2
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.norm_squared()
    }

    /// Update the element's pressure based on boundary conditions or external factors
    pub fn update_pressure(&mut self, new_pressure: f64) {
        self.pressure = new_pressure;
    }

    /// Compute the volume of the element (in 3D), typically the product of the area and height
    pub fn compute_volume(&self) -> f64 {
        self.area * self.height
    }

    /// Add mass to the element
    pub fn add_mass(&mut self, additional_mass: f64) {
        self.mass += additional_mass;
    }

    /// Remove mass from the element (ensure non-negative mass)
    pub fn remove_mass(&mut self, mass_to_remove: f64) {
        self.mass = (self.mass - mass_to_remove).max(0.0);
    }

    /// Reset the element's properties, useful in scenarios where periodic boundary conditions are applied
    pub fn reset_properties(&mut self) {
        self.momentum = Vector3::new(0.0, 0.0, 0.0);
        self.velocity = Vector3::new(0.0, 0.0, 0.0);
        self.pressure = 0.0;
    }
}

// Default implementation for Element, initializing fields to default values
impl Default for Element {
    fn default() -> Self {
        Element {
            id: 0,
            nodes: vec![],
            faces: vec![],
            pressure: 0.0,
            height: 0.0,
            area: 0.0,
            neighbor_ref: 0,
            mass: 0.0,
            element_type: 0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            laminar_viscosity: Some(0.0),
        }
    }
}
