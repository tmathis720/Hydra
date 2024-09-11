pub struct Face {
    pub id: u32,
    pub nodes: (usize, usize), // The two nodes that make up the face
    pub velocity: (f64, f64),  // Velocity on the face (u, v)
    pub area: f64,             // Face area or length in 2D
}

impl Face {
    // Constructor
    pub fn new(id: u32, nodes: (usize, usize), velocity: (f64, f64), area: f64) -> Self {
        Face { id, nodes, velocity, area }
    }
}

impl Default for Face {
    fn default() -> Self {
        Face {
            id: 0,
            velocity: (0.0, 0.0),
            area: 0.0,
            nodes: (0, 0),
        }
    }
}
