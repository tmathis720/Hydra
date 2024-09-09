// mesh_mod.rs
#[derive(Debug)]
pub struct Cell {
    pub volume: f64,
    pub density: f64,
    pub momentum_x: f64,
    pub flux_left: f64,
    pub flux_right: f64,
}

impl Cell {
    // Returns the mass in the cell
    pub fn mass(&self) -> f64 {
        self.density * self.volume
    }

    // Returns the momentum in the cell
    pub fn momentum(&self) -> f64 {
        self.momentum_x
    }

    // Update mass in the cell using FVM principles
    pub fn update_mass(&mut self, dt: f64) {
        let net_flux = self.flux_right - self.flux_left;
        self.density += net_flux * dt / self.volume;
    }

    // Update momentum in the cell using FVM principles
    pub fn update_momentum(&mut self, dt: f64) {
        let net_flux = self.flux_right - self.flux_left;

        // Recalculate velocity based on current momentum and mass
        let velocity = if self.density > 0.0 {
            self.momentum_x / self.density
        } else {
            0.0  // Prevent division by zero
        };

        // Update momentum based on mass flux and velocity
        self.momentum_x += net_flux * velocity * dt / self.volume;
    }

    // Function to compute total mass in the domain
    pub fn total_mass(cells: &[Cell]) -> f64 {
        cells.iter().map(|cell| cell.mass()).sum()
    }

    // Function to compute total momentum in the domain
    pub fn total_momentum(cells: &[Cell]) -> f64 {
        cells.iter().map(|cell| cell.momentum()).sum()
    }

}

// Function to enforce exact mass conservation
pub fn enforce_mass_conservation(cells: &mut [Cell], expected_mass: f64) {
    let current_total_mass = Cell::total_mass(cells);
    let correction_factor = expected_mass / current_total_mass;

    // Apply a correction factor to each cell to ensure the total mass is correct
    for cell in cells.iter_mut() {
        cell.density *= correction_factor;
    }
}

// Function to initialize the domain with uniform cells
pub fn initialize_domain(num_cells: usize, length: f64, velocity: f64) -> Vec<Cell> {
    let cell_size = length / num_cells as f64;
    let mut cells = Vec::with_capacity(num_cells);
    
    for _ in 0..num_cells {
        let density = 1.0;  // Initial density
        let momentum_x = density * velocity;  // Momentum is mass * velocity
        cells.push(Cell {
            volume: cell_size,
            density,
            momentum_x,
            flux_left: 0.0,
            flux_right: 0.0,
        });
    }

    cells
}
