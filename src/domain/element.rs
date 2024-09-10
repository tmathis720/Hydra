// mesh_mod.rs
#[derive(Debug)]
pub struct Element {
    pub volume: f64,
    pub density: f64,
    pub momentum_x: f64,
    pub flux_left: f64,
    pub flux_right: f64,
}

impl Element {
    // Returns the mass in the element
    pub fn mass(&self) -> f64 {
        self.density * self.volume
    }

    // Returns the momentum in the element
    pub fn momentum(&self) -> f64 {
        self.momentum_x
    }

    // Update mass in the element using FVM principles
    pub fn update_mass(&mut self, dt: f64) {
        let net_flux = self.flux_right - self.flux_left;
        self.density += net_flux * dt / self.volume;
    }

    // Update momentum in the element using FVM principles
    pub fn update_momentum(&mut self, dt: f64) {
        let net_flux = self.flux_right - self.flux_left;

        // Calculate velocity before updating momentum to maintain consistency
        let velocity = if self.density > 0.0 {
            self.momentum_x / self.density
        } else {
            0.0  // Prevent division by zero
        };

        // Update momentum based on mass flux and velocity
        self.momentum_x += net_flux * velocity * dt / self.volume;
    }

    // Function to compute total mass in the domain
    pub fn total_mass(elements: &[Element]) -> f64 {
        elements.iter().map(|element| element.mass()).sum()
    }

    // Function to compute total momentum in the domain
    pub fn total_momentum(elements: &[Element]) -> f64 {
        elements.iter().map(|element| element.momentum()).sum()
    }



    // Function to enforce exact mass conservation
    pub fn enforce_mass_conservation(elements: &mut [Element], expected_mass: f64) {
        let current_total_mass = Element::total_mass(elements);
        let correction_factor = expected_mass / current_total_mass;

        // Apply a correction factor to each element to ensure the total mass is correct
        for element in elements.iter_mut() {
            element.density *= correction_factor;
        }
    }

    // Function to enforce exact momentum conservation
    pub fn enforce_momentum_conservation(elements: &mut [Element], expected_momentum: f64) {
        let current_total_momentum = Element::total_momentum(elements);
        let correction_factor = expected_momentum / current_total_momentum;

        // Apply a correction factor to each element's momentum to ensure total momentum is correct
        for element in elements.iter_mut() {
            element.momentum_x *= correction_factor;
        }
    }

    // Function to initialize the domain with uniform elements
    pub fn initialize_domain(num_elements: usize, length: f64, velocity: f64) -> Vec<Element> {
        let element_size = length / num_elements as f64;
        let mut elements = Vec::with_capacity(num_elements);
        
        for _ in 0..num_elements {
            let density = 1.0;  // Initial density
            let momentum_x = density * velocity;  // Momentum is mass * velocity
            elements.push(Element {
                volume: element_size,
                density,
                momentum_x,
                flux_left: 0.0,
                flux_right: 0.0,
            });
        }

        elements
    }
}