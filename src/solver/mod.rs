pub mod scalar;
pub use scalar::ScalarTransportSolver;
pub mod turbulence;
pub use turbulence::EddyViscositySolver;

use crate::domain::{Face, Element};

pub struct FlowField {
    pub elements: Vec<Element>,
    pub initial_mass: f64,
}

impl FlowField {
    pub fn new(elements: Vec<Element>) -> Self {
        let initial_mass: f64 = elements.iter().map(|e| e.calculate_mass()).sum();
        Self {
            elements,
            initial_mass,
        }
    }

    // Computes the mass for a specific element
    pub fn compute_mass(&self, element: &Element) -> f64 {
        // You can calculate the mass using the velocity, pressure, and other physical properties
        element.mass // For now, returning the mass of the element
    }

    // Compute density based on element's mass and area (or volume)
    pub fn compute_density(&self, element: &Element) -> f64 {
        if element.area > 0.0 {
            element.mass / element.area
        } else {
            0.0 // Avoid division by zero
        }
    }

    // Check mass conservation by summing up the mass and comparing to initial mass
    pub fn check_mass_conservation(&self) -> bool {
        let current_mass: f64 = self.elements.iter().map(|e| e.mass).sum();
        let tolerance = 1e-6; // Define a mass conservation tolerance
        (current_mass - self.initial_mass).abs() < tolerance
    }

    // Placeholder for computing surface velocity, could depend on a free surface element
    pub fn get_surface_velocity(&self) -> f64 {
        0.0  // Add logic to calculate based on a free surface element
    }

    // Returns surface pressure, could depend on atmospheric or fluid surface
    pub fn get_surface_pressure(&self) -> f64 {
        101325.0  // Example: Standard atmospheric pressure (in Pascals)
    }

    // Get inflow velocity for the boundary condition (can be refined further)
    pub fn get_inflow_velocity(&self) -> (f64, f64, f64) {
        (1.0, 0.0, 0.0)  // Example: horizontal inflow velocity
    }

    // Get outflow velocity for the boundary condition
    pub fn get_outflow_velocity(&self) -> (f64, f64, f64) {
        (0.5, 0.0, 0.0)  // Example: horizontal outflow velocity
    }

    // Define inflow mass rate, dependent on the physical properties of the system
    pub fn inflow_mass_rate(&self) -> f64 {
        1.0  // Example: inflow mass rate (this can vary)
    }

    // Define outflow mass rate, this can vary depending on boundary conditions
    pub fn outflow_mass_rate(&self) -> f64 {
        0.8  // Example: outflow mass rate
    }

    // Retrieve a corresponding periodic boundary element (for Periodic boundary condition)
    pub fn get_periodic_element<'a>(&'a self, element: &'a Element) -> &'a Element {
        // Logic to find the periodic counterpart of the current element
        // Currently returns the same element as a placeholder
        element
    }

    // Logic to compute nearby pressure, assuming you're computing pressure using neighbors
    pub fn get_nearby_pressure(&self, element: &Element) -> f64 {
        // Implement logic to compute the pressure of neighboring elements
        let mut total_pressure = 0.0;
        let mut count = 0;

        for neighbor in &self.elements {
            if neighbor.id != element.id {
                total_pressure += neighbor.pressure;
                count += 1;
            }
        }

        if count > 0 {
            total_pressure / count as f64
        } else {
            0.0 // Return zero if there are no neighbors (edge case)
        }
    }
}


pub trait Solver {
    fn compute_flux(&self, face: &Face, left_element: &Element, right_element: &Element) -> f64;
    fn apply_flux(&self, face: &mut Face, flux: f64, dt: f64);
}

pub struct FluxSolver;

impl FluxSolver {
    // Compute the flux for a face based on the left and right elements
    pub fn compute_flux(&self, face: &Face, left_element: &Element, right_element: &Element) -> f64 {
        // Compute the pressure difference
        let pressure_diff = left_element.pressure - right_element.pressure;

        // Determine the flux based on the pressure difference
        let flux = pressure_diff * face.area;

        // Debugging output for clarity
        if flux > 0.0 {
            println!("Flux is positive (flow from left to right): {}", flux);
        } else if flux < 0.0 {
            println!("Flux is negative (flow from right to left): {}", flux);
        } else {
            println!("No flux (equal pressures): {}", flux);
        }

        // Return the computed flux
        flux
    }

    /// Compute flux for no-slip boundary (should always be zero)
    pub fn compute_flux_no_slip(&self, _element: &Element) -> f64 {
        0.0 // No-slip boundary, no flux
    }

    /// Compute flux at the free surface boundary
    pub fn compute_flux_free_surface(&self, element: &Element, surface_pressure: f64) -> f64 {
        // Flux is driven by the pressure difference between the element and the free surface
        let pressure_diff = element.pressure - surface_pressure;
        pressure_diff * element.faces[0] as f64 // Multiply by face area (assuming the first face)
    }

    // Apply the computed flux to update the face's state
    pub fn apply_flux(&self, face: &mut Face, flux: f64, dt: f64) {
        // Example logic: adjust the velocity of the face based on the flux and time step
        face.velocity.0 += flux * dt; // Update the x-component of velocity
        face.velocity.1 += flux * dt; // Update the y-component of velocity
    }
}


impl Solver for FluxSolver {
    fn compute_flux(&self, face: &Face, left_element: &Element, right_element: &Element) -> f64 {
        // Call the existing `compute_flux` method
        self.compute_flux(face, left_element, right_element)
    }

    fn apply_flux(&self, face: &mut Face, flux: f64, dt: f64) {
        // Call the existing `apply_flux` method
        self.apply_flux(face, flux, dt)
    }
}

pub struct FluxLimiter;

impl FluxLimiter {
    /// Superbee flux limiter function to prevent unphysical oscillations
    ///
    /// This function limits the flux to ensure that no new extrema are introduced
    /// into the system, preventing overshoots and undershoots.
    pub fn superbee_limiter(r: f64) -> f64 {
        r.max(0.0).min(2.0).max(r.min(1.0)) // Superbee limiter formula
    }

    /// Apply the flux limiter to the computed flux
    ///
    /// `flux`: The computed flux between elements
    /// `left_flux`: The flux from the left element
    /// `right_flux`: The flux from the right element
    pub fn apply_limiter(&self, flux: f64, left_flux: f64, right_flux: f64) -> f64 {
        let r = right_flux / left_flux;
        let phi = FluxLimiter::superbee_limiter(r);
        phi * flux
    }
}

pub struct SemiImplicitSolver;

impl SemiImplicitSolver {
    /// Semi-implicit time integration for updating momentum
    ///
    /// `flux`: The computed flux between elements
    /// `current_value`: The current momentum or other property being updated
    /// `dt`: Time step size
    ///
    /// The implicit part ensures stability by including current_value in the denominator,
    /// which damps rapid changes in the system and prevents overshoots.
    pub fn semi_implicit_update(&self, flux: f64, current_value: f64, dt: f64) -> f64 {
        let explicit_term = flux * dt;
        let implicit_term = current_value / (1.0 + dt); // Implicit damping term
        implicit_term + explicit_term
    }
}

pub struct CrankNicolsonSolver;

impl CrankNicolsonSolver {
    pub fn crank_nicolson_update(&self, flux: f64, current_value: f64, dt: f64) -> f64 {
        // Crank-Nicolson update: combines implicit and explicit terms
        let explicit_term = 0.5 * flux * dt;
        let implicit_term = current_value / (1.0 + 0.5 * dt);
        let new_value = implicit_term + explicit_term;

        // Prevent the new value from going below zero
        new_value.max(0.0)
    }
}
