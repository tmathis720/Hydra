pub mod scalar_transport;
pub use scalar_transport::ScalarTransportSolver;

use crate::domain::face::Face;
use crate::domain::element::Element;

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

    // Apply the computed flux to update the face's state
    pub fn apply_flux(&self, face: &mut Face, flux: f64, dt: f64) {
        // Example logic: adjust the velocity of the face based on the flux and time step
        face.velocity.0 += flux * dt; // Update the x-component of velocity
        face.velocity.1 += flux * dt; // Update the y-component of velocity
    }
}
