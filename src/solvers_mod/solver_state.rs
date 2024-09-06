/// Solver state management
use crate::mesh_mod::mesh_ops::{Mesh, Element}; 

pub trait SolverState {
    fn update_states(&mut self, dt: f64, fluxes: Vec<f64>);
    fn compute_flux(&self) -> Vec<f64>;
    fn compute_element_flux(&self, element: &Element, neighbor: &Element) -> f64;
}

impl SolverState for Mesh {
    // Update states using the computed fluxes
    fn update_states(&mut self, dt: f64, fluxes: Vec<f64>) {
        for (element, &flux) in self.elements.iter_mut().zip(fluxes.iter()) {
            element.state += dt * flux;  // Update state with flux
        }
    }

    // Compute the flux between elements and store it in a separate vector
    fn compute_flux(&self) -> Vec<f64> {
        let mut fluxes = vec![0.0; self.elements.len()];

        for (i, element) in self.elements.iter().enumerate() {
            let mut total_flux = 0.0;
            for &neighbor_id in &element.neighbors {
                let neighbor = &self.elements[neighbor_id]; // No mutable borrow here
                let flux = self.compute_element_flux(element, neighbor);
                total_flux += flux;
            }
            fluxes[i] = total_flux; // Store the computed fluxes separately
        }

        fluxes
    }
    /*     // Compute the flux between elements and update the state of each element
        pub fn compute_flux(&mut self) {
            for element in &mut self.elements {
                let mut total_flux = 0.0;
                for &neighbor_id in &element.neighbors {
                    let neighbor = &self.elements[neighbor_id];
                    let flux = self.compute_element_flux(element, neighbor);
                    total_flux += flux;
                }
                element.flux = total_flux;  // Accumulate flux for the element
            }
        } */

    // Simple flux computation between two elements (this can be expanded with physics)
    fn compute_element_flux(&self, element: &Element, neighbor: &Element) -> f64 {
        // Example: a simple difference in state to model diffusion
        let diff = element.state - neighbor.state;
        diff * element.area  // Example of using the area in the flux computation
    }
}