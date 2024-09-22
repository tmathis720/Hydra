use crate::domain::{Element, Face};
use nalgebra::Vector3;

pub struct FluxTransport {
    pub laminar_viscosity: f64,  // Global laminar viscosity
}

impl FluxTransport {
    pub fn get_laminar_viscosity(&self, element: &Element) -> f64 {
        element.laminar_viscosity.unwrap_or(self.laminar_viscosity)  // Use element-specific if set, else global
    }

    /// Compute convective flux between two elements across a face
    pub fn compute_convective_flux(&self, left_element: &Element, right_element: &Element, face: &Face) -> Vector3<f64> {
        // Calculate convective flux based on the velocity and density differences across the face
        let flux_velocity = (left_element.velocity + right_element.velocity) * 0.5;  // Average velocity across the face
        let flux = flux_velocity * face.area;  // Multiply by the face area to get the flux

        flux
    }

    /// Compute diffusive flux between two elements across a face
    pub fn compute_diffusive_flux(&self, left_element: &Element, right_element: &Element, face: &Face, viscosity: f64) -> Vector3<f64> {
        // Calculate diffusive flux using the gradient of velocity (Fick's law for diffusion)
        let velocity_diff = right_element.velocity - left_element.velocity;
        let flux = viscosity * velocity_diff / face.area;  // Diffusion is proportional to gradient of velocity

        flux
    }

    /// Integrate the fluxes over the control volume (element)
    pub fn integrate_fluxes(&self, element: &mut Element, flux: Vector3<f64>, dt: f64) {
        // Update the element's momentum and velocity using the computed flux
        element.momentum += flux * dt;
        element.velocity = element.momentum / element.mass;
    }

    pub fn compute_scalar_flux(&self, left_scalar: f64, right_scalar: f64, face: &Face) -> f64 {
        let scalar_diff = right_scalar - left_scalar;  // Compute the difference in the scalar field
        let scalar_flux = scalar_diff * face.area;  // Diffusive flux for the scalar field
    
        scalar_flux
    }
    
    pub fn compute_turbulent_diffusive_flux(
        &self, 
        left_element: &Element, 
        right_element: &Element, 
        face: &Face, 
        eddy_viscosity: f64
    ) -> Vector3<f64> {
        let left_viscosity = self.get_laminar_viscosity(left_element);
        let right_viscosity = self.get_laminar_viscosity(right_element);
        let avg_viscosity = (left_viscosity + right_viscosity) / 2.0;  // Average the two viscosities

        let velocity_diff = right_element.velocity - left_element.velocity;
        let flux = (eddy_viscosity + avg_viscosity) * velocity_diff / face.area;

        flux
    }
    
}
