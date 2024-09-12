use crate::domain::Element;
use crate::boundary::BoundaryElement;
use crate::solver::FlowField;

pub struct FreeSurfaceBoundary {
    pub pressure_at_surface: f64, // Pressure at the free surface (typically atmospheric pressure)
}

impl FreeSurfaceBoundary {
    /// Apply the free surface boundary condition
    /// This method adjusts the pressure of the element toward the surface pressure over time.
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        // Calculate the pressure difference between the element and surface
        let pressure_difference = element.pressure - self.pressure_at_surface;

        // Gradually adjust the pressure toward the free surface pressure using a relaxation factor
        let relaxation_factor = 0.1;  // This can be adjusted to control the speed of transition
        element.pressure -= relaxation_factor * pressure_difference * dt;

        // Ensure that the pressure doesn't drop below the free surface pressure (e.g., atmospheric)
        element.pressure = element.pressure.max(self.pressure_at_surface);
    }

    /// Update the dynamic fluxes at the free surface
    /// This function calculates and updates the velocity and height of the free surface element
    /// based on surface fluxes and time step.
    pub fn update_dynamic_fluxes(&self, boundary: &mut BoundaryElement, flow_field: &FlowField, time_step: f64) {
        let mut element_ref = boundary.element.borrow_mut();

        // Calculate the surface flux based on physical properties
        let surface_flux = self.compute_surface_flux(&element_ref, flow_field);

        // Update the velocity based on the surface flux (mass conservation)
        // Assume surface_flux is in units of volume or mass per unit time
        element_ref.velocity.2 += surface_flux / element_ref.area;  // Adjust only vertical velocity

        // Update the height of the water surface based on the velocity and the time step
        element_ref.height += element_ref.velocity.2 * time_step;
    }

    /// Compute the surface flux at the free surface boundary
    /// This function uses Bernoulli's equation and other principles to compute the flux across the boundary.
    fn compute_surface_flux(&self, element: &Element, flow_field: &FlowField) -> f64 {
        // Calculate pressure difference between the current element and nearby elements
        let pressure_diff = element.pressure - flow_field.get_nearby_pressure(element);
        let density = element.compute_density();
        // Compute the flux based on Bernoulli's principle
        // flux = sqrt(2 * pressure_diff / density)
        // Ensure density is available in the element, otherwise you need to provide it
        let flux = (2.0 * pressure_diff / density).sqrt();

        // Return the flux, adjusted by physical factors (e.g., geometry of the surface)
        flux
    }
}
