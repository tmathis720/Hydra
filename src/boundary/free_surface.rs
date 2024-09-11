use crate::domain::Element;


pub struct FreeSurfaceBoundary {
    pub pressure_at_surface: f64, // Pressure at the free surface (atmospheric pressure)
}

impl FreeSurfaceBoundary {
    /// Apply free surface boundary condition
    /// Gradually adjust the pressure toward the surface pressure over time
    pub fn apply_boundary(&self, element: &mut Element, dt: f64) {
        // Instead of setting the pressure to the surface pressure directly,
        // we apply a gradual adjustment toward the free surface pressure
        let pressure_difference = element.pressure - self.pressure_at_surface;

        // Adjust pressure gradually (use a simple relaxation factor)
        let relaxation_factor = 0.1; // Adjust this as needed for smoother transitions
        element.pressure -= relaxation_factor * pressure_difference * dt;

        // Ensure the element's pressure doesn't drop below the surface pressure
        element.pressure = element.pressure.max(self.pressure_at_surface);
    }
}