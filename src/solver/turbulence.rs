use crate::domain::Element;

pub struct EddyViscositySolver {
    pub nu_t: f64, // Eddy viscosity coefficient
}

impl EddyViscositySolver {
    pub fn apply_diffusion(&self, element_left: &mut Element, element_right: &mut Element, dt: f64) {
        let velocity_diff = element_right.momentum - element_left.momentum;
        let flux = self.nu_t * velocity_diff;

        // Apply the flux to both elements
        element_left.momentum += flux * dt;
        element_right.momentum -= flux * dt;
    }
}
