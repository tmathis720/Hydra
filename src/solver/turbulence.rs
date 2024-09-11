use crate::domain::element::Element;
pub struct TurbulenceSolver {
    pub k: f64,  // Turbulent kinetic energy
    pub epsilon: f64,  // Turbulent dissipation rate
}

impl TurbulenceSolver {
    pub fn apply_turbulence(&self, element: &mut Element, dt: f64) {
        // Update the element's momentum and pressure based on turbulent diffusion
        let turbulent_diffusion = self.k / self.epsilon;
        element.pressure -= turbulent_diffusion * dt;
        element.momentum -= turbulent_diffusion * element.momentum * dt;

        // Prevent the pressure from going negative
        element.pressure = element.pressure.max(0.0);

        // Prevent the momentum from going negative
        element.momentum = element.momentum.max(0.0);
    }
}
