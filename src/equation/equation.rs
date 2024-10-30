pub struct Equation {
    momentum: MomentumEquation,
    continuity: ContinuityEquation,
}

impl Equation {
    /// Calculates fluxes at cell faces using TVD upwinding
    pub fn calculate_fluxes(&self, domain: &Domain) {
        for cell in domain.cells() {
            let grad_phi = self.calculate_gradient(cell);
            let reconstructed_face_values = self.reconstruct_face_values(cell, grad_phi);
            let fluxes = self.apply_flux_limiter(cell, reconstructed_face_values);
            // Compute residuals or apply these fluxes as needed for the update
        }
    }

    pub fn compute_upwind_flux(left_value: f64, right_value: f64, velocity: f64) -> f64 {
        if velocity > 0.0 {
            left_value
        } else {
            right_value
        }
    }
    
}
