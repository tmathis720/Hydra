/// Struct representing a solver for scalar transport.
pub struct ScalarTransportSolver;

impl ScalarTransportSolver {
    /// Computes the flux of a scalar between two elements based on the flow flux and scalar concentrations.
    ///
    /// # Arguments
    /// - `flux`: The flux of fluid between two elements. Positive flux moves scalar from left to right, negative from right to left.
    /// - `left_scalar`: The scalar concentration in the left element.
    /// - `right_scalar`: The scalar concentration in the right element.
    ///
    /// # Returns
    /// The computed scalar flux, which is the amount of scalar transported across the face between the two elements.
    pub fn compute_scalar_flux(&self, flux: f64, left_scalar: f64, right_scalar: f64) -> f64 {
        match flux.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => flux * left_scalar,  // Positive flux: transport from left to right
            Some(std::cmp::Ordering::Less) => flux * right_scalar,    // Negative flux: transport from right to left
            _ => 0.0,  // No flux, no transport
        }
    }

    /// Computes the scalar flux for cases where diffusion might also play a role.
    ///
    /// # Arguments
    /// - `flux`: The advective flux of fluid between two elements.
    /// - `left_scalar`: The scalar concentration in the left element.
    /// - `right_scalar`: The scalar concentration in the right element.
    /// - `diffusion_coeff`: The diffusion coefficient for scalar diffusion.
    /// - `distance`: The distance between the two elements (can be face length/area for 2D/3D).
    ///
    /// # Returns
    /// The total scalar flux, including both advective and diffusive components.
    pub fn compute_advective_diffusive_flux(
        &self, 
        flux: f64, 
        left_scalar: f64, 
        right_scalar: f64, 
        diffusion_coeff: f64, 
        distance: f64
    ) -> f64 {
        let advective_flux = self.compute_scalar_flux(flux, left_scalar, right_scalar);
        let diffusive_flux = self.compute_diffusive_flux(left_scalar, right_scalar, diffusion_coeff, distance);

        advective_flux + diffusive_flux
    }

    /// Computes the diffusive flux of a scalar based on Fick's law of diffusion.
    ///
    /// # Arguments
    /// - `left_scalar`: The scalar concentration in the left element.
    /// - `right_scalar`: The scalar concentration in the right element.
    /// - `diffusion_coeff`: The diffusion coefficient for scalar diffusion.
    /// - `distance`: The distance between the two elements.
    ///
    /// # Returns
    /// The diffusive flux of the scalar.
    fn compute_diffusive_flux(&self, left_scalar: f64, right_scalar: f64, diffusion_coeff: f64, distance: f64) -> f64 {
        if distance > 0.0 {
            diffusion_coeff * (right_scalar - left_scalar) / distance
        } else {
            0.0  // Avoid division by zero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarTransportSolver;

    #[test]
    fn test_compute_scalar_flux_positive_flux() {
        let solver = ScalarTransportSolver;
        let flux = 1.0;
        let left_scalar = 2.0;
        let right_scalar = 1.0;
        let scalar_flux = solver.compute_scalar_flux(flux, left_scalar, right_scalar);
        assert_eq!(scalar_flux, 2.0);
    }

    #[test]
    fn test_compute_scalar_flux_negative_flux() {
        let solver = ScalarTransportSolver;
        let flux = -1.0;
        let left_scalar = 2.0;
        let right_scalar = 1.0;
        let scalar_flux = solver.compute_scalar_flux(flux, left_scalar, right_scalar);
        assert_eq!(scalar_flux, -1.0);
    }

    #[test]
    fn test_compute_scalar_flux_no_flux() {
        let solver = ScalarTransportSolver;
        let flux = 0.0;
        let left_scalar = 2.0;
        let right_scalar = 1.0;
        let scalar_flux = solver.compute_scalar_flux(flux, left_scalar, right_scalar);
        assert_eq!(scalar_flux, 0.0);
    }

    #[test]
    fn test_compute_advective_diffusive_flux() {
        let solver = ScalarTransportSolver;
        let flux = 1.0;
        let left_scalar = 2.0;
        let right_scalar = 1.0;
        let diffusion_coeff = 0.1;
        let distance = 1.0;
        let scalar_flux = solver.compute_advective_diffusive_flux(flux, left_scalar, right_scalar, diffusion_coeff, distance);
        assert_eq!(scalar_flux, 2.0 + (-0.1));  // Advective flux = 2.0, diffusive flux = -0.1
    }
}
