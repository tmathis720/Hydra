pub struct ScalarTransportSolver;

impl ScalarTransportSolver {
    /// Computes the flux of a scalar between two elements based on the flow flux and scalar concentrations
    ///
    /// `flux`: The flux of fluid between two elements
    /// `left_scalar`: The scalar concentration in the left element
    /// `right_scalar`: The scalar concentration in the right element
    pub fn compute_scalar_flux(&self, flux: f64, left_scalar: f64, right_scalar: f64) -> f64 {
        // Scalar transport depends on the direction of flux:
        // - Positive flux moves scalar from left to right
        // - Negative flux moves scalar from right to left

        // If the flux is positive, the scalar moves from left to right
        if flux > 0.0 {
            return flux * left_scalar;
        }
        // If the flux is negative, the scalar moves from right to left
        else if flux < 0.0 {
            return flux * right_scalar;
        }

        // If there's no flux, no scalar is transported
        0.0
    }
}
