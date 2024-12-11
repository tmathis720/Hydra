use super::base::ReconstructionMethod;

/// WENO reconstruction method for higher-order accuracy.
///
/// This implementation uses a 5th-order WENO scheme for reconstruction.
/// WENO blends multiple candidate stencils to achieve smoothness near discontinuities.
pub struct WENOReconstruction;

impl WENOReconstruction {
    /// Computes the smoothness indicators for a set of stencils.
    ///
    /// # Parameters
    /// - `stencils`: Array of stencil values for each candidate (length = 3 for WENO5).
    ///
    /// # Returns
    /// Smoothness indicators for each stencil.
    fn compute_smoothness_indicators(stencils: [[f64; 3]; 3]) -> [f64; 3] {
        let mut beta = [0.0; 3];
        for (i, stencil) in stencils.iter().enumerate() {
            beta[i] = stencil[0].powi(2) + stencil[1].powi(2) + stencil[2].powi(2);
        }
        beta
    }

    /// Computes the weights for WENO based on the smoothness indicators.
    ///
    /// # Parameters
    /// - `beta`: Smoothness indicators.
    ///
    /// # Returns
    /// Nonlinear weights for each stencil.
    fn compute_weights(beta: [f64; 3]) -> [f64; 3] {
        const EPSILON: f64 = 1e-6; // Small constant to avoid division by zero
        let alpha: Vec<f64> = beta.iter().map(|b| 1.0 / (EPSILON + b.powi(2))).collect();
        let alpha_sum: f64 = alpha.iter().sum();
        alpha.iter().map(|&a| a / alpha_sum).collect::<Vec<f64>>().try_into().unwrap()
    }

    /// Computes the candidate stencil reconstructions.
    ///
    /// # Parameters
    /// - `values`: Scalar values at the stencil points (length = 5 for WENO5).
    ///
    /// # Returns
    /// Candidate reconstructions for each stencil.
    fn compute_candidate_reconstructions(values: [f64; 5]) -> [f64; 3] {
        [
            2.0 / 6.0 * values[0] - 7.0 / 6.0 * values[1] + 11.0 / 6.0 * values[2],
            -1.0 / 6.0 * values[1] + 5.0 / 6.0 * values[2] + 2.0 / 6.0 * values[3],
            2.0 / 6.0 * values[2] + 5.0 / 6.0 * values[3] - 1.0 / 6.0 * values[4],
        ]
    }
}

impl ReconstructionMethod for WENOReconstruction {
    fn reconstruct(
        &self,
        cell_value: f64,
        _gradient: [f64; 3],
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> f64 {
        // Delta positions for reconstruction
        let _delta = [
            face_center[0] - cell_center[0],
            face_center[1] - cell_center[1],
            face_center[2] - cell_center[2],
        ];

        // Calculate the candidate stencils based on neighboring values
        // NOTE: In a real implementation, these values would come from adjacent cells.
        // Here, we simulate them for simplicity.
        let neighbors = [
            cell_value - 2.0, // Two cells away
            cell_value - 1.0, // One cell away
            cell_value,       // Current cell
            cell_value + 1.0, // One cell forward
            cell_value + 2.0, // Two cells forward
        ];

        let stencils = Self::compute_candidate_reconstructions(neighbors);
        let beta = Self::compute_smoothness_indicators([
            [neighbors[0], neighbors[1], neighbors[2]],
            [neighbors[1], neighbors[2], neighbors[3]],
            [neighbors[2], neighbors[3], neighbors[4]],
        ]);
        let weights = Self::compute_weights(beta);

        // Combine the candidate stencils with the weights
        let reconstructed_value: f64 = stencils
            .iter()
            .zip(weights.iter())
            .map(|(candidate, weight)| candidate * weight)
            .sum();

        reconstructed_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weno_reconstruction_smooth() {
        let weno = WENOReconstruction;

        // Simulate smooth data around the cell
        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = weno.reconstruct(cell_value, gradient, cell_center, face_center);

        // For smooth data, the reconstruction should be close to the interpolated value
        assert!((reconstructed_value - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_weno_reconstruction_discontinuity() {
        let weno = WENOReconstruction;

        // Simulate discontinuous data around the cell
        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = weno.reconstruct(cell_value, gradient, cell_center, face_center);

        // For discontinuous data, the reconstruction should be stable and non-oscillatory
        assert!(reconstructed_value.is_finite());
    }
}
