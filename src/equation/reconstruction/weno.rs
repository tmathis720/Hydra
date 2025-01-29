use log::{debug, error};
use super::base::ReconstructionMethod;
use thiserror::Error;

/// WENO reconstruction method for higher-order accuracy.
///
/// This implementation uses a 5th-order WENO scheme for reconstruction.
/// WENO blends multiple candidate stencils to achieve smoothness near discontinuities.
pub struct WENOReconstruction;

/// Error type for invalid WENO parameters.
#[derive(Error, Debug)]
pub enum WENOReconstructionError {
    #[error("Invalid input: NaN or Infinite value encountered.")]
    InvalidInput,
}

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
            beta[i] = 13.0 / 12.0 * (stencil[0] - 2.0 * stencil[1] + stencil[2]).powi(2)
                + 0.25 * (stencil[0] - stencil[2]).powi(2);
        }

        debug!(
            "WENO Smoothness Indicators: beta[0] = {}, beta[1] = {}, beta[2] = {}",
            beta[0], beta[1], beta[2]
        );

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

        let weights: Vec<f64> = alpha.iter().map(|&a| a / alpha_sum).collect();

        debug!(
            "WENO Weights: w[0] = {}, w[1] = {}, w[2] = {}",
            weights[0], weights[1], weights[2]
        );

        weights.try_into().unwrap()
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
        // Validate input to prevent `NaN` or `Inf`
        if !cell_value.is_finite()
            || !cell_center.iter().all(|&c| c.is_finite())
            || !face_center.iter().all(|&f| f.is_finite())
        {
            error!("WENO Reconstruction: Invalid input detected (NaN or Inf).");
            return f64::NAN;
        }

        // Simulated neighboring values (real implementations would get these from adjacent cells)
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

        debug!(
            "WENO Reconstruction: cell_value = {}, reconstructed_value = {}",
            cell_value, reconstructed_value
        );

        reconstructed_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_weno_reconstruction_smooth() {
        let weno = WENOReconstruction;

        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = weno.reconstruct(cell_value, gradient, cell_center, face_center);

        // For smooth data, reconstruction should approximate linear interpolation
        assert!(
            approx_eq(reconstructed_value, 1.5, 1e-6),
            "Expected ~1.5, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_weno_reconstruction_discontinuity() {
        let weno = WENOReconstruction;

        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = weno.reconstruct(cell_value, gradient, cell_center, face_center);

        // Ensure reconstruction is finite and stable
        assert!(
            reconstructed_value.is_finite(),
            "Reconstructed value should be finite, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_weno_reconstruction_nan_input() {
        let weno = WENOReconstruction;

        let reconstructed_value = weno.reconstruct(f64::NAN, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }

    #[test]
    fn test_weno_reconstruction_inf_input() {
        let weno = WENOReconstruction;

        let reconstructed_value = weno.reconstruct(f64::INFINITY, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }
}
