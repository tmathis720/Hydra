use log::{debug, error};
use super::base::ReconstructionMethod;
use thiserror::Error;

/// Linear reconstruction method using gradients.
pub struct LinearReconstruction;

/// Error type for invalid reconstruction parameters.
#[derive(Error, Debug)]
pub enum LinearReconstructionError {
    #[error("Invalid input: NaN or Infinite value encountered.")]
    InvalidInput,
}

impl ReconstructionMethod for LinearReconstruction {
    fn reconstruct(
        &self,
        cell_value: f64,
        gradient: [f64; 3],
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> f64 {
        // Validate input to ensure it doesn't contain NaN or Inf
        if !cell_value.is_finite()
            || !gradient.iter().all(|&g| g.is_finite())
            || !cell_center.iter().all(|&c| c.is_finite())
            || !face_center.iter().all(|&f| f.is_finite())
        {
            error!("Linear Reconstruction: Invalid input detected (NaN or Inf).");
            return f64::NAN;
        }

        let delta = [
            face_center[0] - cell_center[0],
            face_center[1] - cell_center[1],
            face_center[2] - cell_center[2],
        ];

        let reconstructed_value =
            cell_value + gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2];

        debug!(
            "Linear Reconstruction: cell_value = {}, gradient = {:?}, delta = {:?}, reconstructed_value = {}",
            cell_value, gradient, delta, reconstructed_value
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
    fn test_linear_reconstruction() {
        let linear = LinearReconstruction;

        let cell_value = 1.0;
        let gradient = [2.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = linear.reconstruct(cell_value, gradient, cell_center, face_center);

        assert!(
            approx_eq(reconstructed_value, 2.0, 1e-6),
            "Expected 2.0, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_linear_reconstruction_zero_gradient() {
        let linear = LinearReconstruction;

        let cell_value = 3.0;
        let gradient = [0.0, 0.0, 0.0]; // No gradient
        let cell_center = [1.0, 1.0, 1.0];
        let face_center = [2.0, 2.0, 2.0];

        let reconstructed_value = linear.reconstruct(cell_value, gradient, cell_center, face_center);

        assert!(
            approx_eq(reconstructed_value, 3.0, 1e-6),
            "Expected 3.0, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_linear_reconstruction_nan_input() {
        let linear = LinearReconstruction;

        let cell_value = f64::NAN;
        let gradient = [1.0, 1.0, 1.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [1.0, 1.0, 1.0];

        let reconstructed_value = linear.reconstruct(cell_value, gradient, cell_center, face_center);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }

    #[test]
    fn test_linear_reconstruction_inf_input() {
        let linear = LinearReconstruction;

        let cell_value = f64::INFINITY;
        let gradient = [1.0, 1.0, 1.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [1.0, 1.0, 1.0];

        let reconstructed_value = linear.reconstruct(cell_value, gradient, cell_center, face_center);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }
}
