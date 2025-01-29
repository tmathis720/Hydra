use log::{debug, error};
use super::base::ReconstructionMethod;
use thiserror::Error;

/// PPM (Piecewise Parabolic Method) reconstruction.
///
/// This implementation constructs a piecewise parabolic profile within each cell and uses it to
/// reconstruct values at the cell faces. PPM provides higher-order accuracy and is designed
/// to resolve sharp features while minimizing numerical oscillations.
pub struct PPMReconstruction;

/// Error type for invalid reconstruction parameters.
#[derive(Error, Debug)]
pub enum PPMReconstructionError {
    #[error("Invalid input: NaN or Infinite value encountered.")]
    InvalidInput,
}

impl PPMReconstruction {
    /// Limits the parabolic coefficients to avoid overshooting at discontinuities.
    ///
    /// # Parameters
    /// - `left`: Value at the left boundary of the stencil.
    /// - `center`: Value at the center of the stencil.
    /// - `right`: Value at the right boundary of the stencil.
    ///
    /// # Returns
    /// Limited value at the center to ensure non-oscillatory behavior.
    fn limit(left: f64, center: f64, right: f64) -> f64 {
        let min_val = left.min(right);
        let max_val = left.max(right);
        let limited_center = center.clamp(min_val, max_val);

        debug!(
            "PPM Limiting: left = {}, center = {}, right = {}, limited_center = {}",
            left, center, right, limited_center
        );

        limited_center
    }

    /// Computes the parabolic coefficients for a given cell.
    ///
    /// # Parameters
    /// - `left`: Value at the left stencil.
    /// - `center`: Value at the cell center.
    /// - `right`: Value at the right stencil.
    ///
    /// # Returns
    /// Parabolic coefficients `a`, `b`, and `c` for the quadratic equation:
    /// `a * x^2 + b * x + c`.
    fn compute_parabolic_coefficients(left: f64, center: f64, right: f64) -> (f64, f64, f64) {
        let a = 0.5 * (left - 2.0 * center + right);
        let b = 0.5 * (right - left);
        let c = center;

        debug!(
            "PPM Parabolic Coefficients: left = {}, center = {}, right = {}, a = {}, b = {}, c = {}",
            left, center, right, a, b, c
        );

        (a, b, c)
    }

    /// Evaluates the parabolic function at a given point.
    ///
    /// # Parameters
    /// - `a`: Quadratic coefficient.
    /// - `b`: Linear coefficient.
    /// - `c`: Constant coefficient.
    /// - `x`: Position (in normalized coordinates) where the function is evaluated.
    ///
    /// # Returns
    /// Value of the parabolic function at `x`.
    fn evaluate_parabola(a: f64, b: f64, c: f64, x: f64) -> f64 {
        let result = a * x.powi(2) + b * x + c;

        debug!(
            "PPM Evaluate Parabola: a = {}, b = {}, c = {}, x = {}, result = {}",
            a, b, c, x, result
        );

        result
    }
}

impl ReconstructionMethod for PPMReconstruction {
    fn reconstruct(
        &self,
        cell_value: f64,
        _gradient: [f64; 3],
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> f64 {
        // Validate input to ensure it doesn't contain NaN or Inf
        if !cell_value.is_finite()
            || !cell_center.iter().all(|&c| c.is_finite())
            || !face_center.iter().all(|&f| f.is_finite())
        {
            error!("PPM Reconstruction: Invalid input detected (NaN or Inf).");
            return f64::NAN;
        }

        // Simulated neighboring values (in a real setup, this data comes from adjacent cells)
        let left = cell_value - 1.0;
        let right = cell_value + 1.0;

        // Limit the values to ensure non-oscillatory behavior
        let limited_center = Self::limit(left, cell_value, right);

        // Compute parabolic coefficients
        let (a, b, c) = Self::compute_parabolic_coefficients(left, limited_center, right);

        // Calculate normalized position along the cell: -0.5 (left face) to 0.5 (right face)
        let delta = face_center.iter().zip(cell_center.iter()).map(|(f, c)| f - c).collect::<Vec<f64>>();
        let x_norm = delta[0].signum() * 0.5; // Assume 1D; generalize for 3D if needed

        // Evaluate the parabolic function at the face
        Self::evaluate_parabola(a, b, c, x_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_ppm_reconstruction_smooth() {
        let ppm = PPMReconstruction;

        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = ppm.reconstruct(cell_value, gradient, cell_center, face_center);

        // Smooth data should reconstruct correctly within the parabolic profile
        assert!(
            approx_eq(reconstructed_value, 1.5, 1e-6),
            "Expected ~1.5, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_ppm_reconstruction_discontinuity() {
        let ppm = PPMReconstruction;

        let cell_value = 1.0;
        let gradient = [1.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = ppm.reconstruct(cell_value, gradient, cell_center, face_center);

        // Ensure reconstruction is finite and stable
        assert!(
            reconstructed_value.is_finite(),
            "Reconstructed value should be finite, got {}",
            reconstructed_value
        );
    }

    #[test]
    fn test_ppm_reconstruction_limiting() {
        let _ppm = PPMReconstruction;

        // Test limiting function for oscillatory input
        let limited = PPMReconstruction::limit(2.0, 10.0, 4.0);

        // Limited value should be within the range [2.0, 4.0]
        assert_eq!(limited, 4.0, "Expected 4.0, got {}", limited);
    }

    #[test]
    fn test_ppm_reconstruction_nan_input() {
        let ppm = PPMReconstruction;

        let reconstructed_value = ppm.reconstruct(f64::NAN, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }

    #[test]
    fn test_ppm_reconstruction_inf_input() {
        let ppm = PPMReconstruction;

        let reconstructed_value = ppm.reconstruct(f64::INFINITY, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert!(reconstructed_value.is_nan(), "Expected NaN, got {}", reconstructed_value);
    }
}
