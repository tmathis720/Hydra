// src/equation/reconstruction/reconstruct.rs

/// Reconstructs the solution at a face center by extrapolating from the cell value 
/// and its gradient. This approach is critical for finite volume methods as it
/// provides a face-centered scalar value, which is essential for flux calculations.
///
/// # Arguments
///
/// * `cell_value` - The scalar field value at the cell center, representing the primary
///                  field quantity (e.g., temperature, pressure).
/// * `gradient` - The gradient vector `[f64; 3]` representing the rate of change of the
///                scalar field within the cell in each spatial direction. This gradient
///                allows for a linear approximation of the field near the cell center.
/// * `cell_center` - Coordinates `[f64; 3]` of the cell center, where `cell_value` and
///                   `gradient` are defined.
/// * `face_center` - Coordinates `[f64; 3]` of the face center where the scalar field 
///                   value is to be reconstructed.
///
/// # Returns
///
/// The reconstructed scalar field value at the face center, determined by linearly 
/// extrapolating from the cell center using the gradient.
///
/// # Example
///
/// ```rust
/// use hydra::equation::reconstruction::reconstruct::reconstruct_face_value;
///
/// let cell_value = 1.0;
/// let gradient = [2.0, 0.0, 0.0];
/// let cell_center = [0.0, 0.0, 0.0];
/// let face_center = [0.5, 0.0, 0.0];
///
/// let reconstructed_value = reconstruct_face_value(cell_value, gradient, cell_center, face_center);
/// assert_eq!(reconstructed_value, 2.0);
/// ```
pub fn reconstruct_face_value(
    cell_value: f64,
    gradient: [f64; 3],
    cell_center: [f64; 3],
    face_center: [f64; 3],
) -> f64 {
    let delta = [
        face_center[0] - cell_center[0],
        face_center[1] - cell_center[1],
        face_center[2] - cell_center[2],
    ];
    cell_value + gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2]
}

/// Unit tests for `reconstruct_face_value`.
///
/// Tests a variety of scenarios to ensure correct reconstruction of values
/// at face centers, verifying the linear extrapolation approach based on the
/// provided gradient and cell/face positions.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct_face_value() {
        // Test case 1: Gradient in the x-direction
        let cell_value = 1.0;
        let gradient = [2.0, 0.0, 0.0]; // Gradient along x-axis
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: cell_value + gradient_x * delta_x = 1.0 + 2.0 * 0.5 = 2.0
        let expected_value = 2.0;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );

        // Test case 2: Gradient in the y-direction
        let cell_value = 3.0;
        let gradient = [0.0, -1.0, 0.0]; // Gradient along negative y-axis
        let cell_center = [1.0, 1.0, 0.0];
        let face_center = [1.0, 0.5, 0.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: 3.0 + (-1.0) * (-0.5) = 3.0 + 0.5 = 3.5
        let expected_value = 3.5;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );

        // Test case 3: Gradient in all directions
        let cell_value = 0.0;
        let gradient = [1.0, 2.0, 3.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [1.0, 1.0, 1.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: 0.0 + 1*1 + 2*1 + 3*1 = 6.0
        let expected_value = 6.0;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );
    }
}
