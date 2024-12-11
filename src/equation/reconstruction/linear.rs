use super::base::ReconstructionMethod;

/// Linear reconstruction method using gradients.
pub struct LinearReconstruction;

impl ReconstructionMethod for LinearReconstruction {
    fn reconstruct(
        &self,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_reconstruction() {
        let linear = LinearReconstruction;

        let cell_value = 1.0;
        let gradient = [2.0, 0.0, 0.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = linear.reconstruct(cell_value, gradient, cell_center, face_center);

        assert!((reconstructed_value - 2.0).abs() < 1e-6);
    }
}
