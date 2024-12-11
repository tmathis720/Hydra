/// Trait defining the interface for all reconstruction methods.
///
/// A reconstruction method estimates the value at a specific location
/// (e.g., face center) using cell-centered data and gradients.
pub trait ReconstructionMethod {
    /// Reconstructs a value at the specified location based on input data.
    ///
    /// # Parameters
    /// - `cell_value`: Scalar value at the cell center.
    /// - `gradient`: Gradient vector of the scalar field.
    /// - `cell_center`: Coordinates of the cell center.
    /// - `face_center`: Coordinates of the face center.
    ///
    /// # Returns
    /// Reconstructed value at the face center.
    fn reconstruct(
        &self,
        cell_value: f64,
        gradient: [f64; 3],
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> f64;
}
