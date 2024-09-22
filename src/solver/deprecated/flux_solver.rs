use crate::domain::{Face, Element};
use nalgebra::Vector3;

/// Struct representing a solver for flux computations.
pub struct FluxSolver;

impl FluxSolver {
    /// Compute the 3D flux between two elements.
    ///
    /// This method calculates the flux vector (flux_x, flux_y, flux_z) based on the
    /// pressure difference between elements, face geometry, and velocity.
    ///
    /// # Returns
    /// A `Vector3<f64>` representing the 3D flux in the (x, y, z) directions.
    pub fn compute_flux_3d(&self, face: &Face, left_element: &Element, right_element: &Element) -> Vector3<f64> {
        let pressure_diff = left_element.pressure - right_element.pressure;

        // Compute flux in all three directions based on face velocity and area.
        let flux = face.velocity * pressure_diff * face.area;
        flux
    }

    /// Compute the magnitude of a 3D flux vector.
    ///
    /// # Arguments
    /// * `flux_3d` - A `Vector3<f64>` representing the 3D flux.
    ///
    /// # Returns
    /// The scalar magnitude of the 3D flux vector.
    pub fn compute_flux_magnitude(&self, flux_3d: Vector3<f64>) -> f64 {
        // Compute the magnitude (Euclidean norm) of the 3D flux.
        flux_3d.norm()
    }

    /// Apply the 3D flux to the face, updating its velocity.
    ///
    /// # Arguments
    /// * `face` - The face whose velocity will be updated.
    /// * `flux_3d` - A `Vector3<f64>` representing the 3D flux to be applied.
    /// * `dt` - The time step size.
    pub fn apply_flux_3d(&self, face: &mut Face, flux_3d: Vector3<f64>, dt: f64) {
        // Update the face velocity in each direction (x, y, z).
        face.velocity += flux_3d * dt / face.area;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Face;
    use nalgebra::Vector3;

    #[test]
    fn test_compute_flux_3d() {
        let face = Face {
            id: 1,
            nodes: vec![0, 1],
            velocity: Vector3::new(1.0, 0.0, 0.0),
            area: 10.0,
            ..Face::default()
        };
        let left_element = Element {
            pressure: 5.0,
            ..Default::default()
        };
        let right_element = Element {
            pressure: 3.0,
            ..Default::default()
        };
        let solver = FluxSolver;

        let flux = solver.compute_flux_3d(&face, &left_element, &right_element);
        assert_eq!(flux, Vector3::new(20.0, 0.0, 0.0)); // (2.0 * 10.0) for the x-component
    }

    #[test]
    fn test_compute_flux_magnitude() {
        let flux_3d = Vector3::new(3.0, 4.0, 0.0);
        let solver = FluxSolver;
        let magnitude = solver.compute_flux_magnitude(flux_3d);
        assert_eq!(magnitude, 5.0); // 3-4-5 triangle
    }
}
