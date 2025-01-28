use crate::geometry::{Geometry, FaceShape};

use super::GeometryError;

impl Geometry {
    /// Computes the centroid of a pyramid cell (triangular or square base).
    ///
    /// The centroid is computed for a pyramid with either a triangular or square base.
    /// The pyramid is represented by 4 vertices (triangular base) or 5 vertices (square base).
    ///
    /// - For a triangular base, the pyramid is treated as a tetrahedron.
    /// - For a square base, the pyramid is split into two tetrahedrons, and the weighted centroid of both is calculated.
    ///
    /// # Arguments
    /// * `cell_vertices` - A slice of 3D coordinates representing the vertices of the pyramid.
    ///                     4 vertices for a triangular-based pyramid, or 5 vertices for a square-based pyramid.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The 3D coordinates of the pyramid centroid or an error.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is invalid or if base centroid computation fails.
    pub fn compute_pyramid_centroid(&self, cell_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if cell_vertices.len() != 4 && cell_vertices.len() != 5 {
            log::error!(
                "Invalid number of vertices for pyramid: expected 4 or 5, got {}.",
                cell_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4, // Minimum is 4 for a triangular pyramid
                found: cell_vertices.len(),
            });
        }

        // Separate apex and base vertices
        let (apex, base_vertices, base_shape) = if cell_vertices.len() == 4 {
            (
                cell_vertices[3],
                &cell_vertices[0..3],
                FaceShape::Triangle,
            )
        } else {
            (
                cell_vertices[4],
                &cell_vertices[0..4],
                FaceShape::Quadrilateral,
            )
        };

        // Compute the base centroid
        let base_centroid = self
            .compute_face_centroid(base_shape, &base_vertices.to_vec())
            .map_err(|err| {
                log::error!("Failed to compute base centroid: {:?}", err);
                err
            })?;

        // Apply pyramid centroid formula: 3/4 * base_centroid + 1/4 * apex
        Ok([
            (3.0 * base_centroid[0] + apex[0]) / 4.0,
            (3.0 * base_centroid[1] + apex[1]) / 4.0,
            (3.0 * base_centroid[2] + apex[2]) / 4.0,
        ])
    }


    /// Computes the volume of a pyramid cell (triangular or square base).
    ///
    /// The pyramid is represented by either 4 (triangular base) or 5 (square base) vertices.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 or 5 vertices representing the 3D coordinates of the pyramid's vertices.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed volume of the pyramid or an error if the computation fails.
    ///
    /// # Errors
    /// Returns `GeometryError` if the computation of tetrahedral volumes fails.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not 4 (triangular base) or 5 (square base).
    pub fn compute_pyramid_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> Result<f64, GeometryError> {
        assert!(
            cell_vertices.len() == 4 || cell_vertices.len() == 5,
            "Pyramid must have 4 or 5 vertices"
        );

        let apex = if cell_vertices.len() == 4 { cell_vertices[3] } else { cell_vertices[4] };

        if cell_vertices.len() == 5 {
            // Decompose square-based pyramid into two tetrahedra
            let tetra1 = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let tetra2 = vec![cell_vertices[0], cell_vertices[2], cell_vertices[3], apex];

            // Compute the volumes of the two tetrahedra
            let volume1 = self.compute_tetrahedron_volume(&tetra1)?;
            let volume2 = self.compute_tetrahedron_volume(&tetra2)?;

            Ok(volume1 + volume2)
        } else {
            // Compute the volume of a triangular-based pyramid (single tetrahedron)
            let tetra = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            self.compute_tetrahedron_volume(&tetra)
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_pyramid_volume_square() {
        let geometry = Geometry::new();

        // Define a square pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [1.0, 1.0, 0.0], // base vertex 3
            [0.0, 1.0, 0.0], // base vertex 4
            [0.5, 0.5, 1.0], // apex
        ];

        let volume = geometry.compute_pyramid_volume(&pyramid_vertices);

        // Expected volume: (1/3) * base area * height, where base area = 1.0, height = 1.0
        assert!((volume.unwrap() - (1.0 / 3.0)).abs() < 1e-10, "Volume of the square pyramid is incorrect");
    }

    #[test]
    fn test_pyramid_volume_triangular() {
        let geometry = Geometry::new();

        // Define a triangular pyramid (tetrahedron) in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 1.0], // apex
        ];

        let volume = geometry.compute_pyramid_volume(&pyramid_vertices);

        // Expected volume: (1/3) * base area * height, where base area = 0.5, height = 1.0
        assert!((volume.unwrap() - (1.0 / 6.0)).abs() < 1e-10, "Volume of the triangular pyramid is incorrect");
    }

    #[test]
    fn test_pyramid_centroid_square() {
        let geometry = Geometry::new();

        // Define a square pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [1.0, 1.0, 0.0], // base vertex 3
            [0.0, 1.0, 0.0], // base vertex 4
            [0.5, 0.5, 1.0], // apex
        ];

        let centroid = geometry.compute_pyramid_centroid(&pyramid_vertices);

        // Expected centroid: (0.5, 0.5, 0.25)
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.25]);
    }

    #[test]
    fn test_pyramid_centroid_triangular() {
        let geometry = Geometry::new();

        // Define a triangular pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 1.0], // apex
        ];

        let centroid = geometry.compute_pyramid_centroid(&pyramid_vertices);

        // Expected centroid: (0.375, 0.375, 0.25)
        assert_eq!(centroid.unwrap(), [0.375, 0.375, 0.25]);
    }

    #[test]
    fn test_pyramid_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate pyramid where all points are coplanar
        let degenerate_pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 0.0], // apex (degenerate, on the same plane as the base)
        ];

        let volume = geometry.compute_pyramid_volume(&degenerate_pyramid_vertices);

        // Volume of a degenerate pyramid should be zero
        assert_eq!(volume.unwrap(), 0.0);
    }

    #[test]
    fn test_pyramid_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate pyramid where all points are coplanar
        let degenerate_pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 0.0], // apex (degenerate, on the same plane as the base)
        ];

        let centroid = geometry.compute_pyramid_centroid(&degenerate_pyramid_vertices);

        // Expected centroid in the plane: (0.375, 0.375, 0.0)
        let expected_centroid = [0.375, 0.375, 0.0];
        assert_eq!(centroid.unwrap(), expected_centroid);
    }
}
