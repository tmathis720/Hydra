use crate::geometry::{Geometry, GeometryError};

impl Geometry {
    /// Computes the centroid of a tetrahedral cell.
    ///
    /// This function calculates the centroid (geometric center) of a tetrahedron using
    /// the 3D coordinates of its four vertices. The centroid is the average of the
    /// positions of all vertices.
    ///
    /// # Arguments
    /// * `cell_vertices` - A slice of 3D coordinates representing the vertices of the tetrahedron.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The 3D coordinates of the computed centroid or an error.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly 4.
    ///
    /// # Example
    /// ```rust
    /// use hydra::geometry::Geometry;
    /// let geometry = Geometry::new();
    /// let vertices = [
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ];
    /// let centroid = geometry.compute_tetrahedron_centroid(&vertices).unwrap();
    /// assert_eq!(centroid, [0.25, 0.25, 0.25]);
    /// ```
    pub fn compute_tetrahedron_centroid(&self, cell_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if cell_vertices.len() != 4 {
            log::error!(
                "Invalid number of vertices for tetrahedron: expected 4, got {}.",
                cell_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4,
                found: cell_vertices.len(),
            });
        }

        // Initialize the centroid coordinates
        let mut centroid = [0.0, 0.0, 0.0];

        // Sum the vertex coordinates
        for vertex in cell_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        // Average the vertex coordinates to compute the centroid
        let num_vertices = cell_vertices.len() as f64;
        Ok([
            centroid[0] / num_vertices,
            centroid[1] / num_vertices,
            centroid[2] / num_vertices,
        ])
    }


    /// Computes the volume of a tetrahedral cell.
    ///
    /// This function calculates the volume of a tetrahedron using the 3D coordinates of
    /// its four vertices. The volume is determined by computing the determinant of a matrix
    /// formed by three edges of the tetrahedron originating from a single vertex.
    ///
    /// # Arguments
    /// * `tet_vertices` - A slice of 4 vertices representing the 3D coordinates of the tetrahedron's vertices.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The volume of the tetrahedron or an error.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly four.
    ///
    /// # Example
    /// ```rust
    /// use hydra::geometry::{Geometry, GeometryError};
    /// let geometry = Geometry::new();
    /// let vertices = [
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0]
    /// ];
    /// let volume = geometry.compute_tetrahedron_volume(&vertices).unwrap();
    /// assert!((volume - 1.0 / 6.0).abs() < 1e-10);
    /// ```
    pub fn compute_tetrahedron_volume(&self, tet_vertices: &[[f64; 3]]) -> Result<f64, GeometryError> {
        // Validate the number of vertices
        if tet_vertices.len() != 4 {
            log::error!(
                "Invalid number of vertices for tetrahedron: expected 4, got {}.",
                tet_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4,
                found: tet_vertices.len(),
            });
        }

        // Extract vertices
        let v0 = tet_vertices[0];
        let v1 = tet_vertices[1];
        let v2 = tet_vertices[2];
        let v3 = tet_vertices[3];

        // Compute edge vectors from v0
        let matrix = [
            [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
            [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]],
            [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]],
        ];

        // Compute the determinant of the matrix
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

        // Compute the volume
        let volume = det.abs() / 6.0;

        log::info!("Computed tetrahedron volume: {:.6}", volume);
        Ok(volume)
    }

}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_tetrahedron_volume() {
        let geometry = Geometry::new();

        // Define a simple tetrahedron in 3D space
        let tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [0.0, 0.0, 1.0], // vertex 4
        ];

        let volume = geometry.compute_tetrahedron_volume(&tetrahedron_vertices);

        // The volume of this tetrahedron is 1/6, since it is a right-angled tetrahedron
        assert!((volume.unwrap() - 1.0 / 6.0).abs() < 1e-10, "Volume of the tetrahedron is incorrect");
    }

    #[test]
    fn test_tetrahedron_centroid() {
        let geometry = Geometry::new();

        // Define a simple tetrahedron in 3D space
        let tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [0.0, 0.0, 1.0], // vertex 4
        ];

        let centroid = geometry.compute_tetrahedron_centroid(&tetrahedron_vertices);

        // The centroid of a regular tetrahedron in this configuration is at [0.25, 0.25, 0.25]
        assert_eq!(centroid.unwrap(), [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_tetrahedron_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate tetrahedron where all vertices lie on the same plane
        let degenerate_tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [1.0, 1.0, 0.0], // vertex 4 (lies on the same plane as other vertices)
        ];

        let volume = geometry.compute_tetrahedron_volume(&degenerate_tetrahedron_vertices);

        // The volume of a degenerate tetrahedron is zero
        assert_eq!(volume.unwrap(), 0.0);
    }

    #[test]
    fn test_tetrahedron_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate tetrahedron where all vertices lie on the same plane
        let degenerate_tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [1.0, 1.0, 0.0], // vertex 4 (lies on the same plane)
        ];

        let centroid = geometry.compute_tetrahedron_centroid(&degenerate_tetrahedron_vertices);

        // The centroid of this degenerate tetrahedron should still be calculated
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.0]);
    }
}
