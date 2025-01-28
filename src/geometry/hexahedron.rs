use crate::geometry::Geometry;

use super::GeometryError;

impl Geometry {
    /// Computes the centroid of a hexahedral cell (e.g., a cube or cuboid).
    ///
    /// The centroid is calculated by averaging the positions of all 8 vertices of the hexahedron.
    ///
    /// # Arguments
    /// * `cell_vertices` - A slice of 8 vertices, each represented by a 3D coordinate `[f64; 3]`,
    ///   representing the vertices of the hexahedron.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The 3D coordinates of the centroid of the hexahedron,
    ///   or an error if the input is invalid.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly 8.
    ///
    /// # Example
    /// ```rust
    /// use hydra::geometry::{Geometry, GeometryError};
    /// let geometry = Geometry::new();
    /// let vertices = [
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [1.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    /// ];
    /// let centroid = geometry.compute_hexahedron_centroid(&vertices).unwrap();
    /// assert_eq!(centroid, [0.5, 0.5, 0.5]);
    /// ```
    pub fn compute_hexahedron_centroid(
        &self,
        cell_vertices: &[[f64; 3]],
    ) -> Result<[f64; 3], GeometryError> {
        // Validate the input
        if cell_vertices.len() != 8 {
            log::error!(
                "Invalid number of vertices for hexahedron: expected 8, got {}.",
                cell_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 8,
                found: cell_vertices.len(),
            });
        }

        // Initialize centroid coordinates
        let mut centroid = [0.0, 0.0, 0.0];

        // Accumulate vertex coordinates
        for vertex in cell_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        // Compute the average for each coordinate
        for i in 0..3 {
            centroid[i] /= 8.0;
        }

        log::info!(
            "Computed hexahedron centroid: ({:.6}, {:.6}, {:.6})",
            centroid[0],
            centroid[1],
            centroid[2]
        );

        Ok(centroid)
    }


    /// Computes the volume of a hexahedral cell using tetrahedral decomposition.
    ///
    /// A hexahedron (e.g., a cube or cuboid) is decomposed into 5 tetrahedrons, and the volume
    /// of each tetrahedron is calculated. The sum of the volumes of all tetrahedrons gives the total
    /// volume of the hexahedron.
    ///
    /// # Arguments
    /// * `cell_vertices` - A slice of 8 vertices, each represented by a 3D coordinate `[f64; 3]`,
    ///   representing the vertices of the hexahedron.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The volume of the hexahedron or an error if the input is invalid.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly 8.
    ///
    /// # Example
    /// ```rust
    /// use hydra::geometry::{Geometry, GeometryError};
    /// let geometry = Geometry::new();
    /// let vertices = [
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [1.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    /// ];
    /// let volume = geometry.compute_hexahedron_volume(&vertices).unwrap();
    /// assert!((volume - 1.0).abs() < 1e-6);
    /// ```
    pub fn compute_hexahedron_volume(
        &self,
        cell_vertices: &[[f64; 3]],
    ) -> Result<f64, GeometryError> {
        // Validate the input
        if cell_vertices.len() != 8 {
            log::error!(
                "Invalid number of vertices for hexahedron: expected 8, got {}.",
                cell_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 8,
                found: cell_vertices.len(),
            });
        }

        // Define the tetrahedrons for decomposition
        let tets = [
            [cell_vertices[0], cell_vertices[1], cell_vertices[3], cell_vertices[4]], // Tet 1
            [cell_vertices[1], cell_vertices[2], cell_vertices[3], cell_vertices[6]], // Tet 2
            [cell_vertices[1], cell_vertices[3], cell_vertices[4], cell_vertices[6]], // Tet 3
            [cell_vertices[3], cell_vertices[4], cell_vertices[6], cell_vertices[7]], // Tet 4
            [cell_vertices[1], cell_vertices[4], cell_vertices[5], cell_vertices[6]], // Tet 5
        ];

        // Sum the volumes of the tetrahedrons
        let total_volume = tets.iter().try_fold(0.0, |acc, tet| {
            let tet_volume = self.compute_tetrahedron_volume(tet)?;
            Ok(acc + tet_volume)
        })?;

        log::info!("Computed hexahedron volume: {:.6}", total_volume);

        Ok(total_volume)
    }

}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_hexahedron_volume_regular() {
        let geometry = Geometry::new();

        // Define a regular hexahedron (cube) in 3D space
        let hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 1.0], // vertex 5
            [1.0, 0.0, 1.0], // vertex 6
            [1.0, 1.0, 1.0], // vertex 7
            [0.0, 1.0, 1.0], // vertex 8
        ];

        let volume = geometry.compute_hexahedron_volume(&hexahedron_vertices);

        // The volume of a cube is side^3, here side = 1.0, so volume = 1.0^3 = 1.0
        assert!((volume.unwrap() - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
    }

    #[test]
    fn test_hexahedron_centroid_regular() {
        let geometry = Geometry::new();

        // Define a regular hexahedron (cube) in 3D space
        let hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 1.0], // vertex 5
            [1.0, 0.0, 1.0], // vertex 6
            [1.0, 1.0, 1.0], // vertex 7
            [0.0, 1.0, 1.0], // vertex 8
        ];

        let centroid = geometry.compute_hexahedron_centroid(&hexahedron_vertices);

        // The centroid of a unit cube is at the center: (0.5, 0.5, 0.5)
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_hexahedron_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate hexahedron where all vertices collapse into a plane
        let degenerate_hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 0.0], // vertex 5 (collapsing to the same plane)
            [1.0, 0.0, 0.0], // vertex 6 (collapsing to the same plane)
            [1.0, 1.0, 0.0], // vertex 7 (collapsing to the same plane)
            [0.0, 1.0, 0.0], // vertex 8 (collapsing to the same plane)
        ];

        let volume = geometry.compute_hexahedron_volume(&degenerate_hexahedron_vertices);

        // Since the hexahedron collapses into a plane, the volume should be zero
        assert_eq!(volume.unwrap(), 0.0);
    }

    #[test]
    fn test_hexahedron_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate hexahedron where all vertices collapse into a plane
        let degenerate_hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 0.0], // vertex 5 (collapsing to the same plane)
            [1.0, 0.0, 0.0], // vertex 6 (collapsing to the same plane)
            [1.0, 1.0, 0.0], // vertex 7 (collapsing to the same plane)
            [0.0, 1.0, 0.0], // vertex 8 (collapsing to the same plane)
        ];

        let centroid = geometry.compute_hexahedron_centroid(&degenerate_hexahedron_vertices);

        // The centroid should still be the average of the vertices, which lie in the XY plane
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.0]);
    }
}
