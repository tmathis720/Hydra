use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a tetrahedral cell
    pub fn compute_tetrahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        let mut centroid = [0.0, 0.0, 0.0];
        for v in cell_vertices {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }
        let num_vertices = cell_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;
        centroid
    }

    /// Computes the volume of a tetrahedral cell
    ///
    /// # Arguments
    /// * `tet_vertices` - A vector of 4 vertices representing the 3D coordinates of the tetrahedron's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the tetrahedron.
    pub fn compute_tetrahedron_volume(&self, tet_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(tet_vertices.len() == 4, "Tetrahedron must have exactly 4 vertices");

        let v0 = tet_vertices[0];
        let v1 = tet_vertices[1];
        let v2 = tet_vertices[2];
        let v3 = tet_vertices[3];

        let matrix = [
            [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
            [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]],
            [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]],
        ];

        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

        det.abs() / 6.0
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
        assert!((volume - 1.0 / 6.0).abs() < 1e-10, "Volume of the tetrahedron is incorrect");
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
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
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
        assert_eq!(volume, 0.0);
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
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }
}
