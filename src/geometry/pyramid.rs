use crate::geometry::{Geometry, FaceShape};

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
    /// * `cell_vertices` - A vector of 4 vertices for a triangular-based pyramid, or 5 vertices for a square-based pyramid.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the pyramid centroid.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not 4 (for a triangular base) or 5 (for a square base).
    pub fn compute_pyramid_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 4 || cell_vertices.len() == 5, "Pyramid must have 4 or 5 vertices");

        let apex = if cell_vertices.len() == 4 { cell_vertices[3] } else { cell_vertices[4] };
        let base_vertices = if cell_vertices.len() == 4 {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]]
        } else {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], cell_vertices[3]]
        };

        let base_centroid = self.compute_face_centroid(
            if cell_vertices.len() == 4 { FaceShape::Triangle } else { FaceShape::Quadrilateral },
            &base_vertices,
        );

        // Apply pyramid centroid formula: 3/4 * base_centroid + 1/4 * apex
        [
            (3.0 * base_centroid[0] + apex[0]) / 4.0,
            (3.0 * base_centroid[1] + apex[1]) / 4.0,
            (3.0 * base_centroid[2] + apex[2]) / 4.0,
        ]
    }

    /// Computes the volume of a pyramid cell (triangular or square base).
    ///
    /// The pyramid is represented by either 4 (triangular base) or 5 (square base) vertices.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 or 5 vertices representing the 3D coordinates of the pyramid's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the pyramid.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not 4 (triangular) or 5 (square).
    pub fn compute_pyramid_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 4 || cell_vertices.len() == 5, "Pyramid must have 4 or 5 vertices");

        let apex = if cell_vertices.len() == 4 { cell_vertices[3] } else { cell_vertices[4] };
        
        if cell_vertices.len() == 5 {
            let tetra1 = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let tetra2 = vec![cell_vertices[0], cell_vertices[2], cell_vertices[3], apex];

            let volume1 = self.compute_tetrahedron_volume(&tetra1);
            let volume2 = self.compute_tetrahedron_volume(&tetra2);

            volume1 + volume2
        } else {
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
        assert!((volume - (1.0 / 3.0)).abs() < 1e-10, "Volume of the square pyramid is incorrect");
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
        assert!((volume - (1.0 / 6.0)).abs() < 1e-10, "Volume of the triangular pyramid is incorrect");
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
        assert_eq!(centroid, [0.5, 0.5, 0.25]);
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
        assert_eq!(centroid, [0.375, 0.375, 0.25]);
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
        assert_eq!(volume, 0.0);
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
        assert_eq!(centroid, expected_centroid);
    }
}
