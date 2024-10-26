use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a triangular prism.
    ///
    /// The prism is assumed to have two parallel triangular faces (top and bottom),
    /// each defined by 3 vertices. The centroid is calculated by averaging the centroids
    /// of the top and bottom triangles.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices. 
    /// The first 3 vertices define the top triangle, and the last 3 vertices define the bottom triangle.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the centroid of the triangular prism.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 6.
    pub fn compute_prism_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let top_triangle = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]];
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the centroids of both triangles
        let top_centroid = self.compute_triangle_centroid(&top_triangle);
        let bottom_centroid = self.compute_triangle_centroid(&bottom_triangle);

        // Compute the centroid of the prism by averaging the top and bottom centroids
        [
            (top_centroid[0] + bottom_centroid[0]) / 2.0,
            (top_centroid[1] + bottom_centroid[1]) / 2.0,
            (top_centroid[2] + bottom_centroid[2]) / 2.0,
        ]
    }

    /// Computes the volume of a triangular prism.
    ///
    /// The volume is calculated by multiplying the area of the triangular base (bottom face)
    /// with the height, which is the distance between the centroids of the top and bottom triangles.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices. 
    /// The first 3 vertices define the top triangle, and the last 3 vertices define the bottom triangle.
    ///
    /// # Returns
    /// * `f64` - The volume of the triangular prism.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 6.
    pub fn compute_prism_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the area of the base triangle (bottom triangle)
        let base_area = self.compute_triangle_area(&bottom_triangle);

        // Compute the height of the prism as the distance between the top and bottom triangle centroids
        let top_centroid = self.compute_triangle_centroid(&cell_vertices[0..3].to_vec());
        let bottom_centroid = self.compute_triangle_centroid(&bottom_triangle);
        let height = Geometry::compute_distance(&top_centroid, &bottom_centroid);

        // Volume of the prism = base area * height
        base_area * height
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_prism_volume_regular() {
        let geometry = Geometry::new();

        // Define a regular triangular prism in 3D space
        let prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 1.0], // bottom triangle vertex 1
            [1.0, 0.0, 1.0], // bottom triangle vertex 2
            [0.0, 1.0, 1.0], // bottom triangle vertex 3
        ];

        let volume = geometry.compute_prism_volume(&prism_vertices);

        // Base area = 0.5 (triangular area), height = 1.0 -> Volume = 0.5 * 1.0 = 0.5
        assert!((volume - 0.5).abs() < 1e-10, "Volume of the triangular prism is incorrect");
    }

    #[test]
    fn test_prism_centroid_regular() {
        let geometry = Geometry::new();

        // Define a regular triangular prism in 3D space
        let prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 1.0], // bottom triangle vertex 1
            [1.0, 0.0, 1.0], // bottom triangle vertex 2
            [0.0, 1.0, 1.0], // bottom triangle vertex 3
        ];

        let centroid = geometry.compute_prism_centroid(&prism_vertices);

        // The centroid of the top triangle = (1/3, 1/3, 0), bottom = (1/3, 1/3, 1) -> overall = (1/3, 1/3, 0.5)
        assert_eq!(centroid, [1.0 / 3.0, 1.0 / 3.0, 0.5]);
    }

    #[test]
    fn test_prism_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangular prism where all points lie on the same plane
        let degenerate_prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 0.0], // bottom triangle vertex 1 (collapsing onto the top triangle)
            [1.0, 0.0, 0.0], // bottom triangle vertex 2
            [0.0, 1.0, 0.0], // bottom triangle vertex 3
        ];

        let volume = geometry.compute_prism_volume(&degenerate_prism_vertices);

        // Since the prism collapses into a plane, the volume should be zero
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_prism_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangular prism where all points lie on the same plane
        let degenerate_prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 0.0], // bottom triangle vertex 1 (collapsing onto the top triangle)
            [1.0, 0.0, 0.0], // bottom triangle vertex 2
            [0.0, 1.0, 0.0], // bottom triangle vertex 3
        ];

        let centroid = geometry.compute_prism_centroid(&degenerate_prism_vertices);

        // The centroid is still computed as the average of all vertices
        assert_eq!(centroid, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    }
}
