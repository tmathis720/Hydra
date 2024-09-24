use crate::geometry::Geometry;

impl Geometry {
    /// Computes the area of a quadrilateral face.
    ///
    /// # Arguments
    /// * `quad_vertices` - A vector of 3D coordinates for the quadrilateral vertices.
    ///
    /// # Returns
    /// * `f64` - The area of the quadrilateral.
    pub fn compute_quadrilateral_area(&self, quad_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(quad_vertices.len() == 4, "Quadrilateral must have exactly 4 vertices");

        // Split the quadrilateral into two triangles
        let triangle1 = vec![quad_vertices[0], quad_vertices[1], quad_vertices[2]];
        let triangle2 = vec![quad_vertices[2], quad_vertices[3], quad_vertices[0]];

        // Compute the area of the two triangles and sum them
        let area1 = self.compute_triangle_area(&triangle1);
        let area2 = self.compute_triangle_area(&triangle2);

        area1 + area2
    }

    /// Computes the centroid of a quadrilateral face.
    pub fn compute_quadrilateral_centroid(&self, quad_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(quad_vertices.len() == 4, "Quadrilateral must have exactly 4 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in quad_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = quad_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;

        centroid
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_quadrilateral_area_square() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&quad_vertices);

        // The area of this square is 1.0 * 1.0 = 1.0
        assert_eq!(area, 1.0);
    }

    #[test]
    fn test_quadrilateral_area_non_planar() {
        let geometry = Geometry::new();

        // Define a non-planar quadrilateral in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0],  // vertex 1
            [1.0, 0.0, 0.0],  // vertex 2
            [1.0, 1.0, 1.0],  // vertex 3 (non-planar point)
            [0.0, 1.0, 0.0],  // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&quad_vertices);

        // Since this quadrilateral is non-planar, the area should still be computed correctly
        assert!(area > 1.0, "Area of non-planar quadrilateral should be greater than 1.0");
    }

    #[test]
    fn test_quadrilateral_centroid_square() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let centroid = geometry.compute_quadrilateral_centroid(&quad_vertices);

        // The centroid of a square centered at (0.5, 0.5, 0.0)
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_quadrilateral_area_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate quadrilateral (all vertices are the same point)
        let degenerate_quad_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
            [1.0, 1.0, 1.0], // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&degenerate_quad_vertices);

        // The area of a degenerate quadrilateral is zero
        assert_eq!(area, 0.0);
    }

    #[test]
    fn test_quadrilateral_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate quadrilateral (all vertices are the same point)
        let degenerate_quad_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
            [1.0, 1.0, 1.0], // vertex 4
        ];

        let centroid = geometry.compute_quadrilateral_centroid(&degenerate_quad_vertices);

        // The centroid of a degenerate quadrilateral is the same as the vertex
        assert_eq!(centroid, [1.0, 1.0, 1.0]);
    }
}