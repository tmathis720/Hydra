use crate::geometry::{Geometry, GeometryError};

impl Geometry {
    /// Computes the centroid of a triangular face.
    ///
    /// This function calculates the centroid of a triangle using its three vertices. The
    /// centroid is the point that is the average of the positions of the vertices.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A slice of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The computed 3D coordinates of the triangle's centroid,
    ///   or an error if the input is invalid.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices provided is not exactly three.
    pub fn compute_triangle_centroid(&self, triangle_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if triangle_vertices.len() != 3 {
            log::error!(
                "Invalid number of vertices for triangle: expected 3, got {}.",
                triangle_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 3,
                found: triangle_vertices.len(),
            });
        }

        // Compute the centroid as the average of the vertex coordinates
        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in triangle_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = triangle_vertices.len() as f64;
        Ok([
            centroid[0] / num_vertices,
            centroid[1] / num_vertices,
            centroid[2] / num_vertices,
        ])
    }


    /// Computes the area of a triangular face.
    ///
    /// This function calculates the area of a triangle using the cross product of two edge vectors.
    /// The magnitude of the cross product gives twice the area of the triangle, and thus the final
    /// area is half of that magnitude.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A slice of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed area of the triangle, or an error if the input is invalid.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices provided is not exactly three.
    pub fn compute_triangle_area(&self, triangle_vertices: &[[f64; 3]]) -> Result<f64, GeometryError> {
        // Validate the number of vertices
        if triangle_vertices.len() != 3 {
            log::error!(
                "Invalid number of vertices for triangle: expected 3, got {}.",
                triangle_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 3,
                found: triangle_vertices.len(),
            });
        }

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute vectors v0->v1 and v0->v2
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product of e1 and e2
        let cross_product = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Compute the magnitude of the cross product
        let cross_product_magnitude = (cross_product[0].powi(2)
            + cross_product[1].powi(2)
            + cross_product[2].powi(2))
        .sqrt();

        // Area is half the magnitude of the cross product
        let area = 0.5 * cross_product_magnitude;
        Ok(area)
    }


    /// Computes the normal vector for a triangular face in 3D space.
    ///
    /// The normal vector is computed using the cross product of two edges from the triangle.
    /// The length of the normal vector will be proportional to the area of the triangle.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A slice of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The normal vector for the triangular face or an error.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly three or if the vertices
    /// are collinear (resulting in a zero-length normal vector).
    pub fn compute_triangle_normal(
        &self,
        triangle_vertices: &[[f64; 3]],
    ) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if triangle_vertices.len() != 3 {
            log::error!(
                "Invalid number of vertices for triangle: expected 3, got {}.",
                triangle_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 3,
                found: triangle_vertices.len(),
            });
        }

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Compute the cross product to get the normal vector
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Compute the magnitude of the normal vector
        let magnitude = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();

        // Check if the normal vector is valid (non-zero length)
        if magnitude == 0.0 {
            log::error!(
                "Collinear vertices provided for triangle: {:?}. Normal vector cannot be computed.",
                triangle_vertices
            );
            return Err(GeometryError::DegenerateTriangle);
        }

        // Normalize the normal vector
        Ok([
            normal[0] / magnitude,
            normal[1] / magnitude,
            normal[2] / magnitude,
        ])
    }

}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_triangle_normal() {
        let geometry = Geometry::new();

        let triangle_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let normal = geometry.compute_triangle_normal(&triangle_vertices);

        // Expected normal should be [0.0, 0.0, 1.0] for this triangle in the XY plane
        assert!((normal.unwrap()[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_area() {
        let geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&triangle_vertices);

        // The area of this triangle is 0.5 * base * height = 0.5 * 3.0 * 4.0 = 6.0
        assert!((area.unwrap() - 6.0).abs() < 1e-10, "Area should be approximately 6.0");
    }

    #[test]
    fn test_triangle_centroid() {
        let geometry = Geometry::new();
        
        // Define a triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];
        
        let centroid = geometry.compute_triangle_centroid(&triangle_vertices);
        
        // The centroid of this triangle is the average of the vertices:
        // ([0.0, 0.0, 0.0] + [3.0, 0.0, 0.0] + [0.0, 4.0, 0.0]) / 3 = [1.0, 4.0 / 3.0, 0.0]
        assert_eq!(centroid.unwrap(), [1.0, 4.0 / 3.0, 0.0]);
    }

    #[test]
    fn test_triangle_area_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangle (all vertices are the same point)
        let degenerate_triangle_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&degenerate_triangle_vertices);

        // The area of a degenerate triangle is zero
        assert!(
            area.unwrap().abs() < 1e-10,
            "Area should be approximately zero for a degenerate triangle"
        );
    }

    #[test]
    fn test_triangle_centroid_degenerate_case() {
        let geometry = Geometry::new();
        
        // Define a degenerate triangle (all vertices are the same point)
        let degenerate_triangle_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
        ];
        
        let centroid = geometry.compute_triangle_centroid(&degenerate_triangle_vertices);
        
        // The centroid of a degenerate triangle is the same as the vertex
        assert_eq!(centroid.unwrap(), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_triangle_area_colinear_points() {
        let geometry = Geometry::new();

        // Define a triangle with colinear points
        let colinear_triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [2.0, 2.0, 2.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&colinear_triangle_vertices);

        // The area should be zero for colinear points
        assert!(
            area.unwrap().abs() < 1e-10,
            "Area should be approximately zero for colinear points"
        );
    }
}
