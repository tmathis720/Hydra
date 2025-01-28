use crate::geometry::{Geometry, GeometryError};

impl Geometry {
    /// Computes the area of a quadrilateral face by dividing it into two triangles.
    ///
    /// This function computes the area of a quadrilateral in 3D space by splitting it into
    /// two triangles and summing their respective areas. The quadrilateral is assumed to have
    /// exactly four vertices.
    ///
    /// # Arguments
    ///
    /// * `quad_vertices` - A slice of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    ///
    /// * `Result<f64, GeometryError>` - The computed area of the quadrilateral or an error.
    ///
    /// # Errors
    ///
    /// Returns a `GeometryError` if the number of vertices is not exactly four.
    pub fn compute_quadrilateral_area(&self, quad_vertices: &[[f64; 3]]) -> Result<f64, GeometryError> {
        // Validate the number of vertices
        if quad_vertices.len() != 4 {
            log::error!(
                "Invalid number of vertices for quadrilateral: expected 4, got {}.",
                quad_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4,
                found: quad_vertices.len(),
            });
        }

        // Split the quadrilateral into two triangles
        let triangle1 = [quad_vertices[0], quad_vertices[1], quad_vertices[2]];
        let triangle2 = [quad_vertices[2], quad_vertices[3], quad_vertices[0]];

        // Compute the area of the two triangles
        let area1 = self.compute_triangle_area(&triangle1)?;
        let area2 = self.compute_triangle_area(&triangle2)?;

        // Return the total area
        Ok(area1 + area2)
    }


    /// Computes the centroid of a quadrilateral face.
    ///
    /// This function calculates the centroid (geometric center) of a quadrilateral by averaging the
    /// positions of its four vertices. The centroid is the point that minimizes the sum of the
    /// squared distances to all the vertices.
    ///
    /// # Arguments
    ///
    /// * `quad_vertices` - A slice of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    ///
    /// * `Result<[f64; 3], GeometryError>` - The computed centroid of the quadrilateral or an error.
    ///
    /// # Errors
    ///
    /// Returns a `GeometryError` if the number of vertices is not exactly four.
    pub fn compute_quadrilateral_centroid(&self, quad_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if quad_vertices.len() != 4 {
            log::error!(
                "Invalid number of vertices for quadrilateral: expected 4, got {}.",
                quad_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4,
                found: quad_vertices.len(),
            });
        }

        // Initialize centroid coordinates
        let mut centroid = [0.0, 0.0, 0.0];

        // Compute the sum of vertex coordinates
        for vertex in quad_vertices.iter() {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        // Divide by the number of vertices to get the average
        let num_vertices = quad_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;

        Ok(centroid)
    }


    /// Computes the normal vector for a quadrilateral face in 3D space.
    ///
    /// The normal vector is approximated by dividing the quadrilateral into two triangles,
    /// calculating each triangle's normal, and averaging them.
    ///
    /// # Arguments
    /// * `quad_vertices` - A slice of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The normal vector for the quadrilateral face or an error.
    ///
    /// # Errors
    /// Returns a `GeometryError` if the number of vertices is not exactly four or if any other
    /// computational issue arises.
    pub fn compute_quadrilateral_normal(&self, quad_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices
        if quad_vertices.len() != 4 {
            log::error!(
                "Invalid number of vertices for quadrilateral: expected 4, got {}.",
                quad_vertices.len()
            );
            return Err(GeometryError::InvalidVertexCount {
                expected: 4,
                found: quad_vertices.len(),
            });
        }

        // Split the quadrilateral into two triangles
        let triangle1 = [quad_vertices[0], quad_vertices[1], quad_vertices[2]];
        let triangle2 = [quad_vertices[2], quad_vertices[3], quad_vertices[0]];

        // Compute normals for each triangle
        let normal1 = self.compute_triangle_normal(&triangle1)
            .map_err(|e| {
                log::error!("Error computing normal for triangle 1: {:?}", e);
                e
            })?;

        let normal2 = self.compute_triangle_normal(&triangle2)
            .map_err(|e| {
                log::error!("Error computing normal for triangle 2: {:?}", e);
                e
            })?;

        // Average the two normals
        let averaged_normal = [
            (normal1[0] + normal2[0]) / 2.0,
            (normal1[1] + normal2[1]) / 2.0,
            (normal1[2] + normal2[2]) / 2.0,
        ];

        // Normalize the resulting normal vector
        let magnitude = (averaged_normal[0].powi(2)
            + averaged_normal[1].powi(2)
            + averaged_normal[2].powi(2))
        .sqrt();

        if magnitude == 0.0 {
            log::error!("Computed normal has zero magnitude, which is invalid.");
            return Err(GeometryError::ZeroMagnitudeNormal);
        }

        Ok([
            averaged_normal[0] / magnitude,
            averaged_normal[1] / magnitude,
            averaged_normal[2] / magnitude,
        ])
    }

}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_quadrilateral_normal() {
        let geometry = Geometry::new();

        let quad_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let normal = geometry.compute_quadrilateral_normal(&quad_vertices);

        // Expected normal should be [0.0, 0.0, 1.0] for this quadrilateral in the XY plane
        assert!((normal.unwrap()[2] - 1.0).abs() < 1e-10);
    }

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
        assert_eq!(area.unwrap(), 1.0);
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
        assert!(area.unwrap() > 1.0, "Area of non-planar quadrilateral should be greater than 1.0");
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

        // The centroid of a square is at (0.5, 0.5, 0.0)
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.0]);
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
        assert_eq!(area.unwrap(), 0.0);
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
        assert_eq!(centroid.unwrap(), [1.0, 1.0, 1.0]);
    }
}
