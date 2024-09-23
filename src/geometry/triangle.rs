use crate::geometry::Geometry;

impl Geometry {
    /// Computes the area of a triangular face.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A vector of 3D coordinates for the triangle vertices.
    ///
    /// # Returns
    /// * `f64` - The area of the triangle.
    pub fn compute_triangle_area(&self, triangle_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(triangle_vertices.len() == 3, "Triangle must have exactly 3 vertices");

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute vectors v0->v1 and v0->v2
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Compute cross product of the two vectors
        let cross_product = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // The area is half the magnitude of the cross product
        let area = 0.5 * (cross_product[0].powi(2) + cross_product[1].powi(2) + cross_product[2].powi(2)).sqrt();
        area
    }

    /// Computes the centroid of a triangular face.
    pub fn compute_triangle_centroid(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(triangle_vertices.len() == 3, "Triangle must have exactly 3 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in triangle_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = triangle_vertices.len() as f64;
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
        assert_eq!(area, 6.0);
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
        // ([0.0, 0.0, 0.0] + [3.0, 0.0, 0.0] + [0.0, 4.0, 0.0]) / 3 = [1.0, 1.3333, 0.0]
        assert_eq!(centroid, [1.0, 4.0 / 3.0, 0.0]);
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
        assert_eq!(area, 0.0);
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
        assert_eq!(centroid, [1.0, 1.0, 1.0]);
    }
}
