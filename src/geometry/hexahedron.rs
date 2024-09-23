use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a hexahedral cell
    pub fn compute_hexahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
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

    /// Computes the volume of a hexahedral (cube-like) cell using tetrahedral decomposition.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 8 vertices representing the 3D coordinates of the hexahedron's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the hexahedron.
    pub fn compute_hexahedron_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 8, "Hexahedron must have exactly 8 vertices");

        // Tetrahedral decomposition of a hexahedron (5 tetrahedrons).
        // We create the tetrahedrons using specific vertices from the hexahedron.
        let tets = vec![
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[3], cell_vertices[4]], // Tet 1
            vec![cell_vertices[1], cell_vertices[2], cell_vertices[3], cell_vertices[6]], // Tet 2
            vec![cell_vertices[1], cell_vertices[3], cell_vertices[4], cell_vertices[6]], // Tet 3
            vec![cell_vertices[3], cell_vertices[4], cell_vertices[6], cell_vertices[7]], // Tet 4
            vec![cell_vertices[1], cell_vertices[4], cell_vertices[5], cell_vertices[6]], // Tet 5
        ];

        // Sum the volume of each tetrahedron
        let mut total_volume = 0.0;
        for tet in tets {
            total_volume += self.compute_tetrahedron_volume(&tet);
        }

        total_volume
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
        assert!((volume - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
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
        assert_eq!(centroid, [0.5, 0.5, 0.5]);
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
        assert_eq!(volume, 0.0);
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
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }
}
