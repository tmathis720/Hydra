/// The `VertexExtrusion` struct provides methods for extruding a set of vertices along the z-axis.
/// This extrusion process is commonly used in mesh generation for three-dimensional models, where
/// a 2D base layer is extended in the z-direction to create a volumetric representation.
///
/// # Example
///
/// ```rust
/// use hydra::extrusion::use_cases::vertex_extrusion::VertexExtrusion;
/// let vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
/// let depth = 10.0;
/// let layers = 5;
/// let extruded = VertexExtrusion::extrude_vertices(vertices, depth, layers);
/// ```
pub struct VertexExtrusion;

impl VertexExtrusion {
    /// Extrudes vertices along the z-axis, creating multiple layers of vertices based on
    /// the specified `depth` and number of `layers`.
    ///
    /// This function takes a set of base vertices defined in the XY plane (z = 0) and extrudes them
    /// along the z-axis to generate additional layers at regularly spaced intervals, forming a 
    /// three-dimensional structure.
    ///
    /// # Parameters
    ///
    /// - `vertices`: A `Vec<[f64; 3]>` representing the base vertices, each with an initial z-coordinate.
    /// - `depth`: A `f64` representing the total depth of the extrusion in the z-direction.
    /// - `layers`: An `usize` specifying the number of layers to generate. The function divides
    ///    the depth by this number to determine the z-coordinate increment (`dz`) between each layer.
    ///
    /// # Returns
    ///
    /// A `Vec<[f64; 3]>` containing the extruded vertices. The z-coordinate of each new layer
    /// increases by `dz` until reaching `depth`, thus forming layers from z = 0 to z = `depth`.
    ///
    /// # Panics
    ///
    /// This function does not panic as long as `layers > 0` (to avoid division by zero). If `layers`
    /// is zero, the caller should handle the case to prevent an undefined extrusion.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::extrusion::use_cases::vertex_extrusion::VertexExtrusion;
    /// let vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
    /// let depth = 10.0;
    /// let layers = 2;
    /// let extruded_vertices = VertexExtrusion::extrude_vertices(vertices, depth, layers);
    ///
    /// assert_eq!(extruded_vertices.len(), 6); // 3 layers x 2 vertices per layer
    /// ```
    pub fn extrude_vertices(vertices: Vec<[f64; 3]>, depth: f64, layers: usize) -> Vec<[f64; 3]> {
        let dz = depth / layers as f64;
        let mut extruded_vertices = Vec::with_capacity(vertices.len() * (layers + 1));

        for layer in 0..=layers {
            let z = dz * layer as f64;
            for vertex in &vertices {
                extruded_vertices.push([vertex[0], vertex[1], z]);
            }
        }

        extruded_vertices
    }
}

#[cfg(test)]
mod tests {
    use super::VertexExtrusion;

    #[test]
    /// Test that verifies the correct number of extruded vertices are generated for the
    /// specified layers, and that the z-coordinates increment correctly in each layer.
    fn test_extrusion_with_multiple_layers() {
        // Define a small set of vertices in the XY plane (z=0)
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 10.0;
        let layers = 5;

        // Perform the extrusion
        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        // The expected number of vertices is the original vertices count times the number of layers plus one
        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Check that each layer has the correct z-coordinate incrementally
        let dz = depth / layers as f64;
        for (i, z) in (0..=layers).map(|layer| dz * layer as f64).enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - z).abs() < 1e-6, "Incorrect z-coordinate in layer");
            }
        }
    }

    #[test]
    /// Test that verifies the function correctly extrudes vertices for a single layer,
    /// producing two sets of vertices: one at the base layer (z=0) and one at z=depth.
    fn test_extrusion_with_one_layer() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 5.0;
        let layers = 1;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * 2); // Two layers

        // Verify z-coordinates for each layer
        for (i, z) in [0.0, depth].iter().enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - *z).abs() < 1e-6, "Incorrect z-coordinate for one layer extrusion");
            }
        }
    }

    #[test]
    /// Test that verifies the extrusion with multiple vertices and zero depth,
    /// resulting in no change along the z-axis across all layers.
    fn test_extrusion_with_zero_depth() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 0.0;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Check that all extruded vertices have a z-coordinate of 0.0
        for extruded_vertex in extruded_vertices {
            assert_eq!(extruded_vertex[2], 0.0, "Extrusion with zero depth should have z=0 for all vertices");
        }
    }

    #[test]
    /// Test that verifies that an empty vertex list returns an empty extruded vertex list,
    /// ensuring no extraneous vertices are created when no input is given.
    fn test_extrusion_with_empty_vertices() {
        let base_vertices: Vec<[f64; 3]> = vec![];
        let depth = 5.0;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert!(extruded_vertices.is_empty(), "Extrusion with empty vertices should result in empty output");
    }

    #[test]
    /// Test that checks the precision of extruded z-coordinates to ensure they are calculated
    /// correctly for non-integer values of `depth` and `layers`.
    fn test_extrusion_with_decimal_depth() {
        let base_vertices = vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]];
        let depth = 3.75;
        let layers = 3;

        let extruded_vertices = VertexExtrusion::extrude_vertices(base_vertices.clone(), depth, layers);

        assert_eq!(extruded_vertices.len(), base_vertices.len() * (layers + 1));

        // Verify that the z-coordinates increment by depth/layers, with precision for decimal depth
        let dz = depth / layers as f64;
        for (i, z) in (0..=layers).map(|layer| dz * layer as f64).enumerate() {
            for j in 0..base_vertices.len() {
                let vertex_index = i * base_vertices.len() + j;
                let extruded_vertex = extruded_vertices[vertex_index];
                assert_eq!(extruded_vertex[0], base_vertices[j][0]);
                assert_eq!(extruded_vertex[1], base_vertices[j][1]);
                assert!((extruded_vertex[2] - z).abs() < 1e-6, "Incorrect z-coordinate with decimal depth");
            }
        }
    }
}
