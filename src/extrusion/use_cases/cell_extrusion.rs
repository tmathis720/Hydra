/// The `CellExtrusion` struct provides methods for extruding 2D cell structures (such as quadrilaterals and triangles)
/// into 3D volumetric cells (such as hexahedrons and prisms) across multiple layers.
///
/// This is a core part of mesh generation for 3D modeling, where cells from a 2D mesh are extended
/// in the z-direction to create volumetric cells.
///
/// # Example
///
/// ```
/// let quad_cells = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
/// let layers = 3;
/// let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells, layers);
/// ```
pub struct CellExtrusion;

impl CellExtrusion {
    /// Extrudes a vector of quadrilateral cells into hexahedral cells across multiple layers.
    ///
    /// Each quadrilateral cell is defined by four vertices `[v0, v1, v2, v3]`, which represent
    /// the cell's vertices in a counter-clockwise or clockwise order. This method extrudes each
    /// cell along the z-axis to create a hexahedral cell with eight vertices.
    ///
    /// # Parameters
    ///
    /// - `cells`: A `Vec<Vec<usize>>` where each inner vector represents a quadrilateral cell by storing
    ///   four vertex indices.
    /// - `layers`: An `usize` representing the number of layers to extrude. Each cell will be extended
    ///   into `layers` hexahedral cells.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` where each inner vector represents a hexahedral cell, defined by eight vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let quad_cells = vec![vec![0, 1, 2, 3]];
    /// let layers = 2;
    /// let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells, layers);
    ///
    /// assert_eq!(extruded_cells.len(), 2); // Two layers of hexahedral cells
    /// ```
    pub fn extrude_quadrilateral_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>> {
        let mut extruded_cells = Vec::with_capacity(cells.len() * layers);

        for layer in 0..layers {
            let offset = layer * cells.len();
            let next_offset = (layer + 1) * cells.len();

            for cell in &cells {
                // Each quadrilateral cell [v0, v1, v2, v3] is extruded into a hexahedron with 8 vertices
                let hexahedron = vec![
                    offset + cell[0], offset + cell[1], offset + cell[2], offset + cell[3],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2], next_offset + cell[3],
                ];
                extruded_cells.push(hexahedron);
            }
        }

        extruded_cells
    }

    /// Extrudes a vector of triangular cells into prismatic cells across multiple layers.
    ///
    /// Each triangular cell is defined by three vertices `[v0, v1, v2]`, which represent
    /// the cell's vertices in a counter-clockwise or clockwise order. This method extrudes each
    /// cell along the z-axis to create a prismatic cell with six vertices.
    ///
    /// # Parameters
    ///
    /// - `cells`: A `Vec<Vec<usize>>` where each inner vector represents a triangular cell by storing
    ///   three vertex indices.
    /// - `layers`: An `usize` representing the number of layers to extrude. Each cell will be extended
    ///   into `layers` prismatic cells.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` where each inner vector represents a prismatic cell, defined by six vertices.
    ///
    /// # Example
    ///
    /// ```
    /// let tri_cells = vec![vec![0, 1, 2]];
    /// let layers = 2;
    /// let extruded_cells = CellExtrusion::extrude_triangular_cells(tri_cells, layers);
    ///
    /// assert_eq!(extruded_cells.len(), 2); // Two layers of prismatic cells
    /// ```
    pub fn extrude_triangular_cells(cells: Vec<Vec<usize>>, layers: usize) -> Vec<Vec<usize>> {
        let mut extruded_cells = Vec::with_capacity(cells.len() * layers);

        for layer in 0..layers {
            let offset = layer * cells.len();
            let next_offset = (layer + 1) * cells.len();

            for cell in &cells {
                // Each triangular cell [v0, v1, v2] is extruded into a prism with 6 vertices
                let prism = vec![
                    offset + cell[0], offset + cell[1], offset + cell[2],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2],
                ];
                extruded_cells.push(prism);
            }
        }

        extruded_cells
    }
}

#[cfg(test)]
mod tests {
    use super::CellExtrusion;

    #[test]
    /// Test extrusion of quadrilateral cells across multiple layers.
    /// This test verifies that each quadrilateral cell is correctly transformed into a hexahedral cell
    /// and that the expected number of extruded cells are generated.
    fn test_extrude_quadrilateral_cells() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 3;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);

        // Expect 3 hexahedral layers for each quadrilateral cell
        assert_eq!(extruded_cells.len(), quad_cells.len() * layers);

        // Check the structure of the extruded hexahedral cells
        for layer in 0..layers {
            let offset = layer * quad_cells.len();
            let next_offset = (layer + 1) * quad_cells.len();

            for (i, cell) in quad_cells.iter().enumerate() {
                let hexahedron = extruded_cells[layer * quad_cells.len() + i].clone();
                assert_eq!(hexahedron, vec![
                    offset + cell[0], offset + cell[1], offset + cell[2], offset + cell[3],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2], next_offset + cell[3],
                ]);
            }
        }
    }

    #[test]
    /// Test extrusion of triangular cells across multiple layers.
    /// This test checks that each triangular cell is transformed into a prismatic cell
    /// and that the correct number of extruded cells are produced.
    fn test_extrude_triangular_cells() {
        let tri_cells = vec![vec![0, 1, 2]];
        let layers = 2;

        let extruded_cells = CellExtrusion::extrude_triangular_cells(tri_cells.clone(), layers);

        // Expect 2 prismatic layers for each triangular cell
        assert_eq!(extruded_cells.len(), tri_cells.len() * layers);

        // Check the structure of the extruded prismatic cells
        for layer in 0..layers {
            let offset = layer * tri_cells.len();
            let next_offset = (layer + 1) * tri_cells.len();

            for (i, cell) in tri_cells.iter().enumerate() {
                let prism = extruded_cells[layer * tri_cells.len() + i].clone();
                assert_eq!(prism, vec![
                    offset + cell[0], offset + cell[1], offset + cell[2],
                    next_offset + cell[0], next_offset + cell[1], next_offset + cell[2],
                ]);
            }
        }
    }

    #[test]
    /// Test extrusion with a single layer for quadrilateral cells.
    /// This ensures that the function handles single-layer extrusion correctly.
    fn test_single_layer_quadrilateral_extrusion() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 1;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);

        assert_eq!(extruded_cells.len(), quad_cells.len()); // Only one layer should be extruded

        // Verify that the single layer's z-offset behaves as expected
        let hexahedron = extruded_cells[0].clone();
        assert_eq!(hexahedron, vec![
            0, 1, 2, 3,
            quad_cells.len() + 0, quad_cells.len() + 1, quad_cells.len() + 2, quad_cells.len() + 3,
        ]);
    }

    #[test]
    /// Test extrusion with zero layers to ensure it gracefully handles edge cases without error.
    fn test_zero_layers_extrusion() {
        let quad_cells = vec![vec![0, 1, 2, 3]];
        let layers = 0;

        let extruded_cells = CellExtrusion::extrude_quadrilateral_cells(quad_cells.clone(), layers);
        assert!(extruded_cells.is_empty(), "Extruded cells should be empty when layers is zero.");
    }
}
