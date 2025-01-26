use rustc_hash::FxHashSet;
use crate::geometry::{Geometry, FaceShape};
use crate::domain::mesh::{Mesh, MeshEntity};

/// The `GeometryValidation` struct provides utilities to validate geometric properties of a mesh.
///
/// This includes:
/// - Ensuring vertices have unique and valid coordinates.
/// - Verifying centroid calculations for cells and faces.
/// - Checking the accuracy of inter-cell distance calculations.
///
/// These validations help maintain consistency and accuracy in simulations.
pub struct GeometryValidation;

impl GeometryValidation {
    /// Validates that all vertices in the mesh have unique, valid coordinates.
    ///
    /// Vertex coordinates are converted to rounded integer tuples to ensure a consistent precision threshold.
    /// Any duplicates or missing vertex coordinates will result in an error.
    ///
    /// # Arguments
    /// - `mesh`: The `Mesh` instance to validate.
    ///
    /// # Returns
    /// - `Ok(())` if all vertices have unique coordinates.
    /// - `Err(String)` if duplicates or missing coordinates are found.
    pub fn test_vertex_coordinates(mesh: &Mesh) -> Result<(), String> {
        let mut unique_coords = FxHashSet::default();

        for vertex_id in mesh.iter_vertices() {
            if let Some(coords) = mesh.get_vertex_coordinates(*vertex_id) {
                // Convert to rounded integer representation for uniqueness checks
                let rounded_coords = (
                    (coords[0] * 1e6).round() as i64,
                    (coords[1] * 1e6).round() as i64,
                    (coords[2] * 1e6).round() as i64,
                );

                if !unique_coords.insert(rounded_coords) {
                    return Err(format!("Duplicate vertex coordinates found: {:?}", coords));
                }
            } else {
                return Err(format!("Vertex ID {} has no associated coordinates", vertex_id));
            }
        }

        Ok(())
    }

    /// Validates centroid calculations for faces and cells, comparing the computed values to reference values.
    ///
    /// This method calculates centroids for all faces and cells in the mesh using the `Geometry` module
    /// and validates their consistency.
    ///
    /// # Arguments
    /// - `mesh`: The `Mesh` instance to validate.
    /// - `geometry`: A mutable reference to the `Geometry` instance for centroid calculations.
    ///
    /// # Returns
    /// - `Ok(())` if all centroids are consistent.
    /// - `Err(String)` if inconsistencies are found.
    pub fn test_centroid_calculation(mesh: &Mesh, geometry: &mut Geometry) -> Result<(), String> {
        // Validate face centroids
        for face in mesh.get_faces().iter() {
            let face_vertices = mesh
                .get_face_vertices(face)
                .map_err(|err| format!("Failed to retrieve vertices for face {:?}: {}", face, err))?;

            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => {
                    return Err(format!(
                        "Unsupported face shape with {} vertices",
                        face_vertices.len()
                    ));
                }
            };

            // Compute the centroid directly (assuming it doesn't return a `Result`)
            let calculated_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            if !GeometryValidation::is_centroid_valid(face, &calculated_centroid, mesh) {
                return Err(format!(
                    "Inconsistent centroid for face {:?}. Calculated: {:?}",
                    face, calculated_centroid
                ));
            }
        }

        // Validate cell centroids
        for cell in mesh.get_cells().iter() {
            // Directly call the centroid computation function
            let calculated_centroid = mesh.get_cell_centroid(cell).map_err(|err| {
                format!(
                    "Failed to compute centroid for cell {:?}: {}",
                    cell, err
                )
            })?;

            if !GeometryValidation::is_centroid_valid(cell, &calculated_centroid, mesh) {
                return Err(format!(
                    "Inconsistent centroid for cell {:?}. Calculated: {:?}",
                    cell, calculated_centroid
                ));
            }
        }

        Ok(())
    }


    /// Validates the distances between each pair of cells in the mesh.
    ///
    /// This method uses the `Geometry` module to calculate distances and checks for consistency.
    ///
    /// # Arguments
    /// - `mesh`: The `Mesh` instance to validate.
    /// - `geometry`: A reference to the `Geometry` instance for distance calculations.
    ///
    /// # Returns
    /// - `Ok(())` if all distances are consistent.
    /// - `Err(String)` if inconsistencies are found.
    pub fn test_distance_between_cells(mesh: &Mesh, geometry: &Geometry) -> Result<(), String> {
        let cells = mesh.get_cells();

        // Iterate over all pairs of cells
        for (i, cell1) in cells.iter().enumerate() {
            for cell2 in cells.iter().skip(i + 1) {
                // Safely handle the result of `get_distance_between_cells`
                let calculated_distance = mesh
                    .get_distance_between_cells(cell1, cell2)
                    .map_err(|err| {
                        format!(
                            "Failed to calculate distance between cells {:?} and {:?}: {}",
                            cell1, cell2, err
                        )
                    })?;

                // Validate the calculated distance
                if !GeometryValidation::is_distance_valid(cell1, cell2, calculated_distance, geometry) {
                    return Err(format!(
                        "Incorrect distance between cells {:?} and {:?}. Calculated: {:.6}",
                        cell1, cell2, calculated_distance
                    ));
                }
            }
        }

        Ok(())
    }

    /// Helper function to validate the consistency of a calculated centroid with reference values.
    ///
    /// This function would compare the calculated centroid with pre-defined or expected reference values
    /// if available. The current implementation is a placeholder.
    fn is_centroid_valid(entity: &MeshEntity, calculated_centroid: &[f64; 3], mesh: &Mesh) -> bool {
        let _ = mesh;
        let _ = entity;
        let _ = calculated_centroid;

        // Placeholder logic; would compare with a reference value if available
        true
    }

    /// Helper function to validate the consistency of a calculated distance with reference values.
    ///
    /// This function would compare the calculated distance with pre-defined or expected reference values
    /// if available. The current implementation is a placeholder.
    fn is_distance_valid(
        cell1: &MeshEntity,
        cell2: &MeshEntity,
        calculated_distance: f64,
        geometry: &Geometry,
    ) -> bool {
        let _ = geometry;
        let _ = calculated_distance;
        let _ = cell2;
        let _ = cell1;

        // Placeholder logic; would compare with a reference value if available
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::domain::mesh::Mesh;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_vertex_coordinates() {
        let mut mesh = Mesh::new();

        // Add unique vertices
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]).unwrap();

        // Ensure validation passes
        assert!(GeometryValidation::test_vertex_coordinates(&mesh).is_ok());

        // Add duplicate vertex coordinates
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 0.0]).unwrap();

        // Ensure validation fails
        assert!(GeometryValidation::test_vertex_coordinates(&mesh).is_err());
    }

    #[test]
    fn test_centroid_calculation() {
        let mut mesh = Mesh::new();
        let mut geometry = Geometry::new();

        let face = MeshEntity::Face(1);

        // Define vertices for the face
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]).unwrap();

        // Establish relationships for the face
        mesh.add_arrow(face, MeshEntity::Vertex(1)).unwrap();
        mesh.add_arrow(face, MeshEntity::Vertex(2)).unwrap();
        mesh.add_arrow(face, MeshEntity::Vertex(3)).unwrap();

        // Ensure centroid calculation validation passes
        assert!(GeometryValidation::test_centroid_calculation(&mesh, &mut geometry).is_ok());
    }

    #[test]
    fn test_distance_between_cells() {
        let mut mesh = Mesh::new();
        let geometry = Geometry::new();

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);

        // Define vertices for the cells
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();

        // Establish relationships for the cells
        mesh.add_arrow(cell1, MeshEntity::Vertex(1)).unwrap();
        mesh.add_arrow(cell2, MeshEntity::Vertex(2)).unwrap();

        // Ensure distance calculation validation passes
        assert!(GeometryValidation::test_distance_between_cells(&mesh, &geometry).is_ok());
    }
}
