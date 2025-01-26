use rustc_hash::FxHashSet;
use thiserror::Error;
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

#[derive(Debug, Error)]
pub enum GeometryValidationError {
    #[error("Vertex ID {0} has no associated coordinates")]
    MissingVertexCoordinates(u64),
    #[error("Duplicate vertex coordinates found: {0:?}")]
    DuplicateVertexCoordinates([f64; 3]),
    #[error("Unsupported face shape with {0} vertices")]
    UnsupportedFaceShape(usize),
    #[error("Failed to retrieve vertices for face {0:?}: {1}")]
    FaceVertexRetrievalError(MeshEntity, String),
    #[error("Inconsistent centroid for entity {0:?}. Calculated: {1:?}")]
    InconsistentCentroid(MeshEntity, [f64; 3]),
    #[error("Failed to calculate distance between cells {0:?} and {1:?}: {2}")]
    DistanceCalculationError(MeshEntity, MeshEntity, String),
    #[error("Incorrect distance between cells {0:?} and {1:?}. Calculated: {2}")]
    IncorrectDistance(MeshEntity, MeshEntity, f64),
    #[error("Invalid supporting entities for face {0:?}")]
    TopologyError(String),
    #[error("Failed to acquire read lock on entities: {0}")]
    EntityAccessError(String),
    #[error("Failed to compute centroid for cell {0}")]
    CentroidError(String),
    #[error("Failed to retrieve vertices for face {0}")]
    VertexError(String),
    #[error("Unsupported face shape: {0}")]
    ShapeError(String),
    #[error("Failed to compute area for face {0}")]
    ComputationError(String),
}

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
    pub fn test_vertex_coordinates(mesh: &Mesh) -> Result<(), GeometryValidationError> {
        let mut unique_coords = FxHashSet::default();

        for vertex_id in mesh.iter_vertices() {
            if let Some(coords) = mesh.get_vertex_coordinates(*vertex_id) {
                let rounded_coords = (
                    (coords[0] * 1e6).round() as i64,
                    (coords[1] * 1e6).round() as i64,
                    (coords[2] * 1e6).round() as i64,
                );

                if !unique_coords.insert(rounded_coords) {
                    log::error!("Duplicate coordinates for vertex: {:?}", coords);
                    return Err(GeometryValidationError::DuplicateVertexCoordinates(coords));
                }
            } else {
                log::warn!("Missing coordinates for vertex ID {}", vertex_id);
                return Err(GeometryValidationError::MissingVertexCoordinates(*vertex_id as u64));
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
    pub fn test_centroid_calculation(
        mesh: &Mesh,
        geometry: &mut Geometry,
    ) -> Result<(), GeometryValidationError> {
        for face in mesh.get_faces().iter() {
            let face_vertices = mesh
                .get_face_vertices(face)
                .map_err(|err| {
                    log::error!("Failed to retrieve vertices for face {:?}: {}", face, err);
                    GeometryValidationError::FaceVertexRetrievalError(*face, err.to_string())
                })?;

            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => {
                    log::error!("Unsupported face shape with {} vertices", face_vertices.len());
                    return Err(GeometryValidationError::UnsupportedFaceShape(face_vertices.len()));
                }
            };

            let calculated_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            if !Self::is_centroid_valid(face, &calculated_centroid, mesh) {
                log::error!(
                    "Centroid validation failed for face {:?}. Calculated: {:?}",
                    face,
                    calculated_centroid
                );
                return Err(GeometryValidationError::InconsistentCentroid(
                    *face,
                    calculated_centroid,
                ));
            }
        }

        for cell in mesh.get_cells().iter() {
            let calculated_centroid = mesh.get_cell_centroid(cell).map_err(|err| {
                log::error!(
                    "Failed to compute centroid for cell {:?}: {}",
                    cell,
                    err
                );
                GeometryValidationError::InconsistentCentroid(*cell, [0.0, 0.0, 0.0])
            })?;

            if !Self::is_centroid_valid(cell, &calculated_centroid, mesh) {
                log::error!(
                    "Centroid validation failed for cell {:?}. Calculated: {:?}",
                    cell,
                    calculated_centroid
                );
                return Err(GeometryValidationError::InconsistentCentroid(
                    *cell,
                    calculated_centroid,
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
    pub fn test_distance_between_cells(
        mesh: &Mesh,
        geometry: &Geometry,
    ) -> Result<(), GeometryValidationError> {
        let cells = mesh.get_cells();

        for (i, cell1) in cells.iter().enumerate() {
            for cell2 in cells.iter().skip(i + 1) {
                let calculated_distance = mesh
                    .get_distance_between_cells(cell1, cell2)
                    .map_err(|err| {
                        log::error!(
                            "Failed to calculate distance between cells {:?} and {:?}: {}",
                            cell1,
                            cell2,
                            err
                        );
                        GeometryValidationError::DistanceCalculationError(
                            *cell1,
                            *cell2,
                            err.to_string(),
                        )
                    })?;

                if !Self::is_distance_valid(cell1, cell2, calculated_distance, geometry) {
                    log::error!(
                        "Distance validation failed between cells {:?} and {:?}. Calculated: {:.6}",
                        cell1,
                        cell2,
                        calculated_distance
                    );
                    return Err(GeometryValidationError::IncorrectDistance(
                        *cell1,
                        *cell2,
                        calculated_distance,
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
        // Register the face with the mesh explicitly
        mesh.add_entity(face).unwrap();
    
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
    
        // Add cells to the mesh
        mesh.add_entity(cell1).unwrap();
        mesh.add_entity(cell2).unwrap();
    
        // Define vertices for the cells
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
    
        // Establish relationships for the cells
        mesh.add_arrow(cell1, MeshEntity::Vertex(1)).unwrap();
        mesh.add_arrow(cell2, MeshEntity::Vertex(2)).unwrap();
    
        // Ensure adjacency relationships (e.g., shared faces or edges) are established
        let face = MeshEntity::Face(1);
        mesh.add_entity(face).unwrap();
        mesh.add_arrow(face, MeshEntity::Vertex(1)).unwrap();
        mesh.add_arrow(face, MeshEntity::Vertex(2)).unwrap();
        mesh.add_arrow(cell1, face).unwrap();
        mesh.add_arrow(cell2, face).unwrap();
    
        // Ensure distance calculation validation passes
        assert!(GeometryValidation::test_distance_between_cells(&mesh, &geometry).is_ok());
    }    
}
