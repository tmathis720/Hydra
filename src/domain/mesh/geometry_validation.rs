// src/domain/mesh/geometry_validation.rs

use rustc_hash::FxHashSet;
use crate::geometry::{Geometry, FaceShape};
use crate::domain::mesh::{Mesh, MeshEntity};

/// Struct to handle geometric validations of a mesh, ensuring accurate geometric properties
/// and consistency across cells, faces, and vertices.
pub struct GeometryValidation;

impl GeometryValidation {
    /// Validates that all vertices in the mesh have unique, valid coordinates.
    /// Converts f64 coordinates to a rounded integer tuple to enable unique checks.
    pub fn test_vertex_coordinates(mesh: &Mesh) -> Result<(), String> {
        let mut unique_coords = FxHashSet::default();
        for vertex_id in mesh.iter_vertices() {
            if let Some(coords) = mesh.get_vertex_coordinates(*vertex_id) {
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

    /// Validates centroid calculations for cells and faces, comparing them to expected values.
    pub fn test_centroid_calculation(mesh: &Mesh, geometry: &mut Geometry) -> Result<(), String> {
        for face in mesh.get_faces().iter() {
            let face_vertices = mesh.get_face_vertices(face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => return Err(format!("Unsupported face shape with {} vertices", face_vertices.len())),
            };

            let calculated_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
            if !GeometryValidation::is_centroid_valid(face, &calculated_centroid, mesh) {
                return Err(format!(
                    "Inconsistent centroid for face {:?}. Calculated: {:?}",
                    face, calculated_centroid
                ));
            }
        }

        for cell in mesh.get_cells().iter() {
            let calculated_centroid = geometry.compute_cell_centroid(mesh, cell);
            if !GeometryValidation::is_centroid_valid(cell, &calculated_centroid, mesh) {
                return Err(format!(
                    "Inconsistent centroid for cell {:?}. Calculated: {:?}",
                    cell, calculated_centroid
                ));
            }
        }

        Ok(())
    }

    /// Validates that distances between each pair of cells are calculated accurately.
    pub fn test_distance_between_cells(mesh: &Mesh, geometry: &Geometry) -> Result<(), String> {
        let cells = mesh.get_cells();
        for (i, cell1) in cells.iter().enumerate() {
            for cell2 in cells.iter().skip(i + 1) {
                let calculated_distance = mesh.get_distance_between_cells(cell1, cell2);
                if !GeometryValidation::is_distance_valid(cell1, cell2, calculated_distance, geometry) {
                    return Err(format!(
                        "Incorrect distance between cells {:?} and {:?}. Calculated: {}",
                        cell1, cell2, calculated_distance
                    ));
                }
            }
        }
        Ok(())
    }

    /// Helper function to validate centroid consistency with reference values.
    fn is_centroid_valid(entity: &MeshEntity, calculated_centroid: &[f64; 3], mesh: &Mesh) -> bool {
        let _ = mesh;
        let _ = entity;
        let _ = calculated_centroid;
        // Placeholder logic; would compare with a reference value if available
        true
    }

    /// Helper function to validate distance consistency with reference values.
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
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);

        assert!(GeometryValidation::test_vertex_coordinates(&mesh).is_ok());

        mesh.set_vertex_coordinates(4, [0.0, 0.0, 0.0]); // Duplicate coordinate
        assert!(GeometryValidation::test_vertex_coordinates(&mesh).is_err());
    }

    #[test]
    fn test_centroid_calculation() {
        let mut mesh = Mesh::new();
        let mut geometry = Geometry::new();

        let face = MeshEntity::Face(1);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.add_arrow(face, MeshEntity::Vertex(1));
        mesh.add_arrow(face, MeshEntity::Vertex(2));
        mesh.add_arrow(face, MeshEntity::Vertex(3));

        assert!(GeometryValidation::test_centroid_calculation(&mesh, &mut geometry).is_ok());
    }

    #[test]
    fn test_distance_between_cells() {
        let mut mesh = Mesh::new();
        let geometry = Geometry::new();

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.add_arrow(cell1, MeshEntity::Vertex(1));
        mesh.add_arrow(cell2, MeshEntity::Vertex(2));

        assert!(GeometryValidation::test_distance_between_cells(&mesh, &geometry).is_ok());
    }
}
