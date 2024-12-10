use super::Mesh;
use crate::domain;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;
use crate::domain::section::Vector3;

impl Mesh {
    /// Retrieves all the faces of a given cell.
    ///
    /// This function collects entities connected to the provided cell and filters
    /// them to include only face entities. The result is returned as a `DashMap`.
    ///
    /// # Arguments
    /// * `cell` - A `MeshEntity` representing the cell whose faces are being retrieved.
    ///
    /// # Returns
    /// * `Option<DashMap<MeshEntity, ()>>` - A map of face entities connected to the cell.
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<DashMap<MeshEntity, ()>> {
        self.sieve.cone(cell).map(|set| {
            let faces = DashMap::new();
            set.into_iter()
                .filter(|entity| matches!(entity, MeshEntity::Face(_)))
                .for_each(|face| {
                    faces.insert(face, ());
                });
            faces
        })
    }

    /// Retrieves all the cells that share a given face.
    ///
    /// This function identifies all cell entities that share a specified face,
    /// filtering only valid cell entities present in the mesh.
    ///
    /// # Arguments
    /// * `face` - A `MeshEntity` representing the face.
    ///
    /// # Returns
    /// * `DashMap<MeshEntity, ()>` - A map of cell entities sharing the face.
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let cells = DashMap::new();
        let entities = self.entities.read().unwrap();
        self.sieve
            .support(face)
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
            .for_each(|cell| {
                cells.insert(cell, ());
            });
        cells
    }

    /// Computes the Euclidean distance between two cells based on their centroids.
    ///
    /// # Arguments
    /// * `cell_i` - The first cell entity.
    /// * `cell_j` - The second cell entity.
    ///
    /// # Returns
    /// * `f64` - The computed distance between the centroids of the two cells.
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the area.
    ///
    /// # Returns
    /// * `Option<f64>` - The area of the face, or `None` if the face shape is unsupported.
    pub fn get_face_area(&self, face: &MeshEntity) -> Option<f64> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        Some(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to compute the centroid.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the cell's centroid.
    ///
    /// # Panics
    /// This function panics if the cell has an unsupported number of vertices.
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to a given vertex via shared cells.
    ///
    /// # Arguments
    /// * `vertex` - The vertex entity for which to find neighboring vertices.
    ///
    /// # Returns
    /// * `Vec<MeshEntity>` - A list of vertex entities neighboring the given vertex.
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let neighbors = DashMap::new();
        let connected_cells = self.sieve.support(vertex);

        connected_cells.into_iter().for_each(|cell| {
            if let Some(cell_vertices) = self.sieve.cone(&cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(v.clone(), ());
                    }
                }
            }
        });
        neighbors.into_iter().map(|(vertex, _)| vertex).collect()
    }

    /// Returns an iterator over all vertex IDs in the mesh.
    ///
    /// # Returns
    /// * `impl Iterator<Item = &usize>` - An iterator over vertex IDs.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on its vertex count.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to determine the shape.
    ///
    /// # Returns
    /// * `Result<CellShape, String>` - The determined cell shape or an error message if unsupported.
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell);
        match cell_vertices.len() {
            4 => Ok(CellShape::Tetrahedron),
            5 => Ok(CellShape::Pyramid),
            6 => Ok(CellShape::Prism),
            8 => Ok(CellShape::Hexahedron),
            _ => Err(format!(
                "Unsupported cell shape with {} vertices. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len()
            )),
        }
    }

    /// Retrieves the vertices of a cell, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `cell` - The cell entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Vec<[f64; 3]>` - The 3D coordinates of the cell's vertices, sorted by ID.
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Retrieves the vertices of a face, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `face` - The face entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Vec<[f64; 3]>` - The 3D coordinates of the face's vertices, sorted by ID.
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Computes the outward normal vector for a face based on its shape and vertices.
    ///
    /// Optionally adjusts the normal's orientation based on a reference cell's centroid.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the normal.
    /// * `reference_cell` - An optional reference cell entity to adjust the orientation.
    ///
    /// # Returns
    /// * `Option<Vector3>` - The computed normal vector, or `None` if the face shape is unsupported.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Option<Vector3> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Edge => geometry.compute_edge_normal(&face_vertices),
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // Adjust normal orientation if a reference cell is provided
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell);
            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal if it points inward
                return Some(domain::section::Vector3([-normal[0], -normal[1], -normal[2]]));
            }
        }

        Some(domain::section::Vector3(normal))
    }
}
