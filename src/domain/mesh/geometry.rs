use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;

impl Mesh {
    /// Retrieves all the faces of a given cell, filtering only face entities.
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

    /// Retrieves all the cells that share the given face, filtering only cell entities that are present in the mesh.
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
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.
    pub fn get_face_area(&self, face: &MeshEntity) -> Option<f64> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        Some(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
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

    /// Retrieves all vertices connected to the given vertex by shared cells.
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

    /// Returns an iterator over the IDs of all vertices in the mesh.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.
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

    /// Retrieves the vertices of a cell and their coordinates, sorted by vertex ID.
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

    /// Retrieves the vertices of a face and their coordinates, sorted by vertex ID.
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

    /// Computes the normal vector of a face based on its vertices and shape.
    ///
    /// This function calculates the outward normal vector for a face by leveraging the
    /// `Geometry` module. It determines the face shape and uses the vertices to compute
    /// the vector. The orientation of the normal can optionally depend on a neighboring cell.
    ///
    /// # Arguments
    /// * `face` - The face entity for which the normal is computed.
    /// * `reference_cell` - Optional cell entity to determine the normal orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector if successful, otherwise `None`.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Option<[f64; 3]> {
        // Retrieve face vertices
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // If a reference cell is provided, adjust the normal's orientation
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell);
            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            // Compute the vector from the face centroid to the cell centroid
            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            // Ensure the normal points outward by checking the dot product
            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal direction to make it outward-pointing
                return Some([-normal[0], -normal[1], -normal[2]]);
            }
        }

        Some(normal)
    }
}
