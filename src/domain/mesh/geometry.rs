use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use rustc_hash::FxHashSet;

impl Mesh {
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        self.sieve.cone(cell).map(|set| set.clone())
    }

    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> FxHashSet<MeshEntity> {
        self.sieve.support(face)
    }

    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Get face area (requires geometric data)
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };

        // Use a new or existing geometry instance, providing a unique ID for caching.
        let mut geometry = Geometry::new();
        let face_id = face.id(); // Assuming `id` method provides a unique ID for the face.
        geometry.compute_face_area(face_id, face_shape, &face_vertices)
    }

    /// Get cell centroid
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        // Use a new or existing Geometry instance, providing a unique ID for caching.
        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = FxHashSet::default();
        let connected_cells = self.sieve.support(vertex);

        for cell in &connected_cells {
            if let Some(cell_vertices) = self.sieve.cone(cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(*v);
                    }
                }
            } else {
                panic!("Cell {:?} has no connected vertices", cell);
            }
        }
        neighbors.into_iter().collect()
    }

    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Get the shape of a cell based on its number of vertices.
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

    /// Get the vertices of a cell.
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
    
        // Retrieve entities connected to the cell.
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                // Check if the entity is a vertex and extract its coordinates.
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }
    
        // Sort and deduplicate vertices to ensure unique entries.
        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vertices.dedup();
        vertices
    }

    /// Get the vertices of a face.
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    }
                }
            }
        }
        vertices
    }
}
