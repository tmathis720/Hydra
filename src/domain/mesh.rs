use rustc_hash::{FxHashMap, FxHashSet};
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use crate::geometry::{Geometry, CellShape, FaceShape};  // Import geometry module

#[derive(Clone)]
pub struct Mesh {
    pub sieve: Sieve,                         // Sieve to handle hierarchical relationships
    pub entities: FxHashSet<MeshEntity>,        // Set of all entities in the mesh
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>, // Mapping from vertex IDs to coordinates
}

impl Mesh {
    /// Create a new empty mesh
    pub fn new() -> Self {
        Mesh {
            sieve: Sieve::new(),
            entities: FxHashSet::default(),
            vertex_coordinates: FxHashMap::default(),
        }
    }

    /// Add a new entity to the mesh (vertex, edge, face, or cell)
    pub fn add_entity(&mut self, entity: MeshEntity) {
        self.entities.insert(entity);
    }

    /// Add a relationship between two entities
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Set coordinates for a vertex
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Get coordinates of a vertex
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    // Get all cells in the mesh
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        self.entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    // Get all faces in the mesh
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        self.entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    // Get faces of a cell
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<&FxHashSet<MeshEntity>> {
        self.sieve.cone(cell)
    }

    // Get cells sharing a face
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> FxHashSet<MeshEntity> {
        self.sieve.support(face)
    }

    /// Get face area (requires geometric data)
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        // Retrieve face vertices
        let face_vertices = self.get_face_vertices(face);

        // Determine face shape based on the number of vertices
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };

        // Compute face area using the geometry module
        let geometry = Geometry::new();
        geometry.compute_face_area(face_shape, &face_vertices)
    }

    /// Get distance between cell centers (requires geometric data)
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        // Get cell centroids
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);

        // Compute distance using geometry module
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Get distance from cell center to boundary face
    pub fn get_distance_to_boundary(&self, cell: &MeshEntity, face: &MeshEntity) -> f64 {
        // Get cell centroid
        let centroid = self.get_cell_centroid(cell);

        // Get face centroid
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };
        let geometry = Geometry::new();
        let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

        // Compute distance
        Geometry::compute_distance(&centroid, &face_centroid)
    }

    /// Get cell centroid
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        // Get cell vertices
        let cell_vertices = self.get_cell_vertices(cell);

        // Determine cell shape based on number of vertices
        let cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        // Compute cell centroid
        let geometry = Geometry::new();
        geometry.compute_cell_centroid(cell_shape, &cell_vertices)
    }

    /// Get cell vertices
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        // Retrieve faces connected to the cell
        let mut vertices = Vec::new();
        if let Some(connected_faces) = self.sieve.cone(cell) {
            for face in connected_faces {
                // Get vertices of the face
                let face_vertices = self.get_face_vertices(face);
                vertices.extend(face_vertices);
            }
            // Remove duplicate vertices
            vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vertices.dedup();
        }
        vertices
    }

    /// Get face vertices
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        // Retrieve vertices connected to the face
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(*vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }
        vertices
    }
}
