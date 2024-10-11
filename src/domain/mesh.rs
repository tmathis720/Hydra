use rustc_hash::{FxHashMap, FxHashSet};
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use crate::geometry::{Geometry, CellShape, FaceShape};  // Import geometry module
use std::sync::{Arc, RwLock};
use rayon::iter::*;
use crossbeam::thread;
use crossbeam::channel::{Sender, Receiver};
use crossbeam::channel;

#[derive(Clone)]
pub struct Mesh {
    pub sieve: Arc<Sieve>,                         // Sieve to handle hierarchical relationships
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,      // Set of all entities in the mesh
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>, // Mapping from vertex IDs to coordinates
    boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,
    boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,
}

impl Mesh {
    /// Create a new empty mesh
    pub fn new() -> Self {
        let (sender, receiver) = channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }

    /// Add a new entity to the mesh (vertex, edge, face, or cell)
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    // Apply a function to all entities in parallel
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
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

    /// Get all cells in the mesh
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Get all faces in the mesh
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Get faces of a cell
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        self.sieve.cone(cell).map(|set| set.clone())
    }

    /// Get cells sharing a face
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> FxHashSet<MeshEntity> {
        self.sieve.support(face)
    }

    /// Get face area (requires geometric data)
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };
        let geometry = Geometry::new();
        geometry.compute_face_area(face_shape, &face_vertices)
    }

    /// Get distance between cell centers (requires geometric data)
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Get distance from cell center to boundary face
    pub fn get_distance_to_boundary(&self, cell: &MeshEntity, face: &MeshEntity) -> f64 {
        let centroid = self.get_cell_centroid(cell);
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };
        let geometry = Geometry::new();
        let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);
        Geometry::compute_distance(&centroid, &face_centroid)
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
        let geometry = Geometry::new();
        geometry.compute_cell_centroid(cell_shape, &cell_vertices)
    }

    /// Get cell vertices
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_faces) = self.sieve.cone(cell) {
            for face in connected_faces {
                let face_vertices = self.get_face_vertices(&face);
                vertices.extend(face_vertices);
            }
            vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vertices.dedup();
        }
        vertices
    }

    /// Get face vertices
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }
        vertices
    }

    /// Count the number of MeshEntities of a specific type
    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count()
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

    // Example method to compute some property for each entity in parallel
    pub fn compute_properties<F, PropertyType>(&self, compute_fn: F) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect()
    }

    // Synchronize boundary data using scoped threads
    pub fn sync_boundary_data(&mut self) {
        // Clone the entities and sieve references for thread-safe operations.
        let entities = self.entities.clone();
        let sieve = self.sieve.clone();

        // Prepare and send the boundary data first.
        self.send_boundary_data();

        // Use scoped threads to handle incoming data without overlapping mutable borrows.
        thread::scope(|s| {
            s.spawn(|_| {
                // Handle incoming boundary data.
                // Now, this operation occurs after sending, avoiding simultaneous mutable borrows.
                self.receive_boundary_data();
            });
        })
        .unwrap();
    }

    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender);
        self.boundary_data_receiver = Some(receiver);
    }

    fn receive_boundary_data(&mut self) {
        if let Some(ref receiver) = self.boundary_data_receiver {
            // Attempt to receive boundary data from the channel.
            if let Ok(boundary_data) = receiver.recv() {
                let mut entities = self.entities.write().unwrap();
                for (entity, coords) in boundary_data {
                    // Update the vertex coordinates for received boundary entities.
                    if let MeshEntity::Vertex(id) = entity {
                        self.vertex_coordinates.insert(id, coords);
                    }
                    entities.insert(entity);
                }
            }
        }
    }

    fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default();

            // Collect data for all boundary vertices.
            let entities = self.entities.read().unwrap();
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords);
                    }
                }
            }

            // Send the collected boundary data to the channel.
            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        }
    }

    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;

    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        // Set up vertex coordinates.
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);

        // Add boundary entities.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        // Set up a separate sender and receiver for testing.
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate sending the boundary data.
        mesh.send_boundary_data();

        // Create a second mesh instance to simulate the receiver.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());

        // Simulate receiving the boundary data.
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the updated vertex coordinates.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /* #[test]
    fn test_receive_empty_data() {
        let mut mesh = Mesh::new();
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate receiving without sending any data.
        mesh.receive_boundary_data();

        // Ensure no data has been added.
        assert!(mesh.vertex_coordinates.is_empty());
    } */

    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        // Simulate sending the boundary data without setting a receiver.
        mesh.send_boundary_data();

        // No receiver to process, but this should not panic or fail.
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}