use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

impl Mesh {
    /// Adds a new `MeshEntity` to the mesh.
    ///
    /// This method inserts the entity into the mesh's thread-safe `entities` set,
    /// ensuring it becomes part of the mesh's domain. The `entities` set tracks all
    /// vertices, edges, faces, and cells in the mesh.
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    /// Establishes a directed relationship (arrow) between two mesh entities.
    ///
    /// This relationship is added to the sieve structure, representing a connection
    /// from the `from` entity to the `to` entity. Relationships are useful for
    /// defining adjacency and connectivity in the mesh.
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Adds a directed arrow between two mesh entities.
    ///
    /// This is a direct delegate to the `Sieve`'s `add_arrow` method, simplifying
    /// the addition of directed relationships in the mesh's connectivity structure.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Sets the 3D coordinates of a vertex and adds the vertex to the mesh.
    ///
    /// This method ensures the vertex is registered in the `vertex_coordinates` map
    /// with its corresponding coordinates, and also adds the vertex to the mesh's
    /// `entities` set if not already present.
    ///
    /// # Arguments
    /// - `vertex_id`: Unique identifier of the vertex.
    /// - `coords`: 3D coordinates of the vertex.
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.
    ///
    /// If the vertex does not exist in the `vertex_coordinates` map, this method
    /// returns `None`.
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    /// Counts the number of entities of a specific type in the mesh.
    ///
    /// This method iterates through all entities in the mesh and counts those
    /// matching the type specified in `entity_type`.
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

    /// Applies a user-defined function to each entity in the mesh in parallel.
    ///
    /// The function is executed concurrently using Rayon’s parallel iterator,
    /// ensuring efficient processing of large meshes.
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Retrieves all `Cell` entities from the mesh.
    ///
    /// This method filters the mesh's `entities` set and collects all elements
    /// classified as cells, returning them as a vector.
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all `Face` entities from the mesh.
    ///
    /// Similar to `get_cells`, this method filters the mesh's `entities` set and
    /// collects all elements classified as faces.
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all `Vertex` entities from the mesh.
    /// 
    /// Similar to `get_cells` and `get_faces`, this method filters
    /// the mesh's `entities` set and collects all elements classified as vertices.
    pub fn get_vetices(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .cloned()
            .collect()
    }

    /// Retrieves the vertices of a given face entity.
    ///
    /// This method queries the sieve structure to find all vertices directly
    /// connected to the specified face entity.
    pub fn get_vertices_of_face(&self, face: &MeshEntity) -> Vec<MeshEntity> {
        self.sieve.cone(face).unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .collect()
    }

    /// Computes properties for each entity in the mesh in parallel.
    ///
    /// The user provides a function `compute_fn` that maps a `MeshEntity` to a
    /// property of type `PropertyType`. This function is applied to all entities
    /// in the mesh concurrently, and the results are returned in a map.
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

    /// Retrieves the ordered neighboring cells for a given cell.
    ///
    /// This method is useful for numerical methods that require consistent ordering
    /// of neighbors, such as flux calculations or gradient reconstruction. Neighbors
    /// are sorted by their unique identifiers to ensure deterministic results.
    ///
    /// # Returns
    /// A vector of neighboring cells sorted by ID.
    pub fn get_ordered_neighbors(&self, cell: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = Vec::new();
        if let Some(faces) = self.get_faces_of_cell(cell) {
            for face in faces.iter() {
                let cells_sharing_face = self.get_cells_sharing_face(&face.key());
                for neighbor in cells_sharing_face.iter() {
                    if *neighbor.key() != *cell {
                        neighbors.push(*neighbor.key());
                    }
                }
            }
        }
        neighbors.sort_by(|a, b| a.get_id().cmp(&b.get_id())); // Ensures consistent ordering by ID
        neighbors
    }

    /// Maps each `MeshEntity` in the mesh to a unique index.
    ///
    /// This method creates a mapping from each entity to a unique index, which can
    /// be useful for tasks like matrix assembly or entity-based data storage.
    pub fn get_entity_to_index(&self) -> DashMap<MeshEntity, usize> {
        let entity_to_index = DashMap::new();
        let entities = self.entities.read().unwrap();
        entities.iter().enumerate().for_each(|(index, entity)| {
            entity_to_index.insert(entity.clone(), index);
        });

        entity_to_index
    }

    /// Generates a mapping from `MeshEntity` to matrix or RHS indices.
    ///
    /// # Returns
    /// - `DashMap<MeshEntity, usize>`: A mapping where the key is a `MeshEntity`
    ///   and the value is its corresponding index in the system matrix or RHS vector.
    ///
    /// # Notes
    /// - This function assumes a consistent ordering of mesh entities (e.g., cells, faces, vertices)
    ///   as determined by the mesh generation process.
    /// - The indices are typically used for assembling sparse system matrices or RHS vectors.
    pub fn entity_to_index_map(&self) -> DashMap<MeshEntity, usize> {
        let index_map = DashMap::new();

        // Iterate over all entities in the mesh and assign them indices.
        // This example assumes cells, faces, and vertices are stored in separate collections.
        
        // Assign indices for cells.
        for (i, cell) in self.get_cells().iter().enumerate() {
            index_map.insert(MeshEntity::Cell(cell.get_id()), i);
        }

        // Offset for face indices.
        let face_offset = self.get_cells().len();

        // Assign indices for faces.
        for (i, face) in self.get_faces().iter().enumerate() {
            index_map.insert(MeshEntity::Face(face.get_id()), face_offset + i);
        }

        index_map
    }

    /// Retrieves a `MeshEntity` from a given `usize` key.
    ///
    /// This function maps a `usize` index to a `MeshEntity` using the mesh's entity mapping.
    ///
    /// # Parameters
    /// - `key`: The `usize` index corresponding to the desired `MeshEntity`.
    ///
    /// # Returns
    /// - `Option<MeshEntity>`: The `MeshEntity` if found, or `None` if the index is invalid.
    pub fn get_mesh_entity_from_key(&self, key: usize) -> Option<MeshEntity> {
        // Use the index mapping to find the entity
        let index_map = self.entity_to_index_map();
        
        // Iterate through the mapping to find the matching `key`
        let x = index_map.iter()
            .find_map(|entry| {
                let (entity, &index) = entry.pair();
                if index == key {
                    Some(entity.clone()) // Return the matched entity
                } else {
                    None
                }
            }); x
    }
}
