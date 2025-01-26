use super::{Mesh, MeshError};
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
    ///
    /// # Returns
    /// - `Ok(())` if the entity was added successfully.
    /// - `Err(MeshError)` if the entity already exists.
    pub fn add_entity(&self, entity: MeshEntity) -> Result<(), MeshError> {
        let mut entities = self.entities.write().unwrap();

        // Check if the entity already exists
        if entities.contains(&entity) {
            self.logger.log_warn(&format!(
                "Attempted to add duplicate entity: {:?}",
                entity
            ));
            return Err(MeshError::EntityExists(format!("{:?}", entity)));
        }

        // Insert the entity and log success
        entities.insert(entity);
        self.logger.log_info(&format!("Entity {:?} added successfully.", entity));
        Ok(())
    }

    /// Establishes a directed relationship (arrow) between two mesh entities.
    ///
    /// This relationship is added to the sieve structure, representing a connection
    /// from the `from` entity to the `to` entity. Relationships are useful for
    /// defining adjacency and connectivity in the mesh.
    ///
    /// # Returns
    /// - `Ok(())` if the relationship was added successfully.
    /// - `Err(MeshError)` if either entity does not exist in the mesh.
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) -> Result<(), MeshError> {
        let entities = self.entities.read().unwrap();

        // Validate that both entities exist in the mesh
        if !entities.contains(&from) {
            let error = MeshError::EntityNotFound(format!("From entity: {:?}", from));
            self.logger.log_error(&error);
            return Err(error);
        }

        if !entities.contains(&to) {
            let error = MeshError::EntityNotFound(format!("To entity: {:?}", to));
            self.logger.log_error(&error);
            return Err(error);
        }

        // Add the relationship and log success
        self.sieve.add_arrow(from, to);
        self.logger.log_info(&format!(
            "Successfully added relationship from {:?} to {:?}.",
            from, to
        ));
        Ok(())
    }

    /// Adds a directed arrow between two mesh entities.
    ///
    /// This is a direct delegate to the `Sieve`'s `add_arrow` method, simplifying
    /// the addition of directed relationships in the mesh's connectivity structure.
    ///
    /// # Returns
    /// - `Ok(())` if the arrow was added successfully.
    /// - `Err(MeshError)` if either entity does not exist in the mesh.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) -> Result<(), MeshError> {
        let entities = self.entities.read().unwrap();

        // Validate the existence of the entities
        if !entities.contains(&from) {
            let error = MeshError::EntityNotFound(format!("From entity: {:?}", from));
            self.logger.log_error(&error);
            return Err(error);
        }

        if !entities.contains(&to) {
            let error = MeshError::EntityNotFound(format!("To entity: {:?}", to));
            self.logger.log_error(&error);
            return Err(error);
        }

        // Add the arrow and log success
        self.sieve.add_arrow(from, to);
        self.logger.log_info(&format!(
            "Arrow successfully added from {:?} to {:?}.",
            from, to
        ));
        Ok(())
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
    ///
    /// # Returns
    /// - `Ok(())` if the vertex coordinates were set successfully.
    /// - `Err(MeshError)` if adding the vertex to the mesh fails.
    pub fn set_vertex_coordinates(
        &mut self,
        vertex_id: usize,
        coords: [f64; 3],
    ) -> Result<(), MeshError> {
        self.vertex_coordinates.insert(vertex_id, coords);

        let vertex = MeshEntity::Vertex(vertex_id);
        match self.add_entity(vertex) {
            Ok(_) => {
                self.logger.log_info(&format!(
                    "Vertex {:?} coordinates set to {:?}.",
                    vertex_id, coords
                ));
                Ok(())
            }
            Err(err) => {
                self.logger.log_error(&err);
                Err(err)
            }
        }
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.
    ///
    /// If the vertex does not exist in the `vertex_coordinates` map, this method
    /// logs a warning and returns `None`.
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        match self.vertex_coordinates.get(&vertex_id).cloned() {
            Some(coords) => {
                self.logger.log_info(&format!(
                    "Retrieved coordinates for vertex {:?}: {:?}.",
                    vertex_id, coords
                ));
                Some(coords)
            }
            None => {
                self.logger.log_warn(&format!(
                    "Attempted to retrieve coordinates for non-existent vertex {:?}.",
                    vertex_id
                ));
                None
            }
        }
    }

    /// Counts the number of entities of a specific type in the mesh.
    ///
    /// This method iterates through all entities in the mesh and counts those
    /// matching the type specified in `entity_type`.
    ///
    /// # Returns
    /// - The count of entities of the specified type.
    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        let count = entities
            .iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count();

        self.logger.log_info(&format!(
            "Counted {} entities of type {:?} in the mesh.",
            count, entity_type
        ));
        count
    }

    /// Applies a user-defined function to each entity in the mesh in parallel.
    ///
    /// The function is executed concurrently using Rayon’s parallel iterator,
    /// ensuring efficient processing of large meshes.
    ///
    /// Logs the total number of entities processed.
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();

        // Log the number of entities before processing
        let total_entities = entities.len();
        self.logger.log_info(&format!(
            "Starting parallel processing on {} entities.",
            total_entities
        ));

        // Execute the user-defined function in parallel
        entities.par_iter().for_each(func);

        // Log completion of the operation
        self.logger.log_info(&format!(
            "Completed parallel processing on {} entities.",
            total_entities
        ));
    }

    /// Retrieves all `Cell` entities from the mesh.
    ///
    /// This method filters the mesh's `entities` set and collects all elements
    /// classified as cells, returning them as a vector.
    ///
    /// Logs the total number of `Cell` entities retrieved.
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();

        // Filter and collect all `Cell` entities
        let cells: Vec<MeshEntity> = entities
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect();

        // Log the count of retrieved `Cell` entities
        self.logger.log_info(&format!(
            "Retrieved {} cell entities from the mesh.",
            cells.len()
        ));

        cells
    }

    /// Retrieves all `Face` entities from the mesh.
    ///
    /// Similar to `get_cells`, this method filters the mesh's `entities` set and
    /// collects all elements classified as faces.
    ///
    /// Logs the total number of `Face` entities retrieved.
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();

        // Filter and collect all `Face` entities
        let faces: Vec<MeshEntity> = entities
            .iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect();

        // Log the count of retrieved `Face` entities
        self.logger.log_info(&format!(
            "Retrieved {} face entities from the mesh.",
            faces.len()
        ));

        faces
    }

    /// Retrieves all `Vertex` entities from the mesh.
    ///
    /// Similar to `get_cells` and `get_faces`, this method filters
    /// the mesh's `entities` set and collects all elements classified as vertices.
    ///
    /// Logs the total number of `Vertex` entities retrieved.
    pub fn get_vertices(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();

        // Filter and collect all `Vertex` entities
        let vertices: Vec<MeshEntity> = entities
            .iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .cloned()
            .collect();

        // Log the count of retrieved `Vertex` entities
        self.logger.log_info(&format!(
            "Retrieved {} vertex entities from the mesh.",
            vertices.len()
        ));

        vertices
    }

    /// Retrieves the vertices of a given face entity.
    ///
    /// This method queries the sieve structure to find all vertices directly
    /// connected to the specified face entity.
    ///
    /// Logs the number of vertices retrieved or an error if the operation fails.
    pub fn get_vertices_of_face(&self, face: &MeshEntity) -> Result<Vec<MeshEntity>, MeshError> {
        // Ensure the input entity is a `Face`
        if !matches!(face, MeshEntity::Face(_)) {
            let error = MeshError::InvalidEntityType(format!(
                "Entity {:?} is not a valid Face.",
                face
            ));
            self.logger.log_error(&error);
            return Err(error);
        }

        // Retrieve connected vertices using the sieve structure
        match self.sieve.cone(face) {
            Ok(connected_entities) => {
                let vertices: Vec<MeshEntity> = connected_entities
                    .into_iter()
                    .filter(|e| matches!(e, MeshEntity::Vertex(_)))
                    .collect();

                // Log the result
                self.logger.log_info(&format!(
                    "Retrieved {} vertices for face {:?}.",
                    vertices.len(),
                    face
                ));
                Ok(vertices)
            }
            Err(err) => {
                let error = MeshError::ConnectivityError(
                    format!("Face: {:?}", face),
                    format!("Error: {}", err),
                );
                self.logger.log_error(&error);
                Err(error)
            }
        }
    }

    /// Computes properties for each entity in the mesh in parallel.
    ///
    /// The user provides a function `compute_fn` that maps a `MeshEntity` to a
    /// property of type `PropertyType`. This function is applied to all entities
    /// in the mesh concurrently, and the results are returned in a map.
    ///
    /// Logs the total number of entities processed.
    pub fn compute_properties<F, PropertyType>(
        &self,
        compute_fn: F,
    ) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        let total_entities = entities.len();

        // Log the start of the operation
        self.logger.log_info(&format!(
            "Starting property computation for {} entities.",
            total_entities
        ));

        // Perform the property computation in parallel
        let result: FxHashMap<MeshEntity, PropertyType> = entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect();

        // Log the completion of the operation
        self.logger.log_info(&format!(
            "Completed property computation for {} entities.",
            total_entities
        ));

        result
    }

    /// Retrieves the ordered neighboring cells for a given cell.
    ///
    /// This method is useful for numerical methods that require consistent ordering
    /// of neighbors, such as flux calculations or gradient reconstruction. Neighbors
    /// are sorted by their unique identifiers to ensure deterministic results.
    ///
    /// # Returns
    /// - `Ok(Vec<MeshEntity>)` if neighbors are successfully retrieved and sorted.
    /// - `Err(MeshError)` if there is an error during the retrieval process.
    pub fn get_ordered_neighbors(&self, cell: &MeshEntity) -> Result<Vec<MeshEntity>, MeshError> {
        let mut neighbors = Vec::new();

        // Validate that the input is a Cell entity
        if !matches!(cell, MeshEntity::Cell(_)) {
            let error = MeshError::InvalidEntityType(format!(
                "Entity {:?} is not a valid Cell.",
                cell
            ));
            self.logger.log_error(&error);
            return Err(error);
        }

        // Retrieve faces of the cell
        let faces = self
            .get_faces_of_cell(cell)
            .map_err(|err| {
                let error = MeshError::ConnectivityError(
                    format!("Cell: {:?}", cell),
                    format!("Error retrieving faces: {}", err),
                );
                self.logger.log_error(&error);
                error
            })?;

        // Iterate over each face to find neighboring cells
        for face in faces.iter() {
            let cells_sharing_face = self
                .get_cells_sharing_face(&*face.key())
                .map_err(|err| {
                    let error = MeshError::ConnectivityError(
                        format!("Face: {:?}", face.key()),
                        format!("Error retrieving cells sharing face: {}", err),
                    );
                    self.logger.log_error(&error);
                    error
                })?;

            for neighbor in cells_sharing_face.iter() {
                if neighbor.key() != cell {
                    neighbors.push(neighbor.key().clone());
                }
            }
        }

        // Sort neighbors by their unique IDs for deterministic ordering
        neighbors.sort_by(|a, b| a.get_id().cmp(&b.get_id()));

        // Log the result
        self.logger.log_info(&format!(
            "Retrieved {} ordered neighbors for cell {:?}.",
            neighbors.len(),
            cell
        ));
        Ok(neighbors)
    }


    /// Maps each `MeshEntity` in the mesh to a unique index.
    ///
    /// This method creates a mapping from each entity to a unique index, which can
    /// be useful for tasks like matrix assembly or entity-based data storage.
    ///
    /// Logs the total number of entities processed and ensures unique indexing.
    pub fn get_entity_to_index(&self) -> DashMap<MeshEntity, usize> {
        let entity_to_index = DashMap::new();
        let mut current_index = 0;

        // Collect all entities into a vector for ordered iteration
        let entities: Vec<MeshEntity> = {
            let entities = self.entities.read().unwrap();
            entities.iter().cloned().collect()
        };

        // Log the number of entities being indexed
        self.logger.log_info(&format!(
            "Starting entity-to-index mapping for {} entities.",
            entities.len()
        ));

        // Assign unique indices to all entities
        for entity in entities {
            entity_to_index.insert(entity, current_index);
            current_index += 1;
        }

        // Ensure all entities are uniquely indexed
        assert_eq!(
            entity_to_index.len(),
            current_index,
            "Entity-to-index mapping is incomplete or has duplicates."
        );

        // Log the completion of the operation
        self.logger.log_info(&format!(
            "Completed entity-to-index mapping for {} entities.",
            entity_to_index.len()
        ));

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
        let mut current_index = 0;

        // Retrieve cells from the mesh
        let cells = self.get_cells();

        // Log the number of cells being indexed
        self.logger.log_info(&format!(
            "Starting entity-to-index mapping for {} cells.",
            cells.len()
        ));

        // Assign indices for cells
        for cell in cells {
            index_map.insert(cell.clone(), current_index);
            current_index += 1;
        }

        // Ensure consistency in index assignment
        assert_eq!(
            index_map.len(),
            current_index,
            "Entity-to-index mapping has gaps or inconsistencies."
        );

        // Log the completion of the mapping process
        self.logger.log_info(&format!(
            "Completed entity-to-index mapping for {} entities.",
            index_map.len()
        ));

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

    /// Checks if a given `MeshEntity` exists in the mesh.
    ///
    /// This function validates whether the provided entity is part of the mesh's domain.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` to check.
    ///
    /// # Returns
    /// - `bool`: Returns `true` if the entity exists in the mesh, otherwise `false`.
    pub fn entity_exists(&self, entity: &MeshEntity) -> bool {
        let entities = self.entities.read().unwrap();
        entities.contains(entity)
    }
}
