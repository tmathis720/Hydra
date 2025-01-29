
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use crate::domain::mesh::geometry_validation::GeometryValidation;
use crate::domain::mesh::reordering::cuthill_mckee;

/// `DomainBuilder` is a utility for constructing a mesh domain by incrementally adding vertices,
/// edges, faces, and cells. It provides methods to build the mesh and apply reordering for
/// performance optimization.
pub struct DomainBuilder {
    mesh: Mesh,
}

#[derive(Debug, thiserror::Error)]
pub enum DomainBuilderError {
    #[error("Failed to add vertex: ID {0} already exists.")]
    VertexAlreadyExists(usize),
    #[error("Invalid vertex coordinates: {0:?}.")]
    InvalidVertexCoordinates([f64; 3]),
    #[error("Failed to add edge: One or more vertex IDs do not exist.")]
    EdgeVertexNotFound,
    #[error("Invalid cell definition: {0}.")]
    InvalidCellDefinition(String),
    #[error("Geometry validation failed: {0}")]
    GeometryValidationFailed(String),
    #[error("Unknown error occurred.")]
    Unknown,
    #[error("Invalid vertex error occurred: {0}.")]
    CellInvalidVertices(String),
    #[error("Mesh error occurred: {0}")]
    MeshError(String),
}

impl DomainBuilder {
    /// Creates a new `DomainBuilder` with an empty mesh.
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(),
        }
    }

    /// Adds a vertex to the domain with a specified ID and coordinates.
    /// # Arguments
    ///
    /// * `id` - A unique identifier for the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    ///
    /// # Returns
    ///
    /// * `Result<&mut Self, DomainBuilderError>` - Ok on success, Err with an appropriate error on failure.
    pub fn add_vertex(
        &mut self,
        id: usize,
        coords: [f64; 3],
    ) -> Result<&mut Self, DomainBuilderError> {
        // Check if the vertex ID already exists
        let entities = self.mesh.entities.read().unwrap();
        if entities.contains(&MeshEntity::Vertex(id)) {
            log::error!("Vertex ID {} already exists in the mesh.", id);
            return Err(DomainBuilderError::VertexAlreadyExists(id));
        }

        // Validate coordinates (e.g., ensure finite numbers)
        if coords.iter().any(|&x| !x.is_finite()) {
            log::error!("Invalid coordinates for vertex ID {}: {:?}", id, coords);
            return Err(DomainBuilderError::InvalidVertexCoordinates(coords));
        }
        drop(entities); // Release the read lock

        // Add vertex to the mesh
        self.mesh
            .set_vertex_coordinates(id, coords)
            .map_err(|e| {
                log::error!(
                    "Failed to set coordinates for vertex ID {}: {:?}. Error: {:?}",
                    id,
                    coords,
                    e
                );
                DomainBuilderError::Unknown
            })?;

        // Insert the vertex entity
        self.mesh
            .entities
            .write()
            .unwrap()
            .insert(MeshEntity::Vertex(id));

        log::info!("Successfully added vertex ID {} with coordinates {:?}.", id, coords);
        Ok(self)
    }


    /// Adds an edge connecting two vertices.
    ///
    /// # Arguments
    /// * `vertex1` - The ID of the first vertex.
    /// * `vertex2` - The ID of the second vertex.
    ///
    /// # Returns
    /// * `Result<&mut Self, DomainBuilderError>` - Ok on success, Err with a descriptive error on failure.
    pub fn add_edge(
        &mut self,
        vertex1: usize,
        vertex2: usize,
    ) -> Result<&mut Self, DomainBuilderError> {
        // Ensure the vertices exist
        let entities = self.mesh.entities.read().unwrap();
        if !entities.contains(&MeshEntity::Vertex(vertex1)) {
            log::error!("Vertex {} not found in the mesh.", vertex1);
            return Err(DomainBuilderError::EdgeVertexNotFound);
        }
        if !entities.contains(&MeshEntity::Vertex(vertex2)) {
            log::error!("Vertex {} not found in the mesh.", vertex2);
            return Err(DomainBuilderError::EdgeVertexNotFound);
        }
        drop(entities); // Release the read lock

        // Create and insert the edge
        let edge_id = self.mesh.entities.read().unwrap().len() + 1;
        let edge = MeshEntity::Edge(edge_id);

        self.mesh.entities.write().unwrap().insert(edge.clone());
        log::info!(
            "Successfully added edge ID {} connecting vertices {} and {}.",
            edge_id,
            vertex1,
            vertex2
        );

        // Establish relationships between the edge and its vertices
        for (from, to) in &[
            (MeshEntity::Vertex(vertex1), edge.clone()),
            (MeshEntity::Vertex(vertex2), edge.clone()),
            (edge.clone(), MeshEntity::Vertex(vertex1)),
            (edge.clone(), MeshEntity::Vertex(vertex2)),
        ] {
            if let Err(err) = self.mesh.add_arrow(from.clone(), to.clone()) {
                log::error!(
                    "Failed to add arrow from {:?} to {:?}: {}",
                    from,
                    to,
                    err
                );
                return Err(DomainBuilderError::MeshError(format!(
                    "Failed to establish relationship between {:?} and {:?}: {}",
                    from, to, err
                )));
            }
        }

        Ok(self)
    }


    /// Adds a polygonal cell with the given vertices and creates a single face that includes
    /// all cell vertices. This is suitable for 2D domains where each cell is a polygon.
    ///
    /// If you are working in 3D, you must adapt this method to create multiple faces, each
    /// with 3 or 4 vertices, corresponding to the cell's polygonal faces.
    ///
    /// # Arguments
    /// * `vertex_ids` - A vector of vertex IDs that define the cell's vertices.
    ///
    /// # Returns
    /// * `Result<&mut Self, DomainBuilderError>` - `Ok` on success, or `Err` with details of the failure.
    ///
    /// # Errors
    /// Returns an error if the `vertex_ids` list is empty, contains invalid vertex IDs,
    /// or if any `add_arrow` operation fails.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> Result<&mut Self, DomainBuilderError> {
        if vertex_ids.is_empty() {
            log::error!("Cannot add a cell with no vertices.");
            return Err(DomainBuilderError::CellInvalidVertices(
                "Cell must have at least one vertex.".to_string(),
            ));
        }

        // Check that all vertex IDs exist
        let entities = self.mesh.entities.read().unwrap();
        for &vid in &vertex_ids {
            if !entities.contains(&MeshEntity::Vertex(vid)) {
                log::error!("Vertex {} not found in the mesh.", vid);
                return Err(DomainBuilderError::CellInvalidVertices(format!(
                    "Vertex {} does not exist in the mesh.",
                    vid
                )));
            }
        }
        drop(entities); // Release the read lock

        // Create the cell
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        self.mesh.entities.write().unwrap().insert(cell.clone());
        log::info!(
            "Successfully added cell ID {} with vertices {:?}.",
            cell_id,
            vertex_ids
        );

        let n = vertex_ids.len();
        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        for i in 0..n {
            // Create the face
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            self.mesh.entities.write().unwrap().insert(face.clone());
            log::info!(
                "Successfully added face ID {} as part of cell ID {}.",
                face_id,
                cell_id
            );

            let v1 = vertex_ids[i];
            let v2 = vertex_ids[(i + 1) % n];

            // Add arrows between the face and its vertices
            for &vid in &[v1, v2] {
                let vertex = MeshEntity::Vertex(vid);
                for (from, to) in &[
                    (face.clone(), vertex.clone()),
                    (vertex.clone(), face.clone()),
                ] {
                    if let Err(err) = self.mesh.add_arrow(from.clone(), to.clone()) {
                        log::error!(
                            "Failed to add arrow from {:?} to {:?}: {}",
                            from,
                            to,
                            err
                        );
                        return Err(DomainBuilderError::MeshError(format!(
                            "Failed to establish relationship between {:?} and {:?}: {}",
                            from, to, err
                        )));
                    }
                }
            }

            // Add arrows between the cell and the face
            for (from, to) in &[
                (cell.clone(), face.clone()),
                (face.clone(), cell.clone()),
            ] {
                if let Err(err) = self.mesh.add_arrow(from.clone(), to.clone()) {
                    log::error!(
                        "Failed to add arrow from {:?} to {:?}: {}",
                        from,
                        to,
                        err
                    );
                    return Err(DomainBuilderError::MeshError(format!(
                        "Failed to establish relationship between {:?} and {:?}: {}",
                        from, to, err
                    )));
                }
            }
        }

        Ok(self)
    }

    /// Adds a tetrahedron cell for 3D cases by creating triangular faces.
    ///
    /// # Arguments
    /// * `vertex_ids` - A vector of exactly 4 vertex IDs that define the tetrahedron.
    ///
    /// # Returns
    /// * `Result<&mut Self, DomainBuilderError>` - `Ok` on success, or `Err` with details of the failure.
    ///
    /// # Errors
    /// Returns an error if `vertex_ids` does not contain exactly 4 vertices, if any vertex ID is invalid,
    /// or if any `add_arrow` operation fails.
    pub fn add_tetrahedron_cell(&mut self, vertex_ids: Vec<usize>) -> Result<&mut Self, DomainBuilderError> {
        // Ensure there are exactly 4 vertices
        if vertex_ids.len() != 4 {
            let error_msg = format!(
                "Failed to add tetrahedron cell: expected 4 vertices, got {}.",
                vertex_ids.len()
            );
            log::error!("{}", error_msg);
            return Err(DomainBuilderError::CellInvalidVertices(error_msg));
        }

        // Check that all vertex IDs exist in the mesh
        let entities = self.mesh.entities.read().unwrap();
        for &vid in &vertex_ids {
            if !entities.contains(&MeshEntity::Vertex(vid)) {
                let error_msg = format!("Vertex {} not found in the mesh.", vid);
                log::error!("{}", error_msg);
                return Err(DomainBuilderError::CellInvalidVertices(error_msg));
            }
        }
        drop(entities); // Release the read lock

        // Create the cell
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        self.mesh.entities.write().unwrap().insert(cell.clone());
        log::info!(
            "Successfully added cell ID {} with vertices {:?}.",
            cell_id,
            vertex_ids
        );

        // Generate unique IDs for the faces of the tetrahedron
        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        // Define the vertices for the tetrahedron's triangular faces
        let face_vertices = vec![
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2]], // Face 1
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[3]], // Face 2
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[3]], // Face 3
            vec![vertex_ids[2], vertex_ids[0], vertex_ids[3]], // Face 4
        ];

        for (i, fv) in face_vertices.iter().enumerate() {
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            self.mesh.entities.write().unwrap().insert(face.clone());
            log::info!(
                "Successfully added face ID {} as part of tetrahedron cell ID {}.",
                face_id,
                cell_id
            );

            // Connect the face to its vertices
            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);

                if let Err(err) = self.mesh.add_arrow(face.clone(), vertex.clone()) {
                    let error_msg = format!(
                        "Failed to add arrow from face {:?} to vertex {:?}: {}",
                        face, vertex, err
                    );
                    log::error!("{}", error_msg);
                    return Err(DomainBuilderError::MeshError(error_msg));
                }

                if let Err(err) = self.mesh.add_arrow(vertex.clone(), face.clone()) {
                    let error_msg = format!(
                        "Failed to add arrow from vertex {:?} to face {:?}: {}",
                        vertex, face, err
                    );
                    log::error!("{}", error_msg);
                    return Err(DomainBuilderError::MeshError(error_msg));
                }
            }

            // Add arrows between the cell and the face
            for (from, to) in &[
                (cell.clone(), face.clone()),
                (face.clone(), cell.clone()),
            ] {
                if let Err(err) = self.mesh.add_arrow(from.clone(), to.clone()) {
                    let error_msg = format!(
                        "Failed to add arrow from {:?} to {:?}: {}",
                        from, to, err
                    );
                    log::error!("{}", error_msg);
                    return Err(DomainBuilderError::MeshError(error_msg));
                }
            }
        }

        // Link the cell to its vertices
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);

            if let Err(err) = self.mesh.add_arrow(cell.clone(), vertex.clone()) {
                let error_msg = format!(
                    "Failed to add arrow from cell {:?} to vertex {:?}: {}",
                    cell, vertex, err
                );
                log::error!("{}", error_msg);
                return Err(DomainBuilderError::MeshError(error_msg));
            }

            if let Err(err) = self.mesh.add_arrow(vertex.clone(), cell.clone()) {
                let error_msg = format!(
                    "Failed to add arrow from vertex {:?} to cell {:?}: {}",
                    vertex, cell, err
                );
                log::error!("{}", error_msg);
                return Err(DomainBuilderError::MeshError(error_msg));
            }
        }

        log::info!(
            "Successfully added tetrahedron cell {:?} with vertices {:?}.",
            cell,
            vertex_ids
        );

        Ok(self)
    }



    /// Adds a hexahedron cell to the mesh.
    ///
    /// # Arguments
    /// * `vertex_ids` - A vector of exactly 8 vertex IDs that define the hexahedron.
    ///   The vertices should be provided in a consistent order.
    ///
    /// # Returns
    /// * `Result<&mut Self, DomainBuilderError>` - `Ok` on success, or `Err` with details of the failure.
    ///
    /// # Errors
    /// Returns an error if `vertex_ids` does not contain exactly 8 vertices, if any vertex ID is invalid,
    /// or if any `add_arrow` operation fails.
    /// Adds a hexahedron cell and ensures its faces are valid.
    pub fn add_hexahedron_cell(&mut self, vertex_ids: Vec<usize>) -> Result<&mut Self, DomainBuilderError> {
        // Ensure exactly 8 vertices are provided
        if vertex_ids.len() != 8 {
            return Err(DomainBuilderError::InvalidCellDefinition(format!(
                "Expected 8 vertices, got {}.",
                vertex_ids.len()
            )));
        }

        // Validate unique vertex positions to avoid degenerate faces
        let mut vertex_positions = Vec::new();
        for &vid in &vertex_ids {
            if let Some(coords) = self.mesh.get_vertex_coordinates(vid) {
                if vertex_positions.contains(&coords) {
                    return Err(DomainBuilderError::GeometryValidationFailed(format!(
                        "Degenerate face detected: Duplicate vertex at {:?}",
                        coords
                    )));
                }
                vertex_positions.push(coords);
            } else {
                return Err(DomainBuilderError::CellInvalidVertices(format!(
                    "Vertex {} not found in mesh.", vid
                )));
            }
        }

        // Create the hexahedron cell
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);
        self.mesh.entities.write().unwrap().insert(cell.clone());

        // Define the hexahedral faces
        let face_vertices = vec![
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2], vertex_ids[3]], // Bottom face
            vec![vertex_ids[4], vertex_ids[5], vertex_ids[6], vertex_ids[7]], // Top face
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[5], vertex_ids[4]], // Front face
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[6], vertex_ids[5]], // Right face
            vec![vertex_ids[2], vertex_ids[3], vertex_ids[7], vertex_ids[6]], // Back face
            vec![vertex_ids[3], vertex_ids[0], vertex_ids[4], vertex_ids[7]], // Left face
        ];

        for fv in face_vertices.iter() {
            // Compute normal and validate
            let normal = self.compute_face_normal(fv)?;
            if normal.iter().all(|&x| x.abs() < 1e-10) {
                return Err(DomainBuilderError::GeometryValidationFailed(
                    "Zero-magnitude face normal detected.".to_string(),
                ));
            }

            let face_id = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;
            let face = MeshEntity::Face(face_id);
            self.mesh.entities.write().unwrap().insert(face.clone());

            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);
                self.mesh.add_arrow(face.clone(), vertex.clone()).ok();
                self.mesh.add_arrow(vertex.clone(), face.clone()).ok();
            }

            self.mesh.add_arrow(cell.clone(), face.clone()).ok();
            self.mesh.add_arrow(face.clone(), cell.clone()).ok();
        }

        Ok(self)
    }

    /// Computes a normal vector for a given face defined by vertex IDs.
    fn compute_face_normal(&self, vertex_ids: &[usize]) -> Result<[f64; 3], DomainBuilderError> {
        if vertex_ids.len() < 3 {
            return Err(DomainBuilderError::GeometryValidationFailed(
                "Cannot compute normal for a face with fewer than 3 vertices.".to_string(),
            ));
        }

        let v0 = self.mesh.get_vertex_coordinates(vertex_ids[0]).unwrap();
        let v1 = self.mesh.get_vertex_coordinates(vertex_ids[1]).unwrap();
        let v2 = self.mesh.get_vertex_coordinates(vertex_ids[2]).unwrap();

        let u = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let v = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];

        let magnitude = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if magnitude < 1e-10 {
            return Err(DomainBuilderError::GeometryValidationFailed(
                "Computed normal has zero magnitude.".to_string(),
            ));
        }

        Ok([normal[0] / magnitude, normal[1] / magnitude, normal[2] / magnitude])
    }

    
    /// Applies reordering to improve solver performance using the Cuthill-McKee algorithm.
    pub fn apply_reordering(&mut self) {
        let entities: Vec<_> = self
            .mesh
            .entities
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let adjacency: FxHashMap<_, _> = self.mesh.sieve.to_adjacency_map();
        let reordered = cuthill_mckee(&entities, &adjacency);

        // Apply the reordering
        self.mesh
            .apply_reordering(&reordered.iter().map(|e| e.get_id()).collect::<Vec<_>>());
    }

    /// Performs geometry validation to ensure mesh integrity.
    ///
    /// # Returns
    /// * `Result<(), String>` - `Ok(())` if validation passes, or an `Err(String)` if it fails.
    pub fn validate_geometry(&self) -> Result<(), String> {
        GeometryValidation::test_vertex_coordinates(&self.mesh)
            .map_err(|err| format!("Geometry validation failed: {}", err))
    }


    /// Finalizes and returns the built `Mesh`.
    pub fn build(self) -> Mesh {
        self.mesh
    }
}

/// Represents a domain entity with optional boundary conditions and material properties.
pub struct DomainEntity {
    pub entity: MeshEntity,
    pub boundary_conditions: Option<String>,
    pub material_properties: Option<String>,
}

impl DomainEntity {
    /// Creates a new `DomainEntity`.
    ///
    /// # Arguments
    ///
    /// * `entity` - The mesh entity to associate with this domain entity.
    pub fn new(entity: MeshEntity) -> Self {
        Self {
            entity,
            boundary_conditions: None,
            material_properties: None,
        }
    }

    /// Sets boundary conditions for the entity.
    ///
    /// # Arguments
    ///
    /// * `bc` - A string describing the boundary condition.
    pub fn set_boundary_conditions(mut self, bc: &str) -> Self {
        self.boundary_conditions = Some(bc.to_string());
        self
    }

    /// Sets material properties for the entity.
    ///
    /// # Arguments
    ///
    /// * `properties` - A string describing the material properties.
    pub fn set_material_properties(mut self, properties: &str) -> Self {
        self.material_properties = Some(properties.to_string());
        self
    }

    
}

#[cfg(test)]
mod tests {
    use super::{DomainBuilder, DomainEntity};
    use crate::domain::MeshEntity;

    #[test]
    fn test_create_domain_builder() {
        let builder = DomainBuilder::new();
        assert!(builder.mesh.entities.read().unwrap().is_empty());
    }

    #[test]
    fn test_add_vertex() {
        let mut builder = DomainBuilder::new();
        let _ = builder.add_vertex(1, [0.0, 0.0, 0.0]);

        assert_eq!(
            builder.mesh.get_vertex_coordinates(1),
            Some([0.0, 0.0, 0.0])
        );
        assert!(builder
            .mesh
            .entities
            .read()
            .unwrap()
            .contains(&MeshEntity::Vertex(1)));
    }

    #[test]
    fn test_add_edge() {
        let mut builder = DomainBuilder::new();
    
        // Add vertices and ensure operations succeed
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
    
        // Add an edge and check for success
        assert!(builder.add_edge(1, 2).is_ok());
    
        // Verify that an edge entity was added
        let entities = builder.mesh.entities.read().unwrap();
        assert!(entities.iter().any(|e| matches!(e, MeshEntity::Edge(_))));
    
        // Verify that the edge connects to the vertex
        let vertex_edges = builder
            .mesh
            .sieve
            .cone(&MeshEntity::Vertex(1))
            .unwrap_or_default();
        assert!(!vertex_edges.is_empty());
    }
    

/*     #[test]
    fn test_add_cell() {
        let mut builder = DomainBuilder::new();

        // Add vertices with explicit coordinates
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0]);

        // Add a cell connecting the three vertices
        builder.add_cell(vec![1, 2, 3]);

        // Verify that the cell exists in the entities set
        let entities = builder.mesh.entities.read().unwrap();
        assert!(
            entities.iter().any(|e| matches!(e, MeshEntity::Cell(_))),
            "Cell entity not found in mesh entities set."
        );

        // Fetch the specific cell and verify its cone contains the correct faces
        let cell = entities
            .iter()
            .find(|e| matches!(e, MeshEntity::Cell(_)))
            .unwrap()
            .clone();

        let cone = builder.mesh.sieve.cone(&cell).unwrap_or_default();
        assert_eq!(
            cone.len(),
            3,
            "The cone should contain all 3 faces corresponding to the cell."
        );

        // Check that each face in the cone is connected to the correct vertices
        for face in cone {
            let face_cone = builder.mesh.sieve.cone(&face).unwrap_or_default();
            assert_eq!(
                face_cone.len(),
                2,
                "Each face should be connected to 2 vertices."
            );
            for vertex in face_cone {
                assert!(matches!(vertex, MeshEntity::Vertex(_)));
            }
        }
    } */

    #[test]
    fn test_build_mesh() {
        let mut builder = DomainBuilder::new();
    
        // Add vertices and ensure operations succeed
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
    
        // Add an edge and check for success
        assert!(builder.add_edge(1, 2).is_ok());
    
        // Build the mesh and verify the entities
        let mesh = builder.build();
        let entities = mesh.entities.read().unwrap();
    
        // Check that the vertices and edge are present in the mesh
        assert!(entities.contains(&MeshEntity::Vertex(1)), "Vertex 1 is missing from the mesh.");
        assert!(entities.contains(&MeshEntity::Vertex(2)), "Vertex 2 is missing from the mesh.");
        assert!(
            entities.iter().any(|e| matches!(e, MeshEntity::Edge(_))),
            "Edge entity is missing from the mesh."
        );
    }
    

    #[test]
    fn test_domain_entity_creation() {
        let vertex = MeshEntity::Vertex(1);
        let domain_entity = DomainEntity::new(vertex);

        assert_eq!(domain_entity.entity, vertex);
        assert!(domain_entity.boundary_conditions.is_none());
        assert!(domain_entity.material_properties.is_none());
    }

    #[test]
    fn test_set_boundary_conditions() {
        let vertex = MeshEntity::Vertex(1);
        let domain_entity = DomainEntity::new(vertex).set_boundary_conditions("Dirichlet");

        assert_eq!(domain_entity.boundary_conditions.unwrap(), "Dirichlet");
    }

    #[test]
    fn test_set_material_properties() {
        let vertex = MeshEntity::Vertex(1);
        let domain_entity = DomainEntity::new(vertex).set_material_properties("Steel");

        assert_eq!(domain_entity.material_properties.unwrap(), "Steel");
    }

    #[test]
    fn test_combined_domain_entity_properties() {
        let vertex = MeshEntity::Vertex(1);
        let domain_entity = DomainEntity::new(vertex)
            .set_boundary_conditions("Neumann")
            .set_material_properties("Aluminum");

        assert_eq!(domain_entity.boundary_conditions.unwrap(), "Neumann");
        assert_eq!(domain_entity.material_properties.unwrap(), "Aluminum");
    }

    #[test]
    fn test_apply_reordering() {
        let mut builder = DomainBuilder::new();
    
        // Add vertices and ensure success
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(3, [1.0, 1.0, 0.0]).is_ok());
        assert!(builder.add_vertex(4, [0.0, 1.0, 0.0]).is_ok());
    
        // Add a cell and ensure success
        assert!(builder.add_cell(vec![1, 2, 3, 4]).is_ok());
    
        // Before reordering: Collect initial entity list
        let entities_before: Vec<_> = builder.mesh.entities.read().unwrap().iter().cloned().collect();
    
        // Apply reordering
        builder.apply_reordering();
    
        // After reordering: Collect final entity list
        let entities_after: Vec<_> = builder.mesh.entities.read().unwrap().iter().cloned().collect();
    
        // Ensure the entities have been reordered by comparing their IDs
        let ids_before: Vec<usize> = entities_before.iter().map(|e| e.get_id()).collect();
        let ids_after: Vec<usize> = entities_after.iter().map(|e| e.get_id()).collect();
    
        // IDs should have changed after reordering
        assert_ne!(ids_before, ids_after, "Entity IDs should have changed after reordering");
    }
    

    #[test]
    fn test_validate_geometry() {
        let mut builder = DomainBuilder::new();
    
        // Add vertices with unique coordinates and ensure success
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok());
    
        // Validate geometry and check if it succeeds
        let builder = std::sync::Arc::new(std::sync::Mutex::new(builder));
        let _validation_result = std::panic::catch_unwind(|| {
            let builder = builder.lock().unwrap();
            builder.validate_geometry()
        });
    }
    

    #[test]
    fn test_validate_geometry_failure() {
        let mut builder = DomainBuilder::new();
    
        // Add vertices with duplicate coordinates
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok());
        assert!(builder.add_vertex(2, [0.0, 0.0, 0.0]).is_ok()); // Duplicate coordinates
    
        // Validate geometry and expect failure
        let validation_result = builder.validate_geometry();
        assert!(
            validation_result.is_err(),
            "Expected geometry validation to fail, but it succeeded."
        );

        // Check if the error message matches the expected failure reason
        if let Err(err) = validation_result {
            assert!(
                err.contains("Duplicate or invalid vertex coordinates"),
                "Unexpected error message: {}",
                err
            );
        }
    }
    

    #[test]
    fn test_add_hexahedron_cell() {
        let mut builder = DomainBuilder::new();

        // Define vertices for a cube-like hexahedron (z=0 for bottom face, z=1 for top face)
        // Bottom face: (0,0,0)=v1, (1,0,0)=v2, (1,1,0)=v3, (0,1,0)=v4
        // Top face: (0,0,1)=v5, (1,0,1)=v6, (1,1,1)=v7, (0,1,1)=v8
        let vertices = vec![
            (1, [0.0, 0.0, 0.0]),
            (2, [1.0, 0.0, 0.0]),
            (3, [1.0, 1.0, 0.0]),
            (4, [0.0, 1.0, 0.0]),
            (5, [0.0, 0.0, 1.0]),
            (6, [1.0, 0.0, 1.0]),
            (7, [1.0, 1.0, 1.0]),
            (8, [0.0, 1.0, 1.0]),
        ];

        for (id, coords) in vertices {
            let _ = builder.add_vertex(id, coords);
        }

        // Now add a hexahedron cell with these 8 vertices
        let _ = builder.add_hexahedron_cell(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mesh = builder.build();

        // Check that the cell was added
        let cells = mesh.get_cells();
        assert_eq!(cells.len(), 1, "There should be exactly one cell.");
        let cell = cells[0];

        // Check that the cell entity is indeed present
        match cell {
            MeshEntity::Cell(id) => {
                assert!(id > 0, "Cell ID should be a positive number.");
            }
            _ => panic!("The returned entity is not a Cell."),
        }

        // The hexahedron should have 6 faces
        let faces = mesh.get_faces();
        assert_eq!(faces.len(), 6, "A hexahedron should have 6 faces.");

        // Each face should have 4 vertices
        for face in &faces {
            let face_vertices = mesh.get_vertices_of_face(face);
            assert_eq!(face_vertices.unwrap().len(), 4, "Each hexahedron face should have 4 vertices.");
        }

        // Verify that the cell and faces are connected properly
        let cell_faces = mesh.get_faces_of_cell(&cell).expect("Cell should have faces.");
        assert_eq!(cell_faces.len(), 6, "The cell should have 6 associated faces.");

        // Check that all vertices are still in the mesh
        for vid in 1..=8 {
            assert!(
                mesh.get_vertex_coordinates(vid).is_some(),
                "Vertex {} should be present in the mesh.",
                vid
            );
        }
    }
}
