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

impl DomainBuilder {
    /// Creates a new `DomainBuilder` with an empty mesh.
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(),
        }
    }

    /// Adds a vertex to the domain with a specified ID and coordinates.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn add_vertex(&mut self, id: usize, coords: [f64; 3]) -> &mut Self {
        self.mesh.set_vertex_coordinates(id, coords);
        self.mesh
            .entities
            .write()
            .unwrap()
            .insert(MeshEntity::Vertex(id));
        self
    }

    /// Adds an edge connecting two vertices.
    ///
    /// # Arguments
    ///
    /// * `vertex1` - The ID of the first vertex.
    /// * `vertex2` - The ID of the second vertex.
    pub fn add_edge(&mut self, vertex1: usize, vertex2: usize) -> &mut Self {
        let edge_id = self.mesh.entities.read().unwrap().len() + 1; // Ensure unique IDs
        let edge = MeshEntity::Edge(edge_id);

        // Add relationships in the sieve
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex1), edge);
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex2), edge);
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex1));
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex2));

        // Add the edge to the entities set
        self.mesh.entities.write().unwrap().insert(edge);
        self
    }

    /// Adds a cell with the given vertices and automatically creates faces.
    ///
    /// # Arguments
    ///
    /// * `vertex_ids` - A vector of vertex IDs that define the cell.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        // Compute a unique ID for the new cell
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        // Generate faces (edges in 2D) for the cell
        let num_faces = self.mesh.count_entities(&MeshEntity::Face(0));
        let mut face_id_counter = num_faces + 1;

        let num_vertices = vertex_ids.len();
        for i in 0..num_vertices {
            let v1 = vertex_ids[i];
            let v2 = vertex_ids[(i + 1) % num_vertices];
            let face = MeshEntity::Face(face_id_counter);
            face_id_counter += 1;

            // Add relationships between the face and its vertices
            let vertex1 = MeshEntity::Vertex(v1);
            let vertex2 = MeshEntity::Vertex(v2);
            self.mesh.add_arrow(face, vertex1);
            self.mesh.add_arrow(face, vertex2);
            self.mesh.entities.write().unwrap().insert(face);

            // Add relationship between cell and face
            self.mesh.add_arrow(cell, face);
        }

        // Ensure vertices are in the mesh
        for &vertex_id in &vertex_ids {
            let vertex = MeshEntity::Vertex(vertex_id);
            self.mesh.entities.write().unwrap().insert(vertex);
        }

        // Add the cell entity to the entities set
        self.mesh.entities.write().unwrap().insert(cell);

        // The cone of the cell now includes faces
        let cone = self.mesh.sieve.cone(&cell).unwrap_or_default();
        assert_eq!(
            cone.len(),
            num_vertices, // The cone should have 'num_vertices' faces
            "Topology issue: Cone does not contain all faces"
        );

        self
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
    pub fn validate_geometry(&self) {
        assert!(
            GeometryValidation::test_vertex_coordinates(&self.mesh).is_ok(),
            "Geometry validation failed: Duplicate or invalid vertex coordinates."
        );
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
        builder.add_vertex(1, [0.0, 0.0, 0.0]);

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
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_edge(1, 2);

        let entities = builder.mesh.entities.read().unwrap();
        assert!(entities.iter().any(|e| matches!(e, MeshEntity::Edge(_))));

        let vertex_edges = builder
            .mesh
            .sieve
            .cone(&MeshEntity::Vertex(1))
            .unwrap_or_default();
        assert!(!vertex_edges.is_empty());
    }

    #[test]
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
    }

    #[test]
    fn test_build_mesh() {
        let mut builder = DomainBuilder::new();
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_edge(1, 2);

        let mesh = builder.build();
        let entities = mesh.entities.read().unwrap();
        assert!(entities.contains(&MeshEntity::Vertex(1)));
        assert!(entities.contains(&MeshEntity::Vertex(2)));
        assert!(entities.iter().any(|e| matches!(e, MeshEntity::Edge(_))));
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

        // Add vertices
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [1.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 1.0, 0.0]);

        // Add cells
        builder.add_cell(vec![1, 2, 3, 4]);

        // Before reordering
        let entities_before: Vec<_> = builder.mesh.entities.read().unwrap().iter().cloned().collect();

        // Apply reordering
        builder.apply_reordering();

        // After reordering
        let entities_after: Vec<_> = builder.mesh.entities.read().unwrap().iter().cloned().collect();

        // Ensure that the entities have been reordered by comparing their IDs
        let ids_before: Vec<usize> = entities_before.iter().map(|e| e.get_id()).collect();
        let ids_after: Vec<usize> = entities_after.iter().map(|e| e.get_id()).collect();

        assert_ne!(ids_before, ids_after, "Entity IDs should have changed after reordering");
    }

    #[test]
    fn test_validate_geometry() {
        let mut builder = DomainBuilder::new();

        // Add vertices with unique coordinates
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0]);

        // Validate geometry
        builder.validate_geometry();
    }

    #[test]
    #[should_panic(expected = "Geometry validation failed")]
    fn test_validate_geometry_failure() {
        let mut builder = DomainBuilder::new();

        // Add vertices with duplicate coordinates
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [0.0, 0.0, 0.0]); // Duplicate coordinates

        // Validate geometry
        builder.validate_geometry();
    }
}
