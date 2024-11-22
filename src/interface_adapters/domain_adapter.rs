use rustc_hash::FxHashMap;

use crate::domain::{mesh::Mesh, MeshEntity};
use crate::domain::mesh::geometry_validation::GeometryValidation;
use crate::domain::mesh::reordering::cuthill_mckee;

pub struct DomainBuilder {
    mesh: Mesh,
}

impl DomainBuilder {
    /// Create a new `DomainBuilder`.
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(),
        }
    }

    /// Add a vertex to the domain.
    pub fn add_vertex(&mut self, id: usize, coords: [f64; 3]) -> &mut Self {
        self.mesh.set_vertex_coordinates(id, coords);
        self.mesh.entities.write().unwrap().insert(MeshEntity::Vertex(id));
        self
    }

    /// Add an edge connecting two vertices.
    pub fn add_edge(&mut self, vertex1: usize, vertex2: usize) -> &mut Self {
        let edge_id = self.mesh.entities.read().unwrap().len() + 1; // Ensure unique IDs
        let edge = MeshEntity::Edge(edge_id);

        // Add relationships in the sieve
        self.mesh.add_arrow(MeshEntity::Vertex(vertex1), edge);
        self.mesh.add_arrow(MeshEntity::Vertex(vertex2), edge);
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex1));
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex2));

        // Add the edge to the entities set
        self.mesh.entities.write().unwrap().insert(edge);
        self
    }

    /// Add a cell with given vertices.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        // Compute a unique ID for the new cell
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);
    
        // Add relationships between the cell and its vertices
        for &vertex_id in &vertex_ids {
            let vertex = MeshEntity::Vertex(vertex_id);
            self.mesh.add_arrow(cell, vertex); // Cell points to Vertex
            self.mesh.entities.write().unwrap().insert(vertex); // Ensure vertices are in the mesh
        }
    
        // Add the cell entity to the entities set
        self.mesh.entities.write().unwrap().insert(cell);
    
        // Debugging: Verify the cone of the new cell
        let cone = self.mesh.sieve.cone(&cell).unwrap_or_default();
        assert_eq!(
            cone.len(),
            vertex_ids.len(),
            "Topology issue: Cone does not contain all vertices"
        );
    
/*         // Perform topology validation
        let topology_validation = TopologyValidation::new(&self.mesh);
        assert!(
            topology_validation.validate_connectivity(),
            "Topology validation failed: Cell is not properly connected to vertices."
        ); */
    
        self
    }

    /// Apply reordering to improve solver performance.
    pub fn apply_reordering(&mut self) {
        let entities: Vec<_> = self.mesh.entities.read().unwrap().iter().cloned().collect();
        let adjacency: FxHashMap<_, _> = self.mesh.sieve.to_adjacency_map();
        let reordered = cuthill_mckee(&entities, &adjacency);

        // Apply the reordering
        self.mesh.apply_reordering(&reordered.iter().map(|e| e.get_id()).collect::<Vec<_>>());
    }

    /// Perform geometry validation to ensure mesh integrity.
    pub fn validate_geometry(&self) {
        assert!(
            GeometryValidation::test_vertex_coordinates(&self.mesh).is_ok(),
            "Geometry validation failed: Duplicate or invalid vertex coordinates."
        );
    }

    /// Finalize and return the built `Mesh`.
    pub fn build(self) -> Mesh {
        self.mesh
    }
}

pub struct DomainEntity {
    pub entity: MeshEntity,
    pub boundary_conditions: Option<String>,
    pub material_properties: Option<String>,
}

impl DomainEntity {
    /// Create a new DomainEntity.
    pub fn new(entity: MeshEntity) -> Self {
        Self {
            entity,
            boundary_conditions: None,
            material_properties: None,
        }
    }

    /// Set boundary conditions for the entity.
    pub fn set_boundary_conditions(mut self, bc: &str) -> Self {
        self.boundary_conditions = Some(bc.to_string());
        self
    }

    /// Set material properties for the entity.
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

        assert_eq!(builder.mesh.get_vertex_coordinates(1), Some([0.0, 0.0, 0.0]));
        assert!(builder.mesh.entities.read().unwrap().contains(&MeshEntity::Vertex(1)));
    }

    #[test]
    fn test_add_edge() {
        let mut builder = DomainBuilder::new();
        builder.add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_edge(1, 2);

        let entities = builder.mesh.entities.read().unwrap();
        assert!(entities.iter().any(|e| matches!(e, MeshEntity::Edge(_))));

        let vertex_edges = builder.mesh.sieve.cone(&MeshEntity::Vertex(1)).unwrap_or_default();
        assert!(!vertex_edges.is_empty());
    }

    #[test]
    fn test_add_cell() {
        let mut builder = DomainBuilder::new();

        // Add vertices with explicit coordinates
        builder.add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0]);

        // Log the state of the mesh before adding the cell
        println!("Mesh state before adding cell: {:?}", builder.mesh.entities.read().unwrap());

        // Add a cell connecting the three vertices
        builder.add_cell(vec![1, 2, 3]);

        // Log the state of the mesh after adding the cell
        println!("Mesh state after adding cell: {:?}", builder.mesh.entities.read().unwrap());

        // Verify that the cell exists in the entities set
        let entities = builder.mesh.entities.read().unwrap();
        assert!(
            entities.iter().any(|e| matches!(e, MeshEntity::Cell(_))),
            "Cell entity not found in mesh entities set."
        );

        // Fetch the specific cell and verify its cone contains the correct vertices
        let cell_id = entities.len(); // Assuming cell ID corresponds to the last added entity
        let cell = MeshEntity::Cell(cell_id);

        let cone = builder.mesh.sieve.cone(&cell).unwrap_or_default();
        println!("Cone of the cell: {:?}", cone);
        assert_eq!(cone.len(), 3, "The cone should contain all 3 vertices.");

        // Sort the vertex IDs in the cone before comparison
        let mut vertex_ids: Vec<_> = cone
            .iter()
            .filter_map(|e| match e {
                MeshEntity::Vertex(id) => Some(*id),
                _ => None,
            })
            .collect();
        vertex_ids.sort_unstable();

        println!("Sorted vertex IDs in the cone: {:?}", vertex_ids);
        assert_eq!(vertex_ids, vec![1, 2, 3], "The cone contains the wrong vertices.");
    }




    #[test]
    fn test_build_mesh() {
        let mut builder = DomainBuilder::new();
        builder.add_vertex(1, [0.0, 0.0, 0.0])
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
}
