
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
        self.mesh.set_vertex_coordinates(id, coords).unwrap();
        self.mesh
            .entities
            .write()
            .unwrap()
            .insert(MeshEntity::Vertex(id));
        self
    }

    /// Adds an edge connecting two vertices.
    pub fn add_edge(&mut self, vertex1: usize, vertex2: usize) -> &mut Self {
        let edge_id = self.mesh.entities.read().unwrap().len() + 1; // Ensure unique IDs
        let edge = MeshEntity::Edge(edge_id);

        // Add relationships in the sieve
        self.mesh.add_arrow(MeshEntity::Vertex(vertex1), edge).unwrap();
        self.mesh.add_arrow(MeshEntity::Vertex(vertex2), edge).unwrap();
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex1)).unwrap();
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex2)).unwrap();

        // Add the edge to the entities set
        self.mesh.entities.write().unwrap().insert(edge);
        self
    }

    /// Adds a polygonal cell with the given vertices and creates a single face that includes
    /// all cell vertices. This is suitable for 2D domains where each cell is a polygon.
    ///
    /// If you are working in 3D, you must adapt this method to create multiple faces, each
    /// with 3 or 4 vertices, corresponding to the cell's polygonal faces.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        let face_id = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;
        let face = MeshEntity::Face(face_id);

        // Connect the face to all vertices of the cell
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);
            self.mesh.add_arrow(face.clone(), vertex.clone()).unwrap();
            self.mesh.add_arrow(vertex.clone(), face.clone()).unwrap();
        }

        // Connect cell to the face and vice versa
        self.mesh.add_arrow(cell.clone(), face.clone()).unwrap();
        self.mesh.add_arrow(face.clone(), cell.clone()).unwrap();

        // Insert the face into entities
        self.mesh.entities.write().unwrap().insert(face);

        // Ensure all vertices are in entities
        for &vertex_id in &vertex_ids {
            self.mesh.entities.write().unwrap().insert(MeshEntity::Vertex(vertex_id));
        }
        self.mesh.entities.write().unwrap().insert(cell);

        self
    }

    /// Adds a tetrahedron cell for 3D cases. This method is correct as is, since it creates proper
    /// triangular faces for the tetrahedron.
    pub fn add_tetrahedron_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        assert_eq!(vertex_ids.len(), 4, "Tetrahedron must have 4 vertices");
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        let face_vertices = vec![
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2]], // Face 1
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[3]], // Face 2
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[3]], // Face 3
            vec![vertex_ids[2], vertex_ids[0], vertex_ids[3]], // Face 4
        ];

        for (i, fv) in face_vertices.iter().enumerate() {
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            // Add arrows from face to vertices
            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);
                self.mesh.add_arrow(face.clone(), vertex.clone()).unwrap();
                self.mesh.add_arrow(vertex.clone(), face.clone()).unwrap();
            }

            self.mesh.entities.write().unwrap().insert(face.clone());

            // Add arrows from cell to face
            self.mesh.add_arrow(cell.clone(), face.clone()).unwrap();
            self.mesh.add_arrow(face.clone(), cell.clone()).unwrap();
        }

        // Add arrows from cell to vertices
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);
            self.mesh.add_arrow(cell.clone(), vertex.clone()).unwrap();
            self.mesh.add_arrow(vertex.clone(), cell.clone()).unwrap();
        }

        self.mesh.entities.write().unwrap().insert(cell);

        self
    }

    /// Adds a hexahedron cell to the mesh. 
    ///
    /// # Arguments
    /// * `vertex_ids` - A vector of exactly 8 vertex IDs that define the hexahedron.
    ///   The vertices should be provided in a consistent order.
    ///
    /// # Panics
    /// Panics if `vertex_ids` does not contain exactly 8 vertices.
    pub fn add_hexahedron_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        assert_eq!(vertex_ids.len(), 8, "Hexahedron must have 8 vertices");
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        // Define the six quadrilateral faces of the hexahedron.
        let face_vertices = vec![
            // Bottom
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2], vertex_ids[3]],
            // Top
            vec![vertex_ids[4], vertex_ids[5], vertex_ids[6], vertex_ids[7]],
            // Front
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[5], vertex_ids[4]],
            // Right
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[6], vertex_ids[5]],
            // Back
            vec![vertex_ids[2], vertex_ids[3], vertex_ids[7], vertex_ids[6]],
            // Left
            vec![vertex_ids[3], vertex_ids[0], vertex_ids[4], vertex_ids[7]],
        ];

        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        // Create and link each face
        for (i, fv) in face_vertices.iter().enumerate() {
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            // Add arrows from face to its vertices and vice versa
            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);
                self.mesh.add_arrow(face.clone(), vertex.clone()).unwrap();
                self.mesh.add_arrow(vertex.clone(), face.clone()).unwrap();
            }

            self.mesh.entities.write().unwrap().insert(face.clone());

            // Add arrows between cell and face
            self.mesh.add_arrow(cell.clone(), face.clone()).unwrap();
            self.mesh.add_arrow(face, cell.clone()).unwrap();
        }

        // Link the cell with its vertices
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);
            self.mesh.add_arrow(cell.clone(), vertex.clone()).unwrap();
            self.mesh.add_arrow(vertex, cell.clone()).unwrap();
        }

        // Insert the cell into the mesh entities
        self.mesh.entities.write().unwrap().insert(cell);

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
/* 
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
    } */

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
            builder.add_vertex(id, coords);
        }

        // Now add a hexahedron cell with these 8 vertices
        builder.add_hexahedron_cell(vec![1, 2, 3, 4, 5, 6, 7, 8]);

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
