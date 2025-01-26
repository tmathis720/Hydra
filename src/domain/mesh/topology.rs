use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use crate::domain::mesh::Mesh;
use rustc_hash::FxHashSet;
use std::sync::{Arc, RwLock};

/// The `TopologyValidation` struct is responsible for validating the connectivity
/// and uniqueness of mesh entities. It ensures:
/// - Cells are correctly connected to faces and vertices.
/// - Edges within a cell are unique.
///
/// This structure leverages the sieve data structure and mesh entities to perform
/// validation checks on the topology of the mesh.
pub struct TopologyValidation<'a> {
    sieve: &'a Sieve, // Reference to the sieve containing adjacency relationships.
    entities: &'a Arc<RwLock<FxHashSet<MeshEntity>>>, // Shared access to mesh entities.
}

impl<'a> TopologyValidation<'a> {
    /// Constructs a new `TopologyValidation` instance using a reference to a mesh.
    ///
    /// # Arguments
    /// * `mesh` - The `Mesh` instance whose topology needs to be validated.
    pub fn new(mesh: &'a Mesh) -> Self {
        TopologyValidation {
            sieve: &mesh.sieve,
            entities: &mesh.entities,
        }
    }

    /// Validates the connectivity of all `Cell` entities in the mesh.
    ///
    /// Ensures that:
    /// - Each cell is connected to valid `Face` entities.
    /// - Each face is connected to valid `Vertex` entities.
    ///
    /// # Returns
    /// * `true` if all cells are correctly connected, `false` otherwise.
    pub fn validate_connectivity(&self) -> bool {
        for cell in self.get_cells() {
            if !self.validate_cell_connectivity(&cell) {
                return false;
            }
        }
        true
    }

    /// Validates that edges within cells are unique.
    ///
    /// Checks for duplicate edges within the same cell and ensures that all edges
    /// adhere to the expected topology.
    ///
    /// # Returns
    /// * `true` if all edges are unique, `false` otherwise.
    pub fn validate_unique_relationships(&self) -> bool {
        for cell in self.get_cells() {
            println!("Validating edges for cell: {:?}", cell); // Debugging statement
            let mut edge_set = FxHashSet::default(); // Tracks edges within the current cell.

            if !self.validate_unique_edges_in_cell(&cell, &mut edge_set) {
                println!("Duplicate edge detected in cell: {:?}", cell); // Debugging statement
                return false;
            }
        }
        true
    }

    /// Retrieves all `Cell` entities from the mesh.
    ///
    /// Filters the list of entities to include only cells.
    ///
    /// # Returns
    /// * `Vec<MeshEntity>` - A vector of all `Cell` entities.
    fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities
            .iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_))) // Include only `Cell` entities.
            .cloned()
            .collect()
    }

    /// Validates the connectivity of a single cell.
    ///
    /// Ensures that:
    /// - The cell is connected to valid `Face` entities.
    /// - Each face is connected to valid `Vertex` entities.
    ///
    /// # Arguments
    /// * `cell` - The `Cell` entity to validate.
    ///
    /// # Returns
    /// * `true` if the cell is correctly connected, `false` otherwise.
    fn validate_cell_connectivity(&self, cell: &MeshEntity) -> bool {
        // Ensure the entity is a Cell
        if !matches!(cell, MeshEntity::Cell(_)) {
            eprintln!("Error: Entity {:?} is not a valid Cell.", cell);
            return false;
        }

        // Retrieve connected faces using the cone function
        let connected_faces = match self.sieve.cone(cell) {
            Ok(faces) => faces,
            Err(err) => {
                eprintln!(
                    "Error retrieving faces connected to Cell {:?}: {}",
                    cell, err
                );
                return false;
            }
        };

        for face in connected_faces {
            // Validate that the entity is a `Face`.
            if !matches!(face, MeshEntity::Face(_)) {
                eprintln!(
                    "Error: Connected entity {:?} is not a valid Face for Cell {:?}.",
                    face, cell
                );
                return false;
            }

            // Retrieve vertices connected to the face
            let connected_vertices = match self.sieve.cone(&face) {
                Ok(vertices) => vertices,
                Err(err) => {
                    eprintln!(
                        "Error retrieving vertices connected to Face {:?}: {}",
                        face, err
                    );
                    return false;
                }
            };

            // Validate that all connected entities are `Vertex`.
            if !connected_vertices.iter().all(|v| matches!(v, MeshEntity::Vertex(_))) {
                eprintln!(
                    "Error: Not all connected entities for Face {:?} are valid Vertices: {:?}",
                    face, connected_vertices
                );
                return false;
            }
        }

        true
    }


    /// Validates that edges within a cell are unique.
    ///
    /// Ensures no duplicate edges exist within the same cell and all edges
    /// adhere to the expected topology.
    ///
    /// # Arguments
    /// * `cell` - The `Cell` entity to validate.
    /// * `edge_set` - A mutable set used to track edges within the cell.
    ///
    /// # Returns
    /// * `true` if all edges are unique, `false` otherwise.
    fn validate_unique_edges_in_cell(
        &self,
        cell: &MeshEntity,
        edge_set: &mut FxHashSet<MeshEntity>,
    ) -> bool {
        // Ensure the entity is a Cell
        if !matches!(cell, MeshEntity::Cell(_)) {
            eprintln!("Error: Entity {:?} is not a valid Cell.", cell);
            return false;
        }

        // Retrieve edges connected to the cell
        let connected_edges = match self.sieve.cone(cell) {
            Ok(edges) => edges,
            Err(err) => {
                eprintln!(
                    "Error retrieving edges connected to Cell {:?}: {}",
                    cell, err
                );
                return false;
            }
        };

        for edge in connected_edges {
            // Validate that the entity is an `Edge`.
            if !matches!(edge, MeshEntity::Edge(_)) {
                eprintln!(
                    "Error: Connected entity {:?} is not a valid Edge for Cell {:?}.",
                    edge, cell
                );
                return false;
            }

            // Debugging: Print edge and current edge set.
            println!(
                "Checking edge {:?} in cell {:?}. Current edge set: {:?}",
                edge, cell, edge_set
            );

            // Check for duplicates in `edge_set`.
            if !edge_set.insert(edge.clone()) {
                eprintln!(
                    "Error: Duplicate edge {:?} found in Cell {:?}.",
                    edge, cell
                );
                return false; // Duplicate edge detected.
            }
        }

        true
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::Mesh;

    #[test]
    fn test_valid_connectivity() {
        let mesh = Mesh::new();
        // Adding sample entities and relationships to the mesh for testing
        let cell = MeshEntity::Cell(1);
        let face1 = MeshEntity::Face(1);
        let face2 = MeshEntity::Face(2);
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);

        mesh.add_entity(cell).unwrap();
        mesh.add_entity(face1).unwrap();
        mesh.add_entity(face2).unwrap();
        mesh.add_entity(vertex1).unwrap();
        mesh.add_entity(vertex2).unwrap();
        mesh.add_entity(vertex3).unwrap();

        mesh.add_arrow(cell, face1).unwrap();
        mesh.add_arrow(cell, face2).unwrap();
        mesh.add_arrow(face1, vertex1).unwrap();
        mesh.add_arrow(face1, vertex2).unwrap();
        mesh.add_arrow(face2, vertex2).unwrap();
        mesh.add_arrow(face2, vertex3).unwrap();

        let topology_validation = TopologyValidation::new(&mesh);
        assert!(topology_validation.validate_connectivity(), "Connectivity validation failed");
    }

    #[test]
    fn test_unique_relationships() {
        let mesh = Mesh::new();
        // Adding sample entities and relationships to the mesh for testing
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let edge1 = MeshEntity::Edge(1);
        let edge2 = MeshEntity::Edge(2);
        let edge3 = MeshEntity::Edge(3);

        mesh.add_entity(cell1).unwrap();
        mesh.add_entity(cell2).unwrap();
        mesh.add_entity(edge1).unwrap();
        mesh.add_entity(edge2).unwrap();
        mesh.add_entity(edge3).unwrap();

        // Establish valid relationships between cells and edges
        mesh.add_arrow(cell1, edge1).unwrap();
        mesh.add_arrow(cell1, edge2).unwrap();
        mesh.add_arrow(cell2, edge2).unwrap(); // Edge2 is reused here, valid as it's unique within each cell.
        mesh.add_arrow(cell2, edge3).unwrap();

        // Initialize TopologyValidation to verify unique relationships per cell
        let topology_validation = TopologyValidation::new(&mesh);

        // Check that relationships are valid and unique within the constraints of the current design
        assert!(topology_validation.validate_unique_relationships(), "Unique relationships validation failed");

        // Additional checks for edge-sharing across cells can be done if necessary
    }


}
