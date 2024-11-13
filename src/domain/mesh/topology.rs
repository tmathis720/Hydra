// src/boundary/mesh/topology.rs

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use crate::domain::mesh::Mesh;
use rustc_hash::FxHashSet;
use std::sync::{Arc, RwLock};

/// `TopologyValidation` struct responsible for checking mesh entity connectivity and uniqueness.
pub struct TopologyValidation<'a> {
    sieve: &'a Sieve,
    entities: &'a Arc<RwLock<FxHashSet<MeshEntity>>>,
}

impl<'a> TopologyValidation<'a> {
    /// Creates a new `TopologyValidation` instance for a given mesh.
    pub fn new(mesh: &'a Mesh) -> Self {
        TopologyValidation {
            sieve: &mesh.sieve,
            entities: &mesh.entities,
        }
    }

    /// Validates that each `Cell` in the mesh has the correct connections to `Faces` and `Vertices`.
    /// Returns `true` if all cells are correctly connected, `false` otherwise.
    pub fn validate_connectivity(&self) -> bool {
        for cell in self.get_cells() {
            if !self.validate_cell_connectivity(&cell) {
                return false;
            }
        }
        true
    }

    /// Validates that `Edges` in the mesh are unique and not duplicated within any `Cell`.
    /// Returns `true` if all edges are unique, `false` otherwise.
    pub fn validate_unique_relationships(&self) -> bool {
        for cell in self.get_cells() {
            println!("Validating edges for cell: {:?}", cell); // Debugging statement
            let mut edge_set = FxHashSet::default(); // Reset edge_set for each cell
    
            if !self.validate_unique_edges_in_cell(&cell, &mut edge_set) {
                println!("Duplicate edge detected in cell: {:?}", cell); // Debugging statement
                return false;
            }
        }
        true
    }

    /// Retrieves all `Cell` entities from the mesh.
    fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Checks if a `Cell` is connected to valid `Faces` and `Vertices`.
    fn validate_cell_connectivity(&self, cell: &MeshEntity) -> bool {
        if let Some(connected_faces) = self.sieve.cone(cell) {
            for face in connected_faces {
                if !matches!(face, MeshEntity::Face(_)) {
                    return false;
                }
                // Check each face is connected to valid vertices
                if let Some(vertices) = self.sieve.cone(&face) {
                    if !vertices.iter().all(|v| matches!(v, MeshEntity::Vertex(_))) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Checks if `Edges` within a `Cell` are unique.
    fn validate_unique_edges_in_cell(&self, cell: &MeshEntity, edge_set: &mut FxHashSet<MeshEntity>) -> bool {
        if let Some(edges) = self.sieve.cone(cell) {
            for edge in edges {
                if !matches!(edge, MeshEntity::Edge(_)) {
                    return false;
                }
                // Debugging: Print edge and current edge set
                println!("Checking edge {:?} in cell {:?}. Current edge set: {:?}", edge, cell, edge_set);
                
                // Check for duplication in `edge_set`
                if !edge_set.insert(edge) {
                    println!("Duplicate edge {:?} found in cell {:?}", edge, cell); // Debugging statement
                    return false; // Duplicate edge found
                }
            }
            true
        } else {
            false
        }
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

        mesh.add_entity(cell);
        mesh.add_entity(face1);
        mesh.add_entity(face2);
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);

        mesh.add_arrow(cell, face1);
        mesh.add_arrow(cell, face2);
        mesh.add_arrow(face1, vertex1);
        mesh.add_arrow(face1, vertex2);
        mesh.add_arrow(face2, vertex2);
        mesh.add_arrow(face2, vertex3);

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

        mesh.add_entity(cell1);
        mesh.add_entity(cell2);
        mesh.add_entity(edge1);
        mesh.add_entity(edge2);
        mesh.add_entity(edge3);

        // Establish valid relationships between cells and edges
        mesh.add_arrow(cell1, edge1);
        mesh.add_arrow(cell1, edge2);
        mesh.add_arrow(cell2, edge2); // Edge2 is reused here, valid as it's unique within each cell.
        mesh.add_arrow(cell2, edge3);

        // Initialize TopologyValidation to verify unique relationships per cell
        let topology_validation = TopologyValidation::new(&mesh);

        // Check that relationships are valid and unique within the constraints of the current design
        assert!(topology_validation.validate_unique_relationships(), "Unique relationships validation failed");

        // Additional checks for edge-sharing across cells can be done if necessary
    }


}
