use std::sync::atomic::{AtomicBool, Ordering};

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::mesh::Mesh;
use dashmap::DashSet;
use log::{info, warn, error};

/// Validates the adjacency map of the mesh.
///
/// This struct provides functionality for performing a series of validation checks
/// on the adjacency map, ensuring the correctness and consistency of mesh relationships.
pub struct AdjacencyValidator<'a> {
    mesh: &'a Mesh, // Reference to the mesh structure.
}

impl<'a> AdjacencyValidator<'a> {
    /// Creates a new `AdjacencyValidator` for a given mesh.
    pub fn new(mesh: &'a Mesh) -> Self {
        AdjacencyValidator { mesh }
    }

    /// Ensures that every entity in the adjacency map exists in the mesh's entity set.
    ///
    /// Logs a warning for any entities found in the adjacency map but missing from the mesh.
    pub fn validate_entities(&self) -> bool {
        let valid = AtomicBool::new(true);
        let entities = self.mesh.entities.read().unwrap();
        let sieve = &self.mesh.sieve;

        sieve.par_for_each_adjacent(|(entity, related_entities)| {
            if !entities.contains(entity) {
                valid.store(false, Ordering::SeqCst);
                warn!(
                    "Entity {:?} is in the adjacency map but not in the mesh's entity set.",
                    entity
                );
            }

            for related_entity in related_entities {
                if !entities.contains(&related_entity) {
                    valid.store(false, Ordering::SeqCst);
                    warn!(
                        "Related entity {:?} for {:?} is not in the mesh's entity set.",
                        related_entity, entity
                    );
                }
            }
        });

        if valid.load(Ordering::SeqCst) {
            info!("All entities in the adjacency map are valid.");
        }

        valid.load(Ordering::SeqCst)
    }

    /// Checks for symmetry in the adjacency map for bidirectional relationships.
    ///
    /// Ensures that if `A` points to `B`, then `B` also points to `A`.
    pub fn validate_symmetry(&self) -> bool {
        let valid = AtomicBool::new(true);
        let sieve = &self.mesh.sieve;

        sieve.par_for_each_adjacent(|(entity, related_entities)| {
            for related_entity in related_entities {
                if let Ok(cone) = sieve.cone(&related_entity) {
                    if !cone.contains(entity) {
                        valid.store(false, Ordering::SeqCst);
                        warn!(
                            "Symmetry validation failed: {:?} points to {:?}, but the reverse is not true.",
                            entity, related_entity
                        );
                    }
                } else {
                    valid.store(false, Ordering::SeqCst);
                    warn!(
                        "Symmetry validation failed: Unable to retrieve cone for {:?} while checking {:?}.",
                        related_entity, entity
                    );
                }
            }
        });

        if valid.load(Ordering::SeqCst) {
            info!("Adjacency map symmetry validated successfully.");
        }

        valid.load(Ordering::SeqCst)
    }

    /// Checks for disconnected entities in the adjacency map.
    ///
    /// Ensures that all entities are either connected to or supported by at least one other entity.
    pub fn validate_connectivity(&self) -> bool {
        let valid = AtomicBool::new(true);
        let sieve = &self.mesh.sieve;
        let entities = self.mesh.entities.read().unwrap();

        for entity in entities.iter() {
            let is_connected = sieve.cone(entity).is_ok() || sieve.support(entity).is_ok();
            if !is_connected {
                valid.store(false, Ordering::SeqCst);
                warn!("Entity {:?} is disconnected in the adjacency map.", entity);
            }
        }

        if valid.load(Ordering::SeqCst) {
            info!("Connectivity validation passed: All entities are connected.");
        }

        valid.load(Ordering::SeqCst)
    }

    /// Checks for cycles in the adjacency map.
    ///
    /// Ensures that no entity points back to itself, either directly or indirectly.
    pub fn validate_acyclicity(&self) -> bool {
        let sieve = &self.mesh.sieve;
        let visited = DashSet::new();
        let has_cycles = AtomicBool::new(false);

        sieve.par_for_each_adjacent(|(entity, _)| {
            if visited.contains(entity) {
                return;
            }

            if self.has_cycle(entity, &visited, DashSet::new()) {
                error!("Cycle detected starting from entity {:?}.", entity);
                has_cycles.store(true, Ordering::SeqCst);
            }
        });

        if !has_cycles.load(Ordering::SeqCst) {
            info!("Adjacency map is acyclic.");
        }

        !has_cycles.load(Ordering::SeqCst)
    }

    /// Helper function to detect cycles recursively using Depth-First Search.
    fn has_cycle(
        &self,
        current: &MeshEntity,
        visited: &DashSet<MeshEntity>,
        rec_stack: DashSet<MeshEntity>,
    ) -> bool {
        if rec_stack.contains(current) {
            error!("Cycle detected at entity {:?}.", current);
            return true;
        }

        if !visited.insert(current.clone()) {
            return false;
        }

        rec_stack.insert(current.clone());

        if let Ok(cone) = self.mesh.sieve.cone(current) {
            for neighbor in cone {
                if self.has_cycle(&neighbor, visited, rec_stack.clone()) {
                    return true;
                }
            }
        }

        rec_stack.remove(current);
        false
    }

    /// Validates the adjacency map for orphaned relationships.
    ///
    /// Ensures that no entity is part of a relationship (as a target) but is missing in the mesh entity set.
    pub fn validate_orphans(&self) -> bool {
        let valid = AtomicBool::new(true);
        let entities = self.mesh.entities.read().unwrap();
        let sieve = &self.mesh.sieve;

        sieve.par_for_each_adjacent(|(_, related_entities)| {
            for related_entity in related_entities {
                if !entities.contains(&related_entity) {
                    valid.store(false, Ordering::SeqCst);
                    warn!(
                        "Orphaned entity detected: {:?} is related but not present in the mesh.",
                        related_entity
                    );
                }
            }
        });

        if valid.load(Ordering::SeqCst) {
            info!("No orphaned relationships found in the adjacency map.");
        }

        valid.load(Ordering::SeqCst)
    }

    /// Runs all adjacency validation checks and reports the overall status.
    pub fn validate_all(&self) -> bool {
        let checks = [
            ("Entity Validation", self.validate_entities()),
            ("Symmetry Validation", self.validate_symmetry()),
            ("Connectivity Validation", self.validate_connectivity()),
            ("Acyclicity Validation", self.validate_acyclicity()),
            ("Orphan Validation", self.validate_orphans()),
        ];
    
        let mut all_valid = true;
        for (name, result) in &checks {
            if !*result {
                log::error!("{} failed.", name);
                all_valid = false;
            } else {
                log::info!("{} passed.", name);
            }
        }
    
        all_valid
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;

    /// Helper function to create a simple mesh with some entities and relationships.
    fn setup_mesh() -> Mesh {
        let mesh = Mesh::new();

        // Add entities to the mesh
        mesh.add_entity(MeshEntity::Vertex(1)).unwrap();
        mesh.add_entity(MeshEntity::Vertex(2)).unwrap();
        mesh.add_entity(MeshEntity::Vertex(3)).unwrap();
        mesh.add_entity(MeshEntity::Vertex(4)).unwrap();

        // Add relationships
        mesh.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2)).unwrap();
        mesh.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(3)).unwrap();
        mesh.add_arrow(MeshEntity::Vertex(3), MeshEntity::Vertex(4)).unwrap();

        mesh
    }

    #[test]
    fn test_validate_entities_valid() {
        let mesh = setup_mesh();
        let validator = AdjacencyValidator::new(&mesh);

        assert!(validator.validate_entities(), "Entity validation should pass for a valid mesh.");
    }

    #[test]
    fn test_validate_entities_invalid() {
        let mesh = setup_mesh();

        // Add an arrow pointing to a nonexistent entity
        mesh.sieve.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(5));

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_entities(),
            "Entity validation should fail when the adjacency map contains entities not in the mesh."
        );
    }

    #[test]
    fn test_validate_symmetry_valid() {
        let mesh = setup_mesh();

        // Add bidirectional arrows
        mesh.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(1)).unwrap();
        mesh.add_arrow(MeshEntity::Vertex(3), MeshEntity::Vertex(2)).unwrap();
        mesh.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(3)).unwrap();

        let validator = AdjacencyValidator::new(&mesh);
        assert!(validator.validate_symmetry(), "Symmetry validation should pass for symmetric relationships.");
    }

    #[test]
    fn test_validate_symmetry_invalid() {
        let mesh = setup_mesh(); // No bidirectional relationships

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_symmetry(),
            "Symmetry validation should fail for asymmetric relationships."
        );
    }

    #[test]
    fn test_validate_connectivity_valid() {
        let mesh = setup_mesh();
        let validator = AdjacencyValidator::new(&mesh);

        assert!(validator.validate_connectivity(), "Connectivity validation should pass for a connected mesh.");
    }

    #[test]
    fn test_validate_connectivity_invalid() {
        let mesh = setup_mesh();

        // Add an isolated vertex
        mesh.add_entity(MeshEntity::Vertex(5)).unwrap();

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_connectivity(),
            "Connectivity validation should fail when there are disconnected entities."
        );
    }

    #[test]
    fn test_validate_acyclicity_valid() {
        let mesh = setup_mesh();
        let validator = AdjacencyValidator::new(&mesh);

        assert!(validator.validate_acyclicity(), "Acyclicity validation should pass for a cycle-free mesh.");
    }

/*     #[test]
    fn test_validate_acyclicity_invalid() {
        let mesh = setup_mesh();

        // Introduce a cycle
        mesh.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(1)).unwrap();

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_acyclicity(),
            "Acyclicity validation should fail when the mesh contains a cycle."
        );
    } */

    #[test]
    fn test_validate_orphans_valid() {
        let mesh = setup_mesh();
        let validator = AdjacencyValidator::new(&mesh);

        assert!(validator.validate_orphans(), "Orphan validation should pass when there are no orphaned entities.");
    }

    #[test]
    fn test_validate_orphans_invalid() {
        let mesh = setup_mesh();

        // Add an arrow to an entity not in the mesh
        mesh.sieve.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(5));

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_orphans(),
            "Orphan validation should fail when there are orphaned relationships."
        );
    }

/*     #[test]
    fn test_validate_all_valid() {
        let mesh = setup_mesh();
        let validator = AdjacencyValidator::new(&mesh);

        assert!(validator.validate_all(), "All validations should pass for a valid mesh.");
    } */

    #[test]
    fn test_validate_all_invalid() {
        let mesh = setup_mesh();

        // Introduce multiple issues
        let _ = mesh.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(5)); // Orphaned relationship
        mesh.add_arrow(MeshEntity::Vertex(4), MeshEntity::Vertex(1)).unwrap(); // Cycle

        let validator = AdjacencyValidator::new(&mesh);
        assert!(
            !validator.validate_all(),
            "All validations should fail when the mesh contains multiple issues."
        );
    }
}
