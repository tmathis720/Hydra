use std::sync::atomic::{AtomicBool, Ordering};

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashSet;
use log::{info, warn, error};

/// A struct for validating the adjacency relationships in a `Sieve`.
pub struct SieveValidator<'a> {
    sieve: &'a Sieve,
}

impl<'a> SieveValidator<'a> {
    /// Creates a new instance of the `SieveValidator`.
    ///
    /// # Arguments
    /// - `sieve`: A reference to the `Sieve` whose adjacency relationships will be validated.
    ///
    /// # Returns
    /// - A new `SieveValidator` instance.
    pub fn new(sieve: &'a Sieve) -> Self {
        SieveValidator { sieve }
    }

    /// Validates that all entities in the adjacency map are consistent.
    ///
    /// Checks if:
    /// - Each entity points to valid entities.
    /// - No entity points to itself.
    ///
    /// Logs warnings or errors for any inconsistencies found.
    pub fn validate_consistency(&self) -> Result<(), String> {
        let valid = AtomicBool::new(true); // Use an AtomicBool for thread-safe mutability.
    
        self.sieve.par_for_each_adjacent(|(entity, related_entities)| {
            if related_entities.contains(entity) {
                error!(
                    "Entity {:?} has a self-referencing relationship in the adjacency map.",
                    entity
                );
                valid.store(false, Ordering::SeqCst); // Safely update the AtomicBool.
            }
        });
    
        if valid.load(Ordering::SeqCst) {
            info!("Adjacency map consistency validated successfully.");
            Ok(())
        } else {
            Err("Adjacency map contains self-referencing relationships.".into())
        }
    }
    

    /// Checks for entities with no connections in the adjacency map.
    ///
    /// Logs warnings for isolated entities.
    pub fn validate_isolated_entities(&self, all_entities: &FxHashSet<MeshEntity>) -> Result<(), String> {
        let mut isolated_entities = FxHashSet::default();

        for entity in all_entities.iter() {
            let is_connected = self.sieve.cone(entity).is_ok()
                || self.sieve.support(entity).is_ok();
            if !is_connected {
                isolated_entities.insert(*entity);
                warn!("Entity {:?} is isolated in the adjacency map.", entity);
            }
        }

        if isolated_entities.is_empty() {
            info!("No isolated entities found in the adjacency map.");
            Ok(())
        } else {
            Err(format!("Found {} isolated entities.", isolated_entities.len()))
        }
    }

    /// Detects cyclic dependencies in the adjacency map.
    ///
    /// Uses depth-first search (DFS) to detect cycles in the graph represented by the adjacency map.
    pub fn validate_no_cycles(&self) -> Result<(), String> {
        let mut visited = FxHashSet::default();
        let mut stack = FxHashSet::default();
        let mut has_cycle = false;

        for entry in self.sieve.adjacency.iter() {
            if self.dfs(entry.key(), &mut visited, &mut stack) {
                has_cycle = true;
            }
        }

        if has_cycle {
            Err("Cyclic dependencies detected in the adjacency map.".into())
        } else {
            info!("No cycles detected in the adjacency map.");
            Ok(())
        }
    }

    /// Depth-first search (DFS) helper for cycle detection.
    ///
    /// Returns `true` if a cycle is detected, otherwise `false`.
    fn dfs(
        &self,
        entity: &MeshEntity,
        visited: &mut FxHashSet<MeshEntity>,
        stack: &mut FxHashSet<MeshEntity>,
    ) -> bool {
        if stack.contains(entity) {
            error!("Cycle detected at entity {:?}.", entity);
            return true;
        }

        if visited.contains(entity) {
            return false;
        }

        visited.insert(*entity);
        stack.insert(*entity);

        if let Ok(related_entities) = self.sieve.cone(entity) {
            for related_entity in related_entities {
                if self.dfs(&related_entity, visited, stack) {
                    return true;
                }
            }
        }

        stack.remove(entity);
        false
    }

    /// Validates that all entities in the adjacency map are reachable.
    ///
    /// This ensures that no orphaned entities exist that cannot be accessed from others.
    pub fn validate_reachability(&self, all_entities: &FxHashSet<MeshEntity>) -> Result<(), String> {
        let mut reachable = FxHashSet::default();

        for entity in all_entities.iter() {
            if let Ok(closure) = self.sieve.closure(entity) {
                closure.iter().for_each(|entry| {
                    reachable.insert(*entry.key());
                });
            }
        }

        let unreachable_entities: FxHashSet<_> = all_entities.difference(&reachable).cloned().collect();

        if unreachable_entities.is_empty() {
            info!("All entities are reachable in the adjacency map.");
            Ok(())
        } else {
            for entity in unreachable_entities.iter() {
                warn!("Entity {:?} is unreachable in the adjacency map.", entity);
            }
            Err(format!(
                "Found {} unreachable entities in the adjacency map.",
                unreachable_entities.len()
            ))
        }
    }

    /// Performs all adjacency map validations in sequence.
    ///
    /// This includes:
    /// - Consistency validation.
    /// - Isolation check.
    /// - Cycle detection.
    /// - Reachability check.
    ///
    /// # Arguments
    /// - `all_entities`: A set of all mesh entities expected in the adjacency map.
    ///
    /// # Returns
    /// - `Ok(())` if all validations pass.
    /// - `Err(String)` if any validation fails.
    pub fn validate_all(&self, all_entities: &FxHashSet<MeshEntity>) -> Result<(), String> {
        self.validate_consistency()?;
        self.validate_isolated_entities(all_entities)?;
        self.validate_no_cycles()?;
        self.validate_reachability(all_entities)?;
        info!("All adjacency map validations passed successfully.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::sieve::Sieve;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_validate_consistency() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_consistency().is_ok());
    }

    #[test]
    fn test_validate_isolated_entities() {
        let sieve = Sieve::new();
        let all_entities: FxHashSet<_> = vec![MeshEntity::Vertex(1), MeshEntity::Vertex(2)]
            .into_iter()
            .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_isolated_entities(&all_entities).is_err());
    }

    #[test]
    fn test_validate_no_cycles() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));
        sieve.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(1));

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_no_cycles().is_err());
    }

    #[test]
    fn test_validate_reachability() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));

        let all_entities: FxHashSet<_> = vec![MeshEntity::Vertex(1), MeshEntity::Vertex(2)]
            .into_iter()
            .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_reachability(&all_entities).is_ok());
    }
}

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use crate::domain::sieve::Sieve;
    use crate::domain::mesh_entity::MeshEntity;
    use rustc_hash::FxHashSet;

    fn setup_sieve_with_data() -> (Sieve, FxHashSet<MeshEntity>) {
        let sieve = Sieve::new();

        // Add entities and relationships
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));
        sieve.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(3));
        sieve.add_arrow(MeshEntity::Vertex(3), MeshEntity::Vertex(4));

        let all_entities: FxHashSet<_> = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
            MeshEntity::Vertex(4),
        ]
        .into_iter()
        .collect();

        (sieve, all_entities)
    }

    #[test]
    fn test_validate_consistency_valid() {
        let (sieve, _) = setup_sieve_with_data();
        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_consistency().is_ok());
    }

    #[test]
    fn test_validate_consistency_self_reference() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(1));

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_consistency().is_err());
    }

    #[test]
    fn test_validate_isolated_entities_valid() {
        let (sieve, all_entities) = setup_sieve_with_data();
        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_isolated_entities(&all_entities).is_ok());
    }

    #[test]
    fn test_validate_isolated_entities_invalid() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));

        let all_entities: FxHashSet<_> = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3), // Isolated entity
        ]
        .into_iter()
        .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_isolated_entities(&all_entities).is_err());
    }

    #[test]
    fn test_validate_no_cycles_valid() {
        let (sieve, _) = setup_sieve_with_data();
        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_no_cycles().is_ok());
    }

    #[test]
    fn test_validate_no_cycles_invalid() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));
        sieve.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(3));
        sieve.add_arrow(MeshEntity::Vertex(3), MeshEntity::Vertex(1)); // Cycle

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_no_cycles().is_err());
    }

    #[test]
    fn test_validate_reachability_valid() {
        let (sieve, all_entities) = setup_sieve_with_data();
        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_reachability(&all_entities).is_ok());
    }

/*     #[test]
    fn test_validate_reachability_unreachable_entities() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));

        let all_entities: FxHashSet<_> = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3), // Unreachable
        ]
        .into_iter()
        .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_reachability(&all_entities).is_err());
    } */

    #[test]
    fn test_validate_all_pass() {
        let (sieve, all_entities) = setup_sieve_with_data();
        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_all(&all_entities).is_ok());
    }

    #[test]
    fn test_validate_all_fail_on_cycles() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));
        sieve.add_arrow(MeshEntity::Vertex(2), MeshEntity::Vertex(1)); // Cycle

        let all_entities: FxHashSet<_> = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
        ]
        .into_iter()
        .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_all(&all_entities).is_err());
    }

    #[test]
    fn test_validate_all_fail_on_isolated_entities() {
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Vertex(2));

        let all_entities: FxHashSet<_> = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3), // Isolated
        ]
        .into_iter()
        .collect();

        let validator = SieveValidator::new(&sieve);
        assert!(validator.validate_all(&all_entities).is_err());
    }
}
