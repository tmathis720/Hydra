use std::collections::{HashMap, HashSet};
use crate::domain::mesh_entity::MeshEntity;

/// Overlap structure to handle relationships between local and ghost entities
pub struct Overlap {
    pub local_entities: HashSet<MeshEntity>,  // Local mesh entities
    pub ghost_entities: HashSet<MeshEntity>,  // Entities shared with other processes
}

impl Overlap {
    /// Creates a new, empty Overlap
    pub fn new() -> Self {
        Overlap {
            local_entities: HashSet::new(),
            ghost_entities: HashSet::new(),
        }
    }

    /// Add a local entity to the overlap
    pub fn add_local_entity(&mut self, entity: MeshEntity) {
        self.local_entities.insert(entity);
    }

    /// Add a ghost entity to the overlap (shared with other processes)
    pub fn add_ghost_entity(&mut self, entity: MeshEntity) {
        self.ghost_entities.insert(entity);
    }

    /// Check if an entity is local
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        self.local_entities.contains(entity)
    }

    /// Check if an entity is a ghost entity (shared with other processes)
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        self.ghost_entities.contains(entity)
    }

    /// Get all local entities
    pub fn local_entities(&self) -> &HashSet<MeshEntity> {
        &self.local_entities
    }

    /// Get all ghost entities
    pub fn ghost_entities(&self) -> &HashSet<MeshEntity> {
        &self.ghost_entities
    }

    /// Merge another overlap into this one (used when communicating between partitions)
    pub fn merge(&mut self, other: &Overlap) {
        self.local_entities.extend(&other.local_entities);
        self.ghost_entities.extend(&other.ghost_entities);
    }
}

/// Delta structure to manage transformation and data consistency across overlaps
pub struct Delta<T> {
    pub data: HashMap<MeshEntity, T>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty Delta
    pub fn new() -> Self {
        Delta {
            data: HashMap::new(),
        }
    }

    /// Set transformation data for a specific mesh entity
    pub fn set_data(&mut self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Get transformation data for a specific entity
    pub fn get_data(&self, entity: &MeshEntity) -> Option<&T> {
        self.data.get(entity)
    }

    /// Remove the data associated with a mesh entity
    pub fn remove_data(&mut self, entity: &MeshEntity) -> Option<T> {
        self.data.remove(entity)
    }

    /// Check if there is transformation data for a specific entity
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        self.data.contains_key(entity)
    }

    /// Apply a function to all entities in the delta
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        for (entity, value) in &self.data {
            func(entity, value);
        }
    }

    /// Merge another delta into this one (used to combine data from different partitions)
    pub fn merge(&mut self, other: &Delta<T>)
    where
        T: Clone,
    {
        for (entity, value) in &other.data {
            self.data.insert(entity.clone(), value.clone());
        }
    }
}

// Unit tests for the Overlap and Delta structures
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_overlap_local_and_ghost_entities() {
        let mut overlap = Overlap::new();
        let vertex_local = MeshEntity::Vertex(1);
        let vertex_ghost = MeshEntity::Vertex(2);

        overlap.add_local_entity(vertex_local);
        overlap.add_ghost_entity(vertex_ghost);

        assert!(overlap.is_local(&vertex_local));
        assert!(overlap.is_ghost(&vertex_ghost));

        assert_eq!(overlap.local_entities().len(), 1);
        assert_eq!(overlap.ghost_entities().len(), 1);
    }

    #[test]
    fn test_overlap_merge() {
        let mut overlap1 = Overlap::new();
        let mut overlap2 = Overlap::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);

        overlap1.add_local_entity(vertex1);
        overlap1.add_ghost_entity(vertex2);

        overlap2.add_local_entity(vertex3);

        overlap1.merge(&overlap2);

        assert!(overlap1.is_local(&vertex1));
        assert!(overlap1.is_ghost(&vertex2));
        assert!(overlap1.is_local(&vertex3));
        assert_eq!(overlap1.local_entities().len(), 2);
    }

    #[test]
    fn test_delta_set_and_get_data() {
        let mut delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 42);

        assert_eq!(delta.get_data(&vertex), Some(&42));
        assert!(delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_remove_data() {
        let mut delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 100);
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_merge() {
        let mut delta1 = Delta::new();
        let mut delta2 = Delta::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        delta1.set_data(vertex1, 10);
        delta2.set_data(vertex2, 20);

        delta1.merge(&delta2);

        assert_eq!(delta1.get_data(&vertex1), Some(&10));
        assert_eq!(delta1.get_data(&vertex2), Some(&20));
    }
}
