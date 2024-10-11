use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crate::domain::mesh_entity::MeshEntity;

pub struct Overlap {
    pub local_entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
    pub ghost_entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
}

impl Overlap {
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(RwLock::new(FxHashSet::default())),
            ghost_entities: Arc::new(RwLock::new(FxHashSet::default())),
        }
    }

    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.write().unwrap().insert(entity);
    }

    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.write().unwrap().insert(entity);
    }

    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        let local = self.local_entities.read().unwrap();
        local.contains(entity)
    }

    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        let ghost = self.ghost_entities.read().unwrap();
        ghost.contains(entity)
    }

    /// Get a clone of all local entities
    pub fn local_entities(&self) -> FxHashSet<MeshEntity> {
        let local = self.local_entities.read().unwrap();
        local.clone()
    }

    /// Get a clone of all ghost entities
    pub fn ghost_entities(&self) -> FxHashSet<MeshEntity> {
        let ghost = self.ghost_entities.read().unwrap();
        ghost.clone()
    }

    /// Merge another overlap into this one (used when communicating between partitions)
    pub fn merge(&self, other: &Overlap) {
        let mut local = self.local_entities.write().unwrap();
        let other_local = other.local_entities.read().unwrap();
        local.extend(other_local.iter().cloned());

        let mut ghost = self.ghost_entities.write().unwrap();
        let other_ghost = other.ghost_entities.read().unwrap();
        ghost.extend(other_ghost.iter().cloned());
    }
}

/// Delta structure to manage transformation and data consistency across overlaps
pub struct Delta<T> {
    pub data: Arc<RwLock<FxHashMap<MeshEntity, T>>>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty Delta
    pub fn new() -> Self {
        Delta {
            data: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Set transformation data for a specific mesh entity
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        let mut data = self.data.write().unwrap();
        data.insert(entity, value);
    }

    /// Get transformation data for a specific entity
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        let data = self.data.read().unwrap();
        data.get(entity).cloned()
    }

    /// Remove the data associated with a mesh entity
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        let mut data = self.data.write().unwrap();
        data.remove(entity)
    }

    /// Check if there is transformation data for a specific entity
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        let data = self.data.read().unwrap();
        data.contains_key(entity)
    }

    /// Apply a function to all entities in the delta
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        let data = self.data.read().unwrap();
        for (entity, value) in data.iter() {
            func(entity, value);
        }
    }

    /// Merge another delta into this one (used to combine data from different partitions)
    pub fn merge(&self, other: &Delta<T>)
    where
        T: Clone,
    {
        let mut data = self.data.write().unwrap();
        let other_data = other.data.read().unwrap();
        for (entity, value) in other_data.iter() {
            data.insert(entity.clone(), value.clone());
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
        let overlap = Overlap::new();
        let vertex_local = MeshEntity::Vertex(1);
        let vertex_ghost = MeshEntity::Vertex(2);
        overlap.add_local_entity(vertex_local);
        overlap.add_ghost_entity(vertex_ghost);
        assert!(overlap.is_local(&vertex_local));
        assert!(overlap.is_ghost(&vertex_ghost));
    }

    #[test]
    fn test_overlap_merge() {
        let overlap1 = Overlap::new();
        let overlap2 = Overlap::new();
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
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 42);

        assert_eq!(delta.get_data(&vertex), Some(42));
        assert!(delta.has_data(&vertex));
    }

    #[test]
    fn test_overlap_entities() {
        let overlap = Overlap::new();
        let vertex = MeshEntity::Vertex(1);
        overlap.add_local_entity(vertex);
        assert!(overlap.local_entities.read().unwrap().contains(&vertex));
    }

    #[test]
    fn test_delta_remove_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 100);
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_merge() {
        let delta1 = Delta::new();
        let delta2 = Delta::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        delta1.set_data(vertex1, 10);
        delta2.set_data(vertex2, 20);

        delta1.merge(&delta2);

        assert_eq!(delta1.get_data(&vertex1), Some(10));
        assert_eq!(delta1.get_data(&vertex2), Some(20));
    }
}
