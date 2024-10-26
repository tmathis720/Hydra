use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;

/// The `Overlap` struct manages two sets of `MeshEntity` elements:  
/// - `local_entities`: Entities that are local to the current partition.
/// - `ghost_entities`: Entities that are shared with other partitions.
pub struct Overlap {
    /// A thread-safe set of local entities.  
    pub local_entities: Arc<DashMap<MeshEntity, ()>>,
    /// A thread-safe set of ghost entities.  
    pub ghost_entities: Arc<DashMap<MeshEntity, ()>>,
}

impl Overlap {
    /// Creates a new `Overlap` with empty sets for local and ghost entities.
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(DashMap::new()),
            ghost_entities: Arc::new(DashMap::new()),
        }
    }

    /// Adds a `MeshEntity` to the set of local entities.
    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.insert(entity, ());
    }

    /// Adds a `MeshEntity` to the set of ghost entities.
    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.insert(entity, ());
    }

    /// Checks if a `MeshEntity` is a local entity.
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        self.local_entities.contains_key(entity)
    }

    /// Checks if a `MeshEntity` is a ghost entity.
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        self.ghost_entities.contains_key(entity)
    }

    /// Retrieves a clone of all local entities.
    pub fn local_entities(&self) -> Vec<MeshEntity> {
        self.local_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves a clone of all ghost entities.
    pub fn ghost_entities(&self) -> Vec<MeshEntity> {
        self.ghost_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Merges another `Overlap` instance into this one, combining local  
    /// and ghost entities from both overlaps.
    pub fn merge(&self, other: &Overlap) {
        other.local_entities.iter().for_each(|entry| {
            self.local_entities.insert(entry.key().clone(), ());
        });

        other.ghost_entities.iter().for_each(|entry| {
            self.ghost_entities.insert(entry.key().clone(), ());
        });
    }
}

/// The `Delta` struct manages transformation data for `MeshEntity` elements  
/// in overlapping regions. It is used to store and apply data transformations  
/// across entities in distributed environments.
pub struct Delta<T> {
    /// A thread-safe map storing transformation data associated with `MeshEntity` objects.  
    pub data: Arc<DashMap<MeshEntity, T>>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty `Delta`.
    pub fn new() -> Self {
        Delta {
            data: Arc::new(DashMap::new()),
        }
    }

    /// Sets the transformation data for a specific `MeshEntity`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves the transformation data associated with a specific `MeshEntity`.
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|entry| entry.clone())
    }

    /// Removes the transformation data associated with a specific `MeshEntity`.
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        self.data.remove(entity).map(|(_, value)| value)
    }

    /// Checks if there is transformation data for a specific `MeshEntity`.
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        self.data.contains_key(entity)
    }

    /// Applies a function to all entities in the delta.
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        self.data.iter().for_each(|entry| func(entry.key(), entry.value()));
    }

    /// Merges another `Delta` instance into this one, combining data from both deltas.
    pub fn merge(&self, other: &Delta<T>)
    where
        T: Clone,
    {
        other.data.iter().for_each(|entry| {
            self.data.insert(entry.key().clone(), entry.value().clone());
        });
    }
}

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
