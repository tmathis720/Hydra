use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crate::domain::mesh_entity::MeshEntity;

/// The `Overlap` struct manages two sets of `MeshEntity` elements:  
/// - `local_entities`: Entities that are local to the current partition.
/// - `ghost_entities`: Entities that are shared with other partitions.  
///
/// It supports adding entities to these sets, checking whether an entity  
/// is local or ghost, and merging overlaps from different partitions.  
///
/// Example usage:
/// 
///    let overlap = Overlap::new();  
///    overlap.add_local_entity(MeshEntity::Vertex(1));  
///    assert!(overlap.is_local(&MeshEntity::Vertex(1)));  
/// 
pub struct Overlap {
    /// A thread-safe set of local entities.  
    pub local_entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
    /// A thread-safe set of ghost entities.  
    pub ghost_entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
}

impl Overlap {
    /// Creates a new `Overlap` with empty sets for local and ghost entities.  
    ///
    /// Example usage:
    /// 
    ///    let overlap = Overlap::new();  
    ///
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(RwLock::new(FxHashSet::default())),
            ghost_entities: Arc::new(RwLock::new(FxHashSet::default())),
        }
    }

    /// Adds a `MeshEntity` to the set of local entities.  
    ///
    /// Example usage:
    /// 
    ///    overlap.add_local_entity(MeshEntity::Vertex(1));  
    ///
    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.write().unwrap().insert(entity);
    }

    /// Adds a `MeshEntity` to the set of ghost entities.  
    ///
    /// Example usage:
    /// 
    ///    overlap.add_ghost_entity(MeshEntity::Vertex(2));  
    ///
    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.write().unwrap().insert(entity);
    }

    /// Checks if a `MeshEntity` is a local entity.  
    ///
    /// Returns `true` if the entity is in the local entities set, otherwise `false`.  
    ///
    /// Example usage:
    /// 
    ///    let is_local = overlap.is_local(&MeshEntity::Vertex(1));  
    ///
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        let local = self.local_entities.read().unwrap();
        local.contains(entity)
    }

    /// Checks if a `MeshEntity` is a ghost entity.  
    ///
    /// Returns `true` if the entity is in the ghost entities set, otherwise `false`.  
    ///
    /// Example usage:
    /// 
    ///    let is_ghost = overlap.is_ghost(&MeshEntity::Vertex(2));  
    ///
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        let ghost = self.ghost_entities.read().unwrap();
        ghost.contains(entity)
    }

    /// Retrieves a clone of all local entities.  
    ///
    /// Example usage:
    /// 
    ///    let local_entities = overlap.local_entities();  
    ///
    pub fn local_entities(&self) -> FxHashSet<MeshEntity> {
        let local = self.local_entities.read().unwrap();
        local.clone()
    }

    /// Retrieves a clone of all ghost entities.  
    ///
    /// Example usage:
    /// 
    ///    let ghost_entities = overlap.ghost_entities();  
    ///
    pub fn ghost_entities(&self) -> FxHashSet<MeshEntity> {
        let ghost = self.ghost_entities.read().unwrap();
        ghost.clone()
    }

    /// Merges another `Overlap` instance into this one, combining local  
    /// and ghost entities from both overlaps.  
    ///
    /// Example usage:
    /// 
    ///    overlap1.merge(&overlap2);  
    ///
    pub fn merge(&self, other: &Overlap) {
        let mut local = self.local_entities.write().unwrap();
        let other_local = other.local_entities.read().unwrap();
        local.extend(other_local.iter().cloned());

        let mut ghost = self.ghost_entities.write().unwrap();
        let other_ghost = other.ghost_entities.read().unwrap();
        ghost.extend(other_ghost.iter().cloned());
    }
}

/// The `Delta` struct manages transformation data for `MeshEntity` elements  
/// in overlapping regions. It is used to store and apply data transformations  
/// across entities in distributed environments.  
///
/// Example usage:
/// 
///    let delta = Delta::new();  
///    delta.set_data(MeshEntity::Vertex(1), 42);  
///    assert_eq!(delta.get_data(&MeshEntity::Vertex(1)), Some(42));  
/// 
pub struct Delta<T> {
    /// A thread-safe map storing transformation data associated with `MeshEntity` objects.  
    pub data: Arc<RwLock<FxHashMap<MeshEntity, T>>>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty `Delta`.  
    ///
    /// Example usage:
    /// 
    ///    let delta = Delta::new();  
    ///
    pub fn new() -> Self {
        Delta {
            data: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Sets the transformation data for a specific `MeshEntity`.  
    ///
    /// Example usage:
    /// 
    ///    delta.set_data(MeshEntity::Vertex(1), 42);  
    ///
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        let mut data = self.data.write().unwrap();
        data.insert(entity, value);
    }

    /// Retrieves the transformation data associated with a specific `MeshEntity`.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    /// 
    ///    let value = delta.get_data(&MeshEntity::Vertex(1));  
    ///
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        let data = self.data.read().unwrap();
        data.get(entity).cloned()
    }

    /// Removes the transformation data associated with a specific `MeshEntity`.  
    ///
    /// Returns the removed data if it exists, otherwise `None`.  
    ///
    /// Example usage:
    /// 
    ///    let removed_value = delta.remove_data(&MeshEntity::Vertex(1));  
    ///
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        let mut data = self.data.write().unwrap();
        data.remove(entity)
    }

    /// Checks if there is transformation data for a specific `MeshEntity`.  
    ///
    /// Returns `true` if the entity has associated data, otherwise `false`.  
    ///
    /// Example usage:
    /// 
    ///    let has_data = delta.has_data(&MeshEntity::Vertex(1));  
    ///
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        let data = self.data.read().unwrap();
        data.contains_key(entity)
    }

    /// Applies a function to all entities in the delta.  
    ///
    /// Example usage:
    /// 
    ///    delta.apply(|entity, value| {  
    ///        println!("{:?}: {:?}", entity, value);  
    ///    });  
    ///
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        let data = self.data.read().unwrap();
        for (entity, value) in data.iter() {
            func(entity, value);
        }
    }

    /// Merges another `Delta` instance into this one, combining data from both deltas.  
    ///
    /// Example usage:
    /// 
    ///    delta1.merge(&delta2);  
    ///
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies the addition and retrieval of local and ghost entities  
    /// in the `Overlap` structure.  
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
    /// Test that verifies merging two `Overlap` structures works as expected.  
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
    /// Test that verifies setting and retrieving data in the `Delta` structure  
    /// works as expected.  
    fn test_delta_set_and_get_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 42);

        assert_eq!(delta.get_data(&vertex), Some(42));
        assert!(delta.has_data(&vertex));
    }

    #[test]
    /// Test that verifies removing data from the `Delta` structure works as expected.  
    fn test_delta_remove_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 100);
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex));
    }

    #[test]
    /// Test that verifies merging two `Delta` structures works as expected.  
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
