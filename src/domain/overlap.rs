use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;

/// The `Overlap` struct manages partitioning in a distributed mesh.
/// 
/// It maintains two categories of `MeshEntity` elements:
/// - **`local_entities`**: Entities owned by the current partition.
/// - **`ghost_entities`**: Entities shared with other partitions for overlap consistency.
pub struct Overlap {
    /// A thread-safe set of entities that are local to the current partition.
    /// 
    /// These entities are fully owned and managed by this partition, and computations
    /// on them do not require data from other partitions.
    pub local_entities: Arc<DashMap<MeshEntity, ()>>,

    /// A thread-safe set of ghost entities shared with other partitions.
    /// 
    /// These entities reside at the boundaries of the partition and require
    /// synchronization or communication with other partitions to ensure data consistency.
    pub ghost_entities: Arc<DashMap<MeshEntity, ()>>,
}

impl Overlap {
    /// Creates a new `Overlap` instance with empty local and ghost entity sets.
    ///
    /// # Returns
    /// - A new `Overlap` instance with no entities.
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(DashMap::new()),
            ghost_entities: Arc::new(DashMap::new()),
        }
    }

    /// Adds a `MeshEntity` to the set of local entities.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` to be added to the local entities.
    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.insert(entity, ());
    }

    /// Adds a `MeshEntity` to the set of ghost entities.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` to be added to the ghost entities.
    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.insert(entity, ());
    }

    /// Checks whether a given `MeshEntity` is a local entity.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` to check.
    ///
    /// # Returns
    /// - `true` if the entity exists in `local_entities`, otherwise `false`.
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        self.local_entities.contains_key(entity)
    }

    /// Checks whether a given `MeshEntity` is a ghost entity.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` to check.
    ///
    /// # Returns
    /// - `true` if the entity exists in `ghost_entities`, otherwise `false`.
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        self.ghost_entities.contains_key(entity)
    }

    /// Retrieves all local entities as a vector of `MeshEntity` objects.
    ///
    /// # Returns
    /// - A `Vec<MeshEntity>` containing all entities in `local_entities`.
    pub fn local_entities(&self) -> Vec<MeshEntity> {
        self.local_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all ghost entities as a vector of `MeshEntity` objects.
    ///
    /// # Returns
    /// - A `Vec<MeshEntity>` containing all entities in `ghost_entities`.
    pub fn ghost_entities(&self) -> Vec<MeshEntity> {
        self.ghost_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Merges another `Overlap` instance into this one.
    ///
    /// Combines the local and ghost entities of the provided `Overlap` instance
    /// with those in the current `Overlap`.
    ///
    /// # Parameters
    /// - `other`: A reference to the `Overlap` instance to be merged.
    pub fn merge(&self, other: &Overlap) {
        // Add all local entities from `other` to `self`.
        other.local_entities.iter().for_each(|entry| {
            self.local_entities.insert(entry.key().clone(), ());
        });

        // Add all ghost entities from `other` to `self`.
        other.ghost_entities.iter().for_each(|entry| {
            self.ghost_entities.insert(entry.key().clone(), ());
        });
    }
}

/// The `Delta` struct manages transformation data for `MeshEntity` elements.  
/// It is specifically designed for distributed mesh environments to handle 
/// data transformations across overlapping regions.
pub struct Delta<T> {
    /// A thread-safe map that associates transformation data of type `T` 
    /// with `MeshEntity` objects.  
    /// 
    /// This map is used to store data for entities in overlapping regions,
    /// allowing for efficient updates, retrievals, and transformations.
    pub data: Arc<DashMap<MeshEntity, T>>,
}

impl<T> Delta<T> {
    /// Creates a new, empty `Delta`.
    ///
    /// # Returns
    /// - A new `Delta` instance with no data.
    pub fn new() -> Self {
        Delta {
            data: Arc::new(DashMap::new()),
        }
    }

    /// Associates transformation data with a specific `MeshEntity`.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` to associate with the data.
    /// - `value`: The transformation data of type `T`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves the transformation data associated with a specific `MeshEntity`.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` for which to retrieve data.
    ///
    /// # Returns
    /// - `Some(T)`: The associated transformation data if it exists.
    /// - `None`: If no data is associated with the given `MeshEntity`.
    ///
    /// # Requirements
    /// - The type `T` must implement the `Clone` trait to return a copy of the data.
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|entry| entry.clone())
    }

    /// Removes the transformation data associated with a specific `MeshEntity`.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` for which to remove data.
    ///
    /// # Returns
    /// - `Some(T)`: The removed transformation data if it existed.
    /// - `None`: If no data was associated with the given `MeshEntity`.
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        self.data.remove(entity).map(|(_, value)| value)
    }

    /// Checks whether a specific `MeshEntity` has associated transformation data.
    ///
    /// # Parameters
    /// - `entity`: A reference to the `MeshEntity` to check.
    ///
    /// # Returns
    /// - `true`: If transformation data exists for the `MeshEntity`.
    /// - `false`: Otherwise.
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        self.data.contains_key(entity)
    }

    /// Applies a function to all entities and their associated data in the delta.
    ///
    /// # Parameters
    /// - `func`: A mutable closure or function that takes a reference to a `MeshEntity`
    ///   and a reference to its associated data of type `T`.
    ///
    /// # Usage
    /// - This method allows for read-only operations or transformations over all
    ///   entries in the delta.
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        self.data.iter().for_each(|entry| func(entry.key(), entry.value()));
    }

    /// Merges another `Delta` instance into this one, combining data from both deltas.
    ///
    /// # Parameters
    /// - `other`: A reference to another `Delta` instance to merge.
    ///
    /// # Behavior
    /// - If a `MeshEntity` exists in both deltas, the data in the `other` delta
    ///   will overwrite the data in the current delta.
    ///
    /// # Requirements
    /// - The type `T` must implement the `Clone` trait to copy data from the other delta.
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

    /// Tests the addition and querying of local and ghost entities in the `Overlap` struct.
    #[test]
    fn test_overlap_local_and_ghost_entities() {
        let overlap = Overlap::new();
        let vertex_local = MeshEntity::Vertex(1); // A local entity
        let vertex_ghost = MeshEntity::Vertex(2); // A ghost entity

        // Add entities to their respective sets
        overlap.add_local_entity(vertex_local);
        overlap.add_ghost_entity(vertex_ghost);

        // Assert that the local entity is correctly identified as local
        assert!(overlap.is_local(&vertex_local));

        // Assert that the ghost entity is correctly identified as ghost
        assert!(overlap.is_ghost(&vertex_ghost));
    }

    /// Tests merging two `Overlap` instances and verifies the combined entities.
    #[test]
    fn test_overlap_merge() {
        let overlap1 = Overlap::new(); // First overlap
        let overlap2 = Overlap::new(); // Second overlap

        let vertex1 = MeshEntity::Vertex(1); // Local to overlap1
        let vertex2 = MeshEntity::Vertex(2); // Ghost in overlap1
        let vertex3 = MeshEntity::Vertex(3); // Local to overlap2

        // Add entities to respective overlaps
        overlap1.add_local_entity(vertex1);
        overlap1.add_ghost_entity(vertex2);
        overlap2.add_local_entity(vertex3);

        // Merge overlap2 into overlap1
        overlap1.merge(&overlap2);

        // Verify merged results
        assert!(overlap1.is_local(&vertex1));
        assert!(overlap1.is_ghost(&vertex2));
        assert!(overlap1.is_local(&vertex3));
        assert_eq!(overlap1.local_entities().len(), 2); // Two local entities in overlap1
    }

    /// Tests setting and retrieving data in the `Delta` struct.
    #[test]
    fn test_delta_set_and_get_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1); // Entity to associate with data

        // Set transformation data for the entity
        delta.set_data(vertex, 42);

        // Verify data retrieval
        assert_eq!(delta.get_data(&vertex), Some(42));
        assert!(delta.has_data(&vertex)); // Ensure the data is marked as present
    }

    /// Tests removing data from the `Delta` struct.
    #[test]
    fn test_delta_remove_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1); // Entity to associate with data

        // Set transformation data
        delta.set_data(vertex, 100);

        // Remove the data and verify removal
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex)); // Ensure the data is no longer present
    }

    /// Tests merging two `Delta` instances and verifies the combined transformation data.
    #[test]
    fn test_delta_merge() {
        let delta1 = Delta::new(); // First delta
        let delta2 = Delta::new(); // Second delta

        let vertex1 = MeshEntity::Vertex(1); // Associated with delta1
        let vertex2 = MeshEntity::Vertex(2); // Associated with delta2

        // Set data for entities in respective deltas
        delta1.set_data(vertex1, 10);
        delta2.set_data(vertex2, 20);

        // Merge delta2 into delta1
        delta1.merge(&delta2);

        // Verify merged data
        assert_eq!(delta1.get_data(&vertex1), Some(10));
        assert_eq!(delta1.get_data(&vertex2), Some(20));
    }
}
