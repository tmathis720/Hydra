use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

pub struct Section<T> {
    pub data: Arc<RwLock<FxHashMap<MeshEntity, T>>>,
}

impl<T> Section<T> {
    pub fn new() -> Self {
        Section {
            data: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    pub fn set_data(&self, entity: MeshEntity, value: T) {
        let mut data = self.data.write().unwrap();
        data.insert(entity, value);
    }

    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> 
    where
        T: Clone,
    {
        let data = self.data.read().unwrap();
        data.get(entity).cloned()
    }

    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
        T: Send + Sync,
    {
        let mut data = self.data.write().unwrap();
        data.par_iter_mut().for_each(|(_, v)| update_fn(v));
    }


    /// Restrict data to a given mesh entity (mutable access)
    pub fn restrict_mut(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        let mut data = self.data.write().unwrap();
        data.get(entity).cloned()
    }

    /// Update the data for a given mesh entity
    pub fn update_data(&self, entity: &MeshEntity, new_value: T) {
        let mut data = self.data.write().unwrap();
        data.insert(*entity, new_value);
    }

    /// Clear all data in the section
    pub fn clear(&self) {
        let mut data = self.data.write().unwrap();
        data.clear();
    }

    /// Get all mesh entities associated with this section
    pub fn entities(&self) -> Vec<MeshEntity> {
        let data = self.data.read().unwrap();
        data.keys().cloned().collect()
    }

    /// Get all data stored in this section (immutable references)
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        let data = self.data.read().unwrap();
        data.values().cloned().collect()
    }

    /// Get mutable access to all data stored in this section
    pub fn all_data_mut(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut data = self.data.write().unwrap();
        data.values().cloned().collect()
    }
}

// Unit tests for the Section structure
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_set_and_restrict_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        section.set_data(vertex, 42);
        assert_eq!(section.restrict(&vertex), Some(42));
    }

    #[test]
    fn test_update_data() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 10);
        assert_eq!(section.restrict(&vertex), Some(10));

        // Update the data
        section.update_data(&vertex, 15);
        assert_eq!(section.restrict(&vertex), Some(15));

        // Try updating data for a non-existent entity (should insert it)
        let non_existent_entity = MeshEntity::Vertex(2);
        section.update_data(&non_existent_entity, 30);
        assert_eq!(section.restrict(&non_existent_entity), Some(30));
    }

    #[test]
    fn test_restrict_mut() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 5);
        if let Some(mut value) = section.restrict_mut(&vertex) {
            value = 50;
            section.set_data(vertex, value);
        }
        assert_eq!(section.restrict(&vertex), Some(50));
    }

    #[test]
    fn test_get_all_entities() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let entities = section.entities();
        assert!(entities.contains(&vertex));
        assert!(entities.contains(&edge));
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_get_all_data() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let all_data = section.all_data();
        assert_eq!(all_data.len(), 2);
        assert!(all_data.contains(&&10));
        assert!(all_data.contains(&&20));
    }
}
