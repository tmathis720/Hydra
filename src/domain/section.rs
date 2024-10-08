use rustc_hash::FxHashMap;
use crate::domain::mesh_entity::MeshEntity;  // Assuming MeshEntity is defined in mesh_entity.rs

/// Section structure for associating data with mesh entities
pub struct Section<T> {
    pub data: FxHashMap<MeshEntity, T>,  // Map from entity to associated data
}

impl<T> Section<T> {
    /// Creates a new, empty Section
    pub fn new() -> Self {
        Section {
            data: FxHashMap::default(),
        }
    }

    /// Associate data with a mesh entity
    pub fn set_data(&mut self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);  // Insert or update the data
    }

    /// Restrict data to a given mesh entity (immutable access)
    pub fn restrict(&self, entity: &MeshEntity) -> Option<&T> {
        self.data.get(entity)
    }

    /// Restrict data to a given mesh entity (mutable access)
    pub fn restrict_mut(&mut self, entity: &MeshEntity) -> Option<&mut T> {
        self.data.get_mut(entity)
    }

    /// Update the data for a given mesh entity
    pub fn update_data(&mut self, entity: &MeshEntity, new_value: T) {
        self.data.insert(*entity, new_value);
    }

    /// Clear all data in the section
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get all mesh entities associated with this section
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.keys().cloned().collect()
    }

    /// Get all data stored in this section (immutable references)
    pub fn all_data(&self) -> Vec<&T> {
        self.data.values().collect()
    }

    /// Get mutable access to all data stored in this section
    pub fn all_data_mut(&mut self) -> Vec<&mut T> {
        self.data.values_mut().collect()
    }
}

// Unit tests for the Section structure
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_set_and_restrict_data() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        assert_eq!(section.restrict(&vertex), Some(&10));
        assert_eq!(section.restrict(&edge), Some(&20));
    }

    #[test]
    fn test_update_data() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 10);
        assert_eq!(section.restrict(&vertex), Some(&10));

        // Update the data
        section.update_data(&vertex, 15);
        assert_eq!(section.restrict(&vertex), Some(&15));

        // Try updating data for a non-existent entity (should insert it)
        let non_existent_entity = MeshEntity::Vertex(2);
        section.update_data(&non_existent_entity, 30);
        assert_eq!(section.restrict(&non_existent_entity), Some(&30));
    }

    #[test]
    fn test_restrict_mut() {
        let mut section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 5);
        if let Some(value) = section.restrict_mut(&vertex) {
            *value = 50;
        }
        assert_eq!(section.restrict(&vertex), Some(&50));
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
