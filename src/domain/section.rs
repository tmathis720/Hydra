use std::collections::HashMap;
use crate::domain::mesh_entity::MeshEntity;  // Assuming MeshEntity is defined in mesh_entity.rs

/// Section structure for associating data with mesh entities
pub struct Section<T> {
    pub data: Vec<T>,  // Contiguous storage for associated data
    pub offsets: HashMap<MeshEntity, usize>,  // Map points to offsets in the data
}

impl<T> Section<T> {
    /// Creates a new, empty Section
    pub fn new() -> Self {
        Section {
            data: Vec::new(),
            offsets: HashMap::new(),
        }
    }

    /// Associate data with a mesh entity
    pub fn set_data(&mut self, entity: MeshEntity, value: T) {
        let offset = self.data.len();
        self.offsets.insert(entity, offset);  // Map the entity to its position in the data
        self.data.push(value);  // Store the associated data in contiguous storage
    }

    /// Restrict data to a given mesh entity (immutable access)
    pub fn restrict(&self, entity: &MeshEntity) -> Option<&T> {
        self.offsets.get(entity).map(|&offset| &self.data[offset])
    }

    /// Restrict data to a given mesh entity (mutable access)
    // Restrict data to a given mesh entity (mutable access)
    pub fn restrict_mut(&mut self, entity: &MeshEntity) -> Option<&mut T> {
        // First, we retrieve the offset immutably from self.offsets
        let offset = self.offsets.get(entity)?;
        
        // Then, we can use the offset to access self.data mutably
        Some(&mut self.data[*offset])
    }

    /// Update the data for a given mesh entity
    pub fn update_data(&mut self, entity: &MeshEntity, new_value: T) -> Result<(), String> {
        if let Some(&offset) = self.offsets.get(entity) {
            self.data[offset] = new_value;  // Update the data at the entity's offset
            Ok(())
        } else {
            Err(format!("Entity {:?} not found in section.", entity))
        }
    }

    /// Get all mesh entities associated with this section
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.offsets.keys().cloned().collect()
    }

    /// Get all data stored in this section
    pub fn all_data(&self) -> &Vec<T> {
        &self.data
    }

    /// Get mutable access to all data stored in this section
    pub fn all_data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
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
        section.update_data(&vertex, 15).unwrap();
        assert_eq!(section.restrict(&vertex), Some(&15));

        // Try updating data for a non-existent entity
        let non_existent_entity = MeshEntity::Vertex(2);
        assert!(section.update_data(&non_existent_entity, 30).is_err());
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
        assert!(all_data.contains(&10));
        assert!(all_data.contains(&20));
    }
}
