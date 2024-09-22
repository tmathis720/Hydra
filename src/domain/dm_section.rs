// src/domain/dm_section.rs

/// In PETSc, a PetscSection is used to manage the 
/// data layout across different mesh dpoints, specifying
///  which parts of an array (or Vec) correspond to 
/// particular sections of the mesh, such as vertices, 
/// edges, or cells. This is important for working 
/// efficiently with structured or unstructured grids.
/// 
use std::collections::HashMap;

#[derive(Debug)]
pub struct Section {
    pub offsets: HashMap<usize, usize>,  // Maps dpoint ID to the start index of its data
    pub sizes: HashMap<usize, usize>,    // Maps dpoint ID to the size of the data block
    pub num_fields: usize,               // Number of fields associated with each dpoint
    pub global_size: usize,              // Total size of the global data
}

impl Section {
    // Create a new, empty Section
    pub fn new(num_fields: usize) -> Self {
        Section {
            offsets: HashMap::new(),
            sizes: HashMap::new(),
            num_fields,
            global_size: 0,
        }
    }

    // Define a section for a given dpoint with a specific size and starting offset
    pub fn set_section(&mut self, dpoint_id: usize, size: usize) {
        let offset = self.global_size;
        self.offsets.insert(dpoint_id, offset);
        self.sizes.insert(dpoint_id, size);
        self.global_size += size * self.num_fields;  // Adjust global size for multiple fields
    }

    // Retrieve the offset for a given dpoint ID
    pub fn get_offset(&self, dpoint_id: usize) -> Option<usize> {
        self.offsets.get(&dpoint_id).copied()
    }

    // Retrieve the size for a given dpoint ID
    pub fn get_size(&self, dpoint_id: usize) -> Option<usize> {
        self.sizes.get(&dpoint_id).copied()
    }

    // Get total size of the global data across all dpoints
    pub fn get_global_size(&self) -> usize {
        self.global_size
    }

    // Print details about the section for debugging purposes
    pub fn debug_section(&self) {
        println!("Global data size: {}", self.global_size);
        for (dpoint_id, offset) in &self.offsets {
            let size = self.sizes.get(dpoint_id).unwrap_or(&0);
            println!("DPoint {} -> Offset: {}, Size: {}", dpoint_id, offset, size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_section() {
        // Create a Section with 2 fields (e.g., velocity components)
        let section = Section::new(2);
        
        // Ensure that the section is initialized correctly
        assert_eq!(section.num_fields, 2);
        assert_eq!(section.get_global_size(), 0);  // Initially, the global size should be 0
    }

    #[test]
    fn test_set_section() {
        let mut section = Section::new(3);  // 3 fields for each DPoint

        // Set sections for a few DPoints
        section.set_section(0, 2);  // DPoint 0 has 2 data entries (size = 2)
        section.set_section(1, 3);  // DPoint 1 has 3 data entries (size = 3)
        
        // Check offsets and sizes for the DPoints
        assert_eq!(section.get_offset(0), Some(0));       // DPoint 0 starts at offset 0
        assert_eq!(section.get_offset(1), Some(6));       // DPoint 1 starts at offset 6 (2 * 3 fields for DPoint 0)
        
        assert_eq!(section.get_size(0), Some(2));         // DPoint 0 has size 2
        assert_eq!(section.get_size(1), Some(3));         // DPoint 1 has size 3
        
        // Global size should be the total size (2 * 3 fields for DPoint 0 + 3 * 3 fields for DPoint 1)
        assert_eq!(section.get_global_size(), 15);  // Global size = 2*3 + 3*3 = 15
    }

    #[test]
    fn test_get_section_info() {
        let mut section = Section::new(1);  // 1 field per DPoint

        // Set sections for a few DPoints
        section.set_section(0, 2);
        section.set_section(1, 1);

        // Verify the information for DPoint 0
        assert_eq!(section.get_offset(0), Some(0));
        assert_eq!(section.get_size(0), Some(2));

        // Verify the information for DPoint 1
        assert_eq!(section.get_offset(1), Some(2));  // Starts after DPoint 0's data
        assert_eq!(section.get_size(1), Some(1));

        // Check the global size
        assert_eq!(section.get_global_size(), 3);  // Total size is 2 + 1
    }

    #[test]
    fn test_debug_section() {
        let mut section = Section::new(2);

        // Set some sections for debugging purposes
        section.set_section(0, 2);  // DPoint 0 has 2 entries
        section.set_section(1, 1);  // DPoint 1 has 1 entry

        // Output the section layout for debugging
        section.debug_section();

        // Expected output:
        // Global data size: 6
        // DPoint 0 -> Offset: 0, Size: 2
        // DPoint 1 -> Offset: 4, Size: 1
    }

    #[test]
    fn test_empty_section() {
        // Ensure the Section initializes correctly with no data
        let section = Section::new(2);
        assert_eq!(section.get_global_size(), 0);
        assert!(section.get_offset(0).is_none());  // No sections set yet
        assert!(section.get_size(0).is_none());
    }
}
