// src/domain/dm_field.rs

use crate::domain::Section;

/// Field for creating prognostic variables on different mesh elements
#[derive(Debug)]
pub struct Field {
    pub data: Vec<f64>,          // Data array storing values for all mesh points
    pub section: Section,        // Section describing the layout of the data
}

impl Field {
    // Create a new field with a section for data layout
    pub fn new(num_fields: usize, num_points: usize) -> Self {
        Field {
            data: vec![0.0; num_fields * num_points],
            section: Section::new(num_fields),
        }
    }

    // Set values for a DPoint based on its section
    pub fn set_values_for_point(&mut self, point_id: usize, values: &[f64]) {
        if let Some(offset) = self.section.get_offset(point_id) {
            let size = self.section.get_size(point_id).unwrap();
            for (i, &val) in values.iter().enumerate() {
                if i < size {
                    self.data[offset + i] = val;
                }
            }
        } else {
            panic!("DPoint ID {} not found in the section", point_id);
        }
    }

    // Retrieve values for a DPoint based on its section
    pub fn get_values_for_point(&self, point_id: usize) -> Option<Vec<f64>> {
        if let Some(offset) = self.section.get_offset(point_id) {
            let size = self.section.get_size(point_id).unwrap();
            Some(self.data[offset..offset + size].to_vec())
        } else {
            None
        }
    }

    // Initialize the field with section layout
    pub fn initialize_section(&mut self, point_id: usize, size: usize) {
        self.section.set_section(point_id, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_field_with_sections() {
        let mut field = Field::new(2, 3);  // Two fields, three DPoints
        
        // Set up sections for points
        field.initialize_section(0, 2);  // DPoint 0 gets 2 values
        field.initialize_section(1, 2);  // DPoint 1 gets 2 values
        field.initialize_section(2, 2);  // DPoint 2 gets 2 values
        
        // Set values for DPoint 0 and check
        field.set_values_for_point(0, &[1.0, 2.0]);
        assert_eq!(field.get_values_for_point(0), Some(vec![1.0, 2.0]));

        // Set values for DPoint 1 and check
        field.set_values_for_point(1, &[3.0, 4.0]);
        assert_eq!(field.get_values_for_point(1), Some(vec![3.0, 4.0]));

        // Check that DPoint 2 still has the default values (zeros)
        assert_eq!(field.get_values_for_point(2), Some(vec![0.0, 0.0]));
    }

    #[test]
    #[should_panic]
    fn test_set_value_mismatched_size() {
        let mut field = Field::new(2, 3);
        
        // Set up a section for DPoint 0
        field.initialize_section(0, 2);
        
        // Try setting the values with the wrong size, which should panic
        field.set_values_for_point(0, &[1.0]);  // Should panic because size mismatch
    }

    #[test]
    fn test_set_and_get_value_for_point() {
        let mut field = Field::new(2, 3);
        
        // Set up sections and assign values
        field.initialize_section(0, 2);
        field.set_values_for_point(0, &[1.0, 2.0]);

        assert_eq!(field.get_values_for_point(0), Some(vec![1.0, 2.0]));
    }
}
