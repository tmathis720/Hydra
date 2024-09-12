use crate::domain::Element;

pub struct PeriodicBoundary {
    pub elements: Vec<Element>,
    // Add a mapping for custom boundaries, where key maps to value
    pub boundary_map: Vec<(usize, usize)>,  // (source_index, target_index)
}

impl PeriodicBoundary {
    // Method to create a periodic boundary with a default mapping (first -> last)
    pub fn new(elements: Vec<Element>) -> Self {
        let boundary_map = if elements.len() > 1 {
            vec![(0, elements.len() - 1)] // Default boundary: first to last element
        } else {
            vec![]
        };

        Self {
            elements,
            boundary_map,
        }
    }

    // Method to apply periodic boundary using the custom mapping
    pub fn apply_boundary(&self, elements: &mut Vec<Element>) {
        for &(source_index, target_index) in &self.boundary_map {
            // Ensure valid indices before applying boundary
            if let (Some(source_element), Some(target_element)) = (
                elements.get(source_index).cloned(),
                elements.get_mut(target_index),
            ) {
                *target_element = source_element;
            } else {
                eprintln!(
                    "Invalid boundary mapping from index {} to index {}",
                    source_index, target_index
                );
            }
        }
    }

    // Method to add a custom boundary mapping
    pub fn add_boundary_mapping(&mut self, source_index: usize, target_index: usize) {
        if source_index < self.elements.len() && target_index < self.elements.len() {
            self.boundary_map.push((source_index, target_index));
        } else {
            eprintln!(
                "Cannot add mapping: Invalid indices {} -> {}",
                source_index, target_index
            );
        }
    }
}
