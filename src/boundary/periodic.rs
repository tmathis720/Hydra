use crate::domain::Element;

pub struct PeriodicBoundary {
    pub elements: Vec<Element>,
    // Custom boundary mapping: (source_index, target_index)
    pub boundary_map: Vec<(usize, usize)>,  
}

impl PeriodicBoundary {
    /// Constructor to create a periodic boundary with a default mapping (first -> last)
    pub fn new(elements: Vec<Element>) -> Self {
        let boundary_map = if elements.len() > 1 {
            vec![(0, elements.len() - 1)]  // Default: first to last element
        } else {
            vec![]
        };

        Self {
            elements,
            boundary_map,
        }
    }

    /// Apply the periodic boundary using the custom mapping.
    /// This applies periodic conditions by copying properties (velocity, momentum, pressure) from source to target elements.
    pub fn apply_boundary(&self, elements: &mut Vec<Element>) {
        for &(source_index, target_index) in &self.boundary_map {
            // Ensure valid indices before applying boundary
            if let (Some(source_element), Some(target_element)) = (
                elements.get(source_index).cloned(),
                elements.get_mut(target_index),
            ) {
                // Transfer periodic boundary conditions: velocity, momentum, pressure, etc.
                target_element.pressure = source_element.pressure;
                target_element.velocity = source_element.velocity;
                target_element.momentum = source_element.momentum;
            } else {
                eprintln!(
                    "Invalid boundary mapping from index {} to index {}",
                    source_index, target_index
                );
            }
        }
    }

    /// Add a custom boundary mapping (source_index -> target_index).
    /// This allows for complex periodic boundaries by defining where properties should be copied.
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_periodic_boundary_application() {
        // Create test elements with different values
        let mut elements = vec![
            Element {
                pressure: 100.0,
                velocity: Vector3::new(1.0, 0.0, 0.0),
                momentum: Vector3::new(10.0, 0.0, 0.0),
                ..Default::default()  // Assuming default implementation for other fields
            },
            Element {
                pressure: 50.0,
                velocity: Vector3::new(0.0, 1.0, 0.0),
                momentum: Vector3::new(5.0, 1.0, 0.0),
                ..Default::default()
            },
        ];

        let mut _periodic_boundary = PeriodicBoundary::new(elements.clone());

        // Apply default periodic boundary (first -> last)
        _periodic_boundary.apply_boundary(&mut elements);

        // Ensure that the last element has copied properties from the first
        assert_eq!(elements[1].pressure, elements[0].pressure);
        assert_eq!(elements[1].velocity, elements[0].velocity);
        assert_eq!(elements[1].momentum, elements[0].momentum);
    }

    #[test]
    fn test_custom_boundary_mapping() {
        let mut elements = vec![
            Element {
                pressure: 100.0,
                velocity: Vector3::new(1.0, 0.0, 0.0),
                momentum: Vector3::new(10.0, 0.0, 0.0),
                ..Default::default()
            },
            Element {
                pressure: 50.0,
                velocity: Vector3::new(0.0, 1.0, 0.0),
                momentum: Vector3::new(5.0, 1.0, 0.0),
                ..Default::default()
            },
            Element {
                pressure: 30.0,
                velocity: Vector3::new(0.0, 0.0, 1.0),
                momentum: Vector3::new(3.0, 0.0, 1.0),
                ..Default::default()
            },
        ];

        let mut periodic_boundary = PeriodicBoundary::new(elements.clone());

        // Add custom mapping: map element 0 to element 2
        periodic_boundary.add_boundary_mapping(0, 2);
        periodic_boundary.apply_boundary(&mut elements);

        // Ensure that element 2 has copied properties from element 0
        assert_eq!(elements[2].pressure, elements[0].pressure);
        assert_eq!(elements[2].velocity, elements[0].velocity);
        assert_eq!(elements[2].momentum, elements[0].momentum);
    }

    #[test]
    fn test_invalid_boundary_mapping() {
        let mut elements = vec![
            Element {
                pressure: 100.0,
                velocity: Vector3::new(1.0, 0.0, 0.0),
                momentum: Vector3::new(10.0, 0.0, 0.0),
                ..Default::default()
            },
        ];

        let mut periodic_boundary = PeriodicBoundary::new(elements.clone());

        // Try to add an invalid mapping (out of bounds)
        periodic_boundary.add_boundary_mapping(0, 1);
        periodic_boundary.apply_boundary(&mut elements);

        // Since the boundary mapping is invalid, the test should print an error, but nothing else should change
        assert_eq!(elements[0].pressure, 100.0);
        assert_eq!(elements[0].velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(elements[0].momentum, Vector3::new(10.0, 0.0, 0.0));
    }
}
