use crate::domain::Element;
use nalgebra::Vector3;

pub struct FlowField {
    pub elements: Vec<Element>,
    pub initial_mass: f64,
}

impl FlowField {
    /// Create a new flow field and compute the initial mass
    pub fn new(elements: Vec<Element>) -> Self {
        let initial_mass: f64 = elements.iter().map(|e| e.calculate_mass()).sum();
        Self {
            elements,
            initial_mass,
        }
    }

    /// Compute the mass of a given element
    pub fn compute_mass(&self, element: &Element) -> f64 {
        element.mass
    }

    /// Compute the density of a given element
    pub fn compute_density(&self, element: &Element) -> f64 {
        if element.area > 0.0 {
            element.mass / element.area
        } else {
            0.0
        }
    }

    /// Check if mass conservation holds, comparing the current mass with the initial mass
    pub fn check_mass_conservation(&self) -> bool {
        let current_mass: f64 = self.elements.iter().map(|e| e.mass).sum();
        let tolerance = 1e-6;
        (current_mass - self.initial_mass).abs() < tolerance
    }

    /// Placeholder function to get the surface velocity (to be refined for surface flow conditions)
    pub fn get_surface_velocity(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 0.0)
    }

    /// Return the surface pressure (assumed atmospheric for now)
    pub fn get_surface_pressure(&self) -> f64 {
        101325.0  // Standard atmospheric pressure in Pascals
    }

    /// Return a default inflow velocity vector (could be refined based on domain configuration)
    pub fn get_inflow_velocity(&self) -> Vector3<f64> {
        Vector3::new(1.0, 0.0, 0.0)  // Example: inflow velocity in x-direction
    }

    /// Return a default outflow velocity vector (could be refined based on domain configuration)
    pub fn get_outflow_velocity(&self) -> Vector3<f64> {
        Vector3::new(0.5, 0.0, 0.0)  // Example: outflow velocity in x-direction
    }

    /// Return a fixed inflow mass rate (could be dynamic depending on domain conditions)
    pub fn inflow_mass_rate(&self) -> f64 {
        1.0
    }

    /// Return a fixed outflow mass rate (could be dynamic depending on domain conditions)
    pub fn outflow_mass_rate(&self) -> f64 {
        0.8
    }

    /// Return the corresponding periodic element (for periodic boundary conditions)
    pub fn get_periodic_element<'a>(&'a self, element: &'a Element) -> &'a Element {
        // Logic to find the periodic counterpart of the current element, here just a placeholder
        element
    }

    /// Compute the average nearby pressure for an element based on its neighbors
    pub fn get_nearby_pressure(&self, element: &Element) -> f64 {
        let mut total_pressure = 0.0;
        let mut count = 0;

        for neighbor in &self.elements {
            if neighbor.id != element.id {
                total_pressure += neighbor.pressure;
                count += 1;
            }
        }

        if count > 0 {
            total_pressure / count as f64
        } else {
            0.0  // Handle the case where no neighbors are found
        }
    }
}

/// Manual implementation of the `Default` trait for `FlowField`
impl Default for FlowField {
    fn default() -> Self {
        Self {
            elements: Vec::new(),  // Empty vector for elements
            initial_mass: 0.0,     // Default mass as 0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Element;
    use nalgebra::Vector3;

    #[test]
    fn test_mass_conservation() {
        let element1 = Element {
            id: 1,
            mass: 10.0,
            ..Default::default()
        };
        let element2 = Element {
            id: 2,
            mass: 5.0,
            ..Default::default()
        };

        let flow_field = FlowField::new(vec![element1.clone(), element2.clone()]);
        assert!(flow_field.check_mass_conservation());

        // Modify mass and check again
        let mut element1_modified = element1.clone();
        element1_modified.mass += 1.0;  // Artificially modify the mass

        let flow_field_modified = FlowField::new(vec![element1_modified, element2]);
        assert!(!flow_field_modified.check_mass_conservation());
    }

    #[test]
    fn test_inflow_outflow_velocity() {
        let flow_field = FlowField::new(vec![]);

        let inflow_velocity = flow_field.get_inflow_velocity();
        let outflow_velocity = flow_field.get_outflow_velocity();

        assert_eq!(inflow_velocity, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(outflow_velocity, Vector3::new(0.5, 0.0, 0.0));
    }
}
