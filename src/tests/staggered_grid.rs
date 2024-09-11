#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::FluxSolver;

    // Helper function to create a basic face between two elements
    fn create_face() -> Face {
        Face {
            id: 0,
            nodes: (1, 2), // Nodes shared between left and right elements
            velocity: (0.0, 0.0), // Initial velocity is zero
            area: 1.0, // Example face area
        }
    }

    #[test]
    fn test_flux_positive_when_left_pressure_higher() {
        // Create the left element (higher pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 2.0, // Higher pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (lower pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 1.0, // Lower pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect positive flux (flow from left to right)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert!(flux > 0.0, "Flux should be positive since left element has higher pressure");
    }

    #[test]
    fn test_flux_negative_when_right_pressure_higher() {
        // Create the left element (lower pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 1.0, // Lower pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (higher pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 3.0, // Higher pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect negative flux (flow from right to left)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert!(flux < 0.0, "Flux should be negative since right element has higher pressure");
    }

    #[test]
    fn test_flux_zero_when_pressures_equal() {
        // Create the left element (equal pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 2.0, // Same pressure in the left element
            momentum: 0.0,
        };

        // Create the right element (equal pressure)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2, 0],
            faces: vec![1, 2],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 2.0, // Same pressure in the right element
            momentum: 0.0,
        };

        let face = create_face();
        let flux_solver = FluxSolver {};

        // Compute flux and expect zero flux (no flow)
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert_eq!(flux, 0.0, "Flux should be zero when both elements have equal pressure");
    }
}
