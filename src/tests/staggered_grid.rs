#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::FluxSolver;

    #[test]
    fn test_flux_solver_staggered_grid() {
        // Create the left element (higher pressure)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1, 0],
            faces: vec![0, 1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 2.0, // Higher pressure in the left element
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
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2), // Nodes shared between left and right elements
            velocity: (0.0, 0.0), // Initial velocity is zero
            area: 1.0, // Example face area
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute flux with left element having higher pressure than the right element
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert!(flux > 0.0, "Flux should be positive since left element has higher pressure");

        // Modify the test to simulate the opposite case (right has higher pressure)
        let right_higher_pressure_element = Element {
            id: right_element.id,
            element_type: right_element.element_type,
            nodes: right_element.nodes.clone(),
            faces: right_element.faces.clone(),
            mass: right_element.mass,
            neighbor_ref: right_element.neighbor_ref.clone(),
            pressure: 3.0, // Higher pressure in the right element
        };

        // Recompute the flux with the right element having higher pressure
        let flux_reversed = flux_solver.compute_flux(&face, &right_higher_pressure_element, &left_element);
        assert!(flux_reversed < 0.0, "Flux should be negative since right element has higher pressure");

        // Edge case: when both elements have equal pressure, flux should be zero
        let equal_pressure_element = Element {
            id: left_element.id,
            element_type: left_element.element_type,
            nodes: left_element.nodes.clone(),
            faces: left_element.faces.clone(),
            mass: left_element.mass,
            neighbor_ref: left_element.neighbor_ref.clone(),
            pressure: 2.0, // Equal pressure on both sides
        };

        let flux_zero = flux_solver.compute_flux(&face, &equal_pressure_element, &equal_pressure_element);
        assert!(flux_zero.abs() < 1e-10, "Flux should be zero when both elements have equal pressure");
    }
}
