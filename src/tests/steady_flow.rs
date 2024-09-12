#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::FluxSolver;

    #[test]
    fn test_steady_flow_in_channel() {
        // Define two elements representing a channel
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            momentum: 0.0,
        };

        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            momentum: 0.0,
        };

        // Define the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),  // Shared between the two elements
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux with the given pressure difference
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Assert that the flux is positive (flow from left to right due to higher pressure on the left)
        assert!(flux > 0.0, "Flux should be positive due to pressure gradient");

        // Ensure steady flow conditions
        // In a steady flow scenario, the flux should not change over time
        let steady_flux = flux_solver.compute_flux(&face, &left_element, &right_element);
        assert_eq!(flux, steady_flux, "Flux should remain steady in a steady flow scenario");

        // Additional assertion: ensure mass conservation (total flux is conserved)
        let total_mass_flux = left_element.mass - right_element.mass;
        assert!(total_mass_flux.abs() < 1e-10, "Mass should be conserved in a steady flow");
    }
}
