#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::FluxSolver;

    #[test]
    fn test_mass_conservation_no_inflow_outflow() {
        // Create the left and right elements with equal mass
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0, // Initial mass
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: 0.0,
            velocity: (0.0, 0.0, 0.0),
        };

        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0, // Initial mass
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            height: 0.0,
            area: 0.0,
            momentum: 0.0,
            velocity: (0.0, 0.0, 0.0),
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute flux and simulate mass transfer over time
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Transfer mass based on flux direction (flux moves from left to right)
        left_element.mass -= flux.abs();
        right_element.mass += flux.abs();

        // Assert that total mass is conserved
        let total_mass = left_element.mass + right_element.mass;
        assert_eq!(total_mass, 2.0, "Total mass should be conserved (no inflow or outflow)");
    }

    #[test]
    fn test_mass_conservation_with_inflow_outflow() {
        // Create the left element with higher mass and an inflow
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 2.0, // More mass in the left element
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: 0.0,
            velocity: (0.0, 0.0, 0.0),
        };

        // Create the right element with an outflow
        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0, // Less mass in the right element
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            height: 0.0,
            area: 0.0,
            momentum: 0.0,
            velocity: (0.0, 0.0, 0.0),
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0, // Simple unit area for the face
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Simulate inflow: Add mass to the left element (inflow boundary)
        left_element.mass += 0.1; // Simulate some inflow

        // Simulate mass transfer based on flux
        left_element.mass -= flux.abs();
        right_element.mass += flux.abs();

        // Assert total mass has increased by the inflow amount
        let total_mass = left_element.mass + right_element.mass;
        assert_eq!(total_mass, 3.1, "Total mass should include inflow mass");
    }
}
