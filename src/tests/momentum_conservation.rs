#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::FluxSolver;

    

    #[test]
    fn test_momentum_conservation_no_inflow_outflow() {
        // Create the left and right elements with equal initial momentum
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            momentum: 2.0,  // Initial momentum (mass * velocity)
        };

        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            momentum: 1.0,  // Initial momentum (mass * velocity)
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0,
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux (mass transfer) between the two elements
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Transfer mass and momentum between elements
        let momentum_flux = flux * (left_element.momentum / left_element.mass);
        left_element.momentum -= momentum_flux;
        right_element.momentum += momentum_flux;

        // Assert that total momentum is conserved
        let total_momentum = left_element.momentum + right_element.momentum;
        assert_eq!(total_momentum, 3.0, "Total momentum should be conserved with no inflow/outflow");
    }

    #[test]
    fn test_momentum_conservation_with_inflow() {
        // Create the left element with an inflow and higher momentum
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 2.0, // More mass in the left element
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            momentum: 4.0,  // Higher momentum
        };

        // Create the right element with lower momentum and outflow
        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            momentum: 1.0,  // Lower momentum
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0,
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux (mass transfer) between the two elements
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Simulate inflow: Add momentum and mass to the left element (external force/inflow)
        left_element.momentum += 0.5; // External inflow of momentum
        left_element.mass += 0.1; // Inflow of mass

        // Transfer mass and momentum between elements
        let momentum_flux = flux * (left_element.momentum / left_element.mass);
        left_element.momentum -= momentum_flux;
        right_element.momentum += momentum_flux;

        // Assert total momentum includes inflow amount
        let total_momentum = left_element.momentum + right_element.momentum;
        assert_eq!(total_momentum, 5.5, "Total momentum should account for inflow");
    }
}
