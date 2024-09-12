#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_long_term_stability() {
        let dt = 0.01; // Time step
        let total_time = 100.0; // Long simulation time

        // Create the left and right elements
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0,
            height: 0.0,
            area: 0.0,
            momentum: 2.0,
            velocity: (0.0, 0.0, 0.0),
        };

        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0,
            height: 0.0,
            area: 0.0,
            momentum: 1.0,
            velocity: (0.0, 0.0, 0.0),
        };

        // Face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0,
        };

        // Instantiate the flux solver and semi-implicit solver
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _time in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Compute the flux at each time step
            let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

            // Use the semi-implicit solver to update the momentum
            left_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux * (left_element.momentum / left_element.mass),
                left_element.momentum,
                dt,
            );
            right_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux * (right_element.momentum / right_element.mass),
                right_element.momentum,
                dt,
            );

            // Apply clamping to ensure momentum does not go negative
            left_element.momentum = left_element.momentum.max(0.0);
            right_element.momentum = right_element.momentum.max(0.0);

            // Ensure no spurious numerical oscillations occur
            assert!(left_element.momentum > 0.0, "Momentum should remain positive in the long term");
            assert!(right_element.momentum > 0.0, "Momentum should remain positive in the long term");
        }
    }
}
