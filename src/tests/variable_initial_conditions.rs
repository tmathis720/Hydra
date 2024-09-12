#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_variable_initial_conditions() {
        let dt = 0.01;
        let total_time = 50.0; // Shorter simulation for initial conditions

        // Create three elements with varying initial conditions
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 15.0,  // High pressure
            momentum: 3.0,   // High momentum
        };

        let mut middle_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 10.0,  // Moderate pressure
            momentum: 2.0,   // Moderate momentum
        };

        let mut right_element = Element {
            id: 2,
            element_type: 2,
            nodes: vec![2, 3],
            faces: vec![2],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 5.0,  // Low pressure
            momentum: 1.0,  // Low momentum
        };

        // Faces between the elements
        let left_face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0,
        };
        let right_face = Face {
            id: 1,
            nodes: (2, 3),
            velocity: (0.0, 0.0),
            area: 1.0,
        };

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _time in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Compute fluxes between elements
            let flux_left = flux_solver.compute_flux(&left_face, &left_element, &middle_element);
            let flux_right = flux_solver.compute_flux(&right_face, &middle_element, &right_element);

            // Update momenta using semi-implicit solver
            left_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_left * (left_element.momentum / left_element.mass),
                left_element.momentum,
                dt,
            );
            middle_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux_left * (middle_element.momentum / middle_element.mass),
                middle_element.momentum,
                dt,
            );
            middle_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_right * (middle_element.momentum / middle_element.mass),
                middle_element.momentum,
                dt,
            );
            right_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux_right * (right_element.momentum / right_element.mass),
                right_element.momentum,
                dt,
            );

            // Ensure momentum remains positive
            assert!(left_element.momentum > 0.0, "Momentum should remain positive in the left element");
            assert!(middle_element.momentum > 0.0, "Momentum should remain positive in the middle element");
            assert!(right_element.momentum > 0.0, "Momentum should remain positive in the right element");
        }
    }
}
