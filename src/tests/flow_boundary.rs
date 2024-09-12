#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::boundary::{Inflow, Outflow};
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_inflow_outflow_boundary_conditions() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create two elements (left with inflow, right with outflow)
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

        // Define the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),  // Example node ids the face connects
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Example face area
        };

        // Define flow boundary conditions (inflow on left, outflow on right)
        let inflow = Inflow { rate: 0.1 };  // Add mass/momentum on left
        let outflow = Outflow { rate: 0.1 };  // Remove mass/momentum on right

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Apply inflow boundary condition
            inflow.apply_boundary(&mut left_element, dt);

            // Apply outflow boundary condition
            outflow.apply_boundary(&mut right_element, dt);

            // Compute flux between the two elements using the face
            let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

            // Update momentum with semi-implicit solver
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

            // Ensure mass and momentum conservation with inflow/outflow
            assert!(left_element.mass > 1.0, "Mass should increase with inflow");
            assert!(right_element.mass < 1.0, "Mass should decrease with outflow");
            assert!(left_element.momentum > 0.0, "Momentum should remain positive with inflow");
            assert!(right_element.momentum > 0.0, "Momentum should remain positive after outflow");
        }
    }
}
