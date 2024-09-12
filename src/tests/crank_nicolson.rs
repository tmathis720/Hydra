#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::CrankNicolsonSolver;

    #[test]
    fn test_crank_nicolson_solver() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create two elements with different initial pressures and momentum
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

        // Define face between the elements
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0),
            area: 1.0,
        };

        // Instantiate the Crank-Nicolson solver
        let crank_nicolson_solver = CrankNicolsonSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Calculate flux (manually computed for now)
            let pressure_diff = left_element.pressure - right_element.pressure;
            let flux = pressure_diff * face.area;

            // Update momentum using Crank-Nicolson solver
            left_element.momentum = crank_nicolson_solver.crank_nicolson_update(flux, left_element.momentum, dt);
            right_element.momentum = crank_nicolson_solver.crank_nicolson_update(-flux, right_element.momentum, dt);

            // Ensure the momentum remains positive across both elements
            assert!(left_element.momentum >= 0.0, "Momentum should remain non-negative for left element");
            assert!(right_element.momentum >= 0.0, "Momentum should remain non-negative for right element");

            // Gradually reduce the pressure difference between the elements
            assert!(left_element.momentum > right_element.momentum, "Left element should have higher momentum initially");
        }
    }
}
