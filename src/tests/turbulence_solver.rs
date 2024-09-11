#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::solver::TurbulenceSolver;

    #[test]
    fn test_turbulence_solver() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create an element with initial pressure and momentum
        let mut element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0,
            momentum: 2.0,
        };

        // Instantiate the turbulence solver with k and epsilon values
        let turbulence_solver = TurbulenceSolver {
            k: 1.0, // Turbulent kinetic energy
            epsilon: 0.5, // Turbulent dissipation rate
        };

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Apply turbulence to the element
            turbulence_solver.apply_turbulence(&mut element, dt);

            // Ensure the pressure decreases due to turbulence dissipation
            assert!(element.pressure <= 10.0, "Pressure should decrease due to turbulence");
            assert!(element.momentum <= 2.0, "Momentum should decrease due to turbulence dissipation");

            // Ensure the pressure and momentum remain positive
            assert!(element.pressure >= 0.0, "Pressure should remain non-negative");
            assert!(element.momentum >= 0.0, "Momentum should remain non-negative");
        }
    }
}
