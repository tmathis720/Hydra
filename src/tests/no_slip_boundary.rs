#[cfg(test)]
mod tests {
    use crate::domain::Element;
    use crate::boundary::NoSlipBoundary;
    use crate::solver::FluxSolver;

    #[test]
    fn test_no_slip_boundary_conditions() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create an element near the no-slip boundary (representing a wall)
        let mut element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: (0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 10.0,
            momentum: 2.0,
        };

        // Define a no-slip boundary condition
        let no_slip_boundary = NoSlipBoundary {};

        // Instantiate flux solver (not used in this case)
        let flux_solver = FluxSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Apply the no-slip boundary condition
            no_slip_boundary.apply_boundary(&mut element, dt);

            // Ensure the velocity remains zero at the boundary
            let flux = flux_solver.compute_flux_no_slip(&element);

            // Assert that no momentum is transferred at the no-slip boundary
            assert_eq!(flux, 0.0, "Flux should be zero at the no-slip boundary");
            assert_eq!(element.momentum, 0.0, "Momentum should remain zero at the no-slip boundary");
        }
    }
}
