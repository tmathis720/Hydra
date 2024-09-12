#[cfg(test)]
mod tests {
    use crate::domain::Element;
    use crate::boundary::FreeSurfaceBoundary;
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_free_surface_boundary_conditions() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create an element near the free surface
        let mut element = Element {
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

        // Define a free surface boundary condition
        let free_surface = FreeSurfaceBoundary { pressure_at_surface: 1.0 };

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Apply the free surface boundary condition
            free_surface.apply_boundary(&mut element, dt);

            // Compute flux between the element and the free surface (pressure difference drives flux)
            let flux = flux_solver.compute_flux_free_surface(&element, free_surface.pressure_at_surface);

            // Update momentum with semi-implicit solver
            element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux * (element.momentum / element.mass),
                element.momentum,
                dt,
            );

            // Ensure the free surface is interacting correctly with the pressure
            assert!(element.pressure > free_surface.pressure_at_surface, "Pressure should remain higher than free surface");
            assert!(element.momentum > 0.0, "Momentum should remain positive");
        }
    }
}
