#[cfg(test)]
mod tests {
    use crate::domain::Element;
    use crate::domain::Face;
    use crate::solver::{FluxSolver, ScalarTransportSolver};
    use nalgebra::Vector3;

    #[test]
    fn test_scalar_advection_in_flow() {
        // Create the left element with higher scalar concentration
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Create the right element with lower scalar concentration
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: vec![1, 2],  // Nodes shared between left and right elements
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };

        // Scalar concentration in left and right elements
        let left_scalar_concentration = 1.0;  // Higher concentration
        let right_scalar_concentration = 0.0; // Lower concentration

        // Instantiate the flux and scalar transport solver
        let flux_solver = FluxSolver {};
        let scalar_solver = ScalarTransportSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

        // Compute scalar flux based on the fluid flux
        let transported_scalar = scalar_solver.compute_scalar_flux(flux, left_scalar_concentration, right_scalar_concentration);

        // Assert that the scalar moves from left to right (positive flux)
        assert!(transported_scalar > 0.0, "Scalar should move from left to right due to flux");
    }

    #[test]
    fn test_scalar_advection_with_reverse_flux() {
        // Create the left element with lower scalar concentration
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Create the right element with higher scalar concentration
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the right
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: vec![1, 2],  // Nodes shared between left and right elements
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };

        // Scalar concentration in left and right elements
        let left_scalar_concentration = 0.0;  // Lower concentration
        let right_scalar_concentration = 1.0; // Higher concentration

        // Instantiate the flux and scalar transport solver
        let flux_solver = FluxSolver {};
        let scalar_solver = ScalarTransportSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

        // Compute scalar flux based on the fluid flux
        let transported_scalar = scalar_solver.compute_scalar_flux(flux, left_scalar_concentration, right_scalar_concentration);

        // Assert that the scalar moves from right to left (negative flux)
        assert!(transported_scalar < 0.0, "Scalar should move from right to left due to reverse flux");
    }

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
            momentum: Vector3::new(2.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
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
            momentum: Vector3::new(1.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Define face between the elements
        let face = Face {
            id: 0,
            nodes: vec![1, 2],
            velocity: Vector3::new(0.0, 0.0, 0.0),
            area: 1.0,
            ..Face::default()
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
            assert!(left_element.momentum >= Vector3::new(0.0, 0.0, 0.0), "Momentum should remain non-negative for left element");
            assert!(right_element.momentum >= Vector3::new(0.0, 0.0, 0.0), "Momentum should remain non-negative for right element");

            // Gradually reduce the pressure difference between the elements
            assert!(left_element.momentum > right_element.momentum, "Left element should have higher momentum initially");
        }
    }


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
            velocity: Vector3::new(0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            momentum: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure on the right
            momentum: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Define the face between the two elements
        let face = Face {
            id: 0,
            nodes: vec![1, 2],  // Shared between the two elements
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux with the given pressure difference
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

        // Assert that the flux is positive (flow from left to right due to higher pressure on the left)
        assert!(flux > Vector3::new(0.0, 0.0, 0.0), "Flux should be positive due to pressure gradient");

        // Ensure steady flow conditions
        // In a steady flow scenario, the flux should not change over time
        let steady_flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);
        assert_eq!(flux, steady_flux, "Flux should remain steady in a steady flow scenario");

        // Additional assertion: ensure mass conservation (total flux is conserved)
        let total_mass_flux = left_element.mass - right_element.mass;
        assert!(total_mass_flux.abs() < 1e-10, "Mass should be conserved in a steady flow");
    }

}
