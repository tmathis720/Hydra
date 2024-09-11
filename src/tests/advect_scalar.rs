#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::{FluxSolver, ScalarTransportSolver};

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
            momentum: 0.0,
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
            momentum: 0.0,
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),  // Nodes shared between left and right elements
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
        };

        // Scalar concentration in left and right elements
        let left_scalar_concentration = 1.0;  // Higher concentration
        let right_scalar_concentration = 0.0; // Lower concentration

        // Instantiate the flux and scalar transport solver
        let flux_solver = FluxSolver {};
        let scalar_solver = ScalarTransportSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

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
            momentum: 0.0,
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
            momentum: 0.0,
        };

        // Create the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),  // Nodes shared between left and right elements
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
        };

        // Scalar concentration in left and right elements
        let left_scalar_concentration = 0.0;  // Lower concentration
        let right_scalar_concentration = 1.0; // Higher concentration

        // Instantiate the flux and scalar transport solver
        let flux_solver = FluxSolver {};
        let scalar_solver = ScalarTransportSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Compute scalar flux based on the fluid flux
        let transported_scalar = scalar_solver.compute_scalar_flux(flux, left_scalar_concentration, right_scalar_concentration);

        // Assert that the scalar moves from right to left (negative flux)
        assert!(transported_scalar < 0.0, "Scalar should move from right to left due to reverse flux");
    }
}
