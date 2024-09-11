#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::FluxSolver;

    #[test]
    fn test_open_boundary_conditions() {
        // Create the left element (with inflow)
        let left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 2.0, // Mass entering the domain
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure for inflow
            momentum: 3.0,  // Higher momentum for inflow
        };

        // Create the right element (with outflow)
        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0, // Lower mass for outflow
            neighbor_ref: 0,
            pressure: 5.0, // Lower pressure for outflow
            momentum: 1.0,  // Lower momentum for outflow
        };

        // Open boundary face (inflow/outflow boundary)
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0), // Allow free flow
            area: 1.0,
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux between the elements
        let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

        // Assert that there is net flux crossing the open boundary
        assert!(flux > 0.0, "Flux should be positive across the open boundary");
    }
}
