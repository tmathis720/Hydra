#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::FluxSolver;

    #[test]
    fn test_reflective_boundary_conditions() {
        // Create the element near the reflective boundary
        let element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0, // Pressure near the boundary
            momentum: 2.0,  // Initial momentum
        };

        // Reflective boundary face (effectively no flux across this face)
        let face = Face {
            id: 0,
            nodes: (1, 2),
            velocity: (0.0, 0.0), // Initial velocity is zero
            area: 1.0, // Simple unit area for the face
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux(&face, &element, &element);

        // Assert that no flux crosses the reflective boundary
        assert_eq!(flux, 0.0, "No flux should cross the reflective boundary");
    }
}
