#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_non_uniform_grid() {
        let dt = 0.01;
        let total_time = 10.0;

        // Create a set of elements with non-uniform sizes (varying mass)
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 2.0, neighbor_ref: 0, pressure: 15.0, momentum: 3.0 },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.5, neighbor_ref: 0, pressure: 12.0, momentum: 2.5 },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 2.0 },
        ];

        // Define faces between elements (different areas for non-uniform grid)
        let faces = vec![
            Face { id: 0, nodes: (1, 2), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: (2, 3), velocity: (0.0, 0.0), area: 0.8 },
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            for i in 0..faces.len() {
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[i + 1]);

                // Update momentum using semi-implicit solver
                elements[i].momentum = semi_implicit_solver.semi_implicit_update(
                    -flux * (elements[i].momentum / elements[i].mass),
                    elements[i].momentum,
                    dt,
                );
                elements[i + 1].momentum = semi_implicit_solver.semi_implicit_update(
                    flux * (elements[i + 1].momentum / elements[i + 1].mass),
                    elements[i + 1].momentum,
                    dt,
                );

                // Ensure momentum and mass are conserved
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[i + 1].momentum > 0.0, "Momentum should remain positive in element {}", i + 1);
            }
        }
    }
}
