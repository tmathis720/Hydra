#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::{FluxSolver, SemiImplicitSolver};

    #[test]
    fn test_multi_element_grid() {
        let dt = 0.01;
        let total_time = 20.0;

        // Define a small grid of elements (e.g., 4 elements in a line)
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 3.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.0, neighbor_ref: 0, pressure: 7.0, momentum: 2.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 5.0, momentum: 1.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
            Element { id: 3, element_type: 2, nodes: vec![3, 4], faces: vec![3], mass: 1.0, neighbor_ref: 0, pressure: 3.0, momentum: 1.0, height: 0.0, area: 0.0, velocity: (0.0,0.0,0.0) },
        ];

        // Define faces between elements
        let faces = vec![
            Face { id: 0, nodes: (1, 2), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: (2, 3), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 2, nodes: (3, 4), velocity: (0.0, 0.0), area: 1.0 },
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            for i in 0..faces.len() {
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[i + 1]);

                // Update momentum with semi-implicit solver
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

                // Assert positive momentum
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[i + 1].momentum > 0.0, "Momentum should remain positive in element {}", i + 1);
            }
        }
    }
}
