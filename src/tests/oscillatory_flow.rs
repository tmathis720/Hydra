#[cfg(test)]
mod tests {
    use crate::domain::{Element, Face};
    use crate::solver::{FluxSolver, SemiImplicitSolver};
    use std::f64::consts::PI;

    #[test]
    fn test_oscillatory_flow_with_periodic_boundaries() {
        let dt = 0.01;
        let total_time = 10.0;
        let oscillation_frequency = 2.0 * PI / total_time; // Define oscillation frequency

        // Define 4 elements in a periodic grid
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 2.0 },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.0, neighbor_ref: 0, pressure: 8.0, momentum: 1.5 },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 5.0, momentum: 1.0 },
            Element { id: 3, element_type: 2, nodes: vec![3, 0], faces: vec![3], mass: 1.0, neighbor_ref: 0, pressure: 7.0, momentum: 1.2 },
        ];

        // Faces between the elements (including periodic connection)
        let faces = vec![
            Face { id: 0, nodes: (1, 2), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: (2, 3), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 2, nodes: (3, 0), velocity: (0.0, 0.0), area: 1.0 },  // Periodic face
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time with oscillating flow
        for t in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Adjust the pressure in a sinusoidal pattern to simulate oscillatory flow
            for (_i, element) in elements.iter_mut().enumerate() {
                element.pressure = 10.0 + 5.0 * (oscillation_frequency * t).sin();
            }

            // Compute flux between elements with periodic boundary
            for i in 0..faces.len() {
                let next_idx = if i == faces.len() - 1 { 0 } else { i + 1 };
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[next_idx]);

                // Update momentum with semi-implicit solver
                elements[i].momentum = semi_implicit_solver.semi_implicit_update(
                    -flux * (elements[i].momentum / elements[i].mass),
                    elements[i].momentum,
                    dt,
                );
                elements[next_idx].momentum = semi_implicit_solver.semi_implicit_update(
                    flux * (elements[next_idx].momentum / elements[next_idx].mass),
                    elements[next_idx].momentum,
                    dt,
                );

                // Assert momentum remains positive
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[next_idx].momentum > 0.0, "Momentum should remain positive in element {}", next_idx);
            }
        }
    }

    #[test]
    fn test_periodic_boundary_conditions() {
        let dt = 0.01;
        let total_time = 20.0;

        // Define a set of 4 elements in a loop (periodic boundary)
        let mut elements: Vec<Element> = vec![
            Element { id: 0, element_type: 2, nodes: vec![0, 1], faces: vec![0], mass: 1.0, neighbor_ref: 0, pressure: 10.0, momentum: 2.0 },
            Element { id: 1, element_type: 2, nodes: vec![1, 2], faces: vec![1], mass: 1.0, neighbor_ref: 0, pressure: 8.0, momentum: 1.5 },
            Element { id: 2, element_type: 2, nodes: vec![2, 3], faces: vec![2], mass: 1.0, neighbor_ref: 0, pressure: 5.0, momentum: 1.0 },
            Element { id: 3, element_type: 2, nodes: vec![3, 0], faces: vec![3], mass: 1.0, neighbor_ref: 0, pressure: 7.0, momentum: 1.2 },
        ];

        // Faces between the elements (including a face between the last and the first element for periodicity)
        let faces = vec![
            Face { id: 0, nodes: (1, 2), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 1, nodes: (2, 3), velocity: (0.0, 0.0), area: 1.0 },
            Face { id: 2, nodes: (3, 0), velocity: (0.0, 0.0), area: 1.0 },  // Periodic face
        ];

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _ in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            for i in 0..faces.len() {
                // Compute flux between elements with periodic boundary (wraps around from last to first element)
                let next_idx = if i == faces.len() - 1 { 0 } else { i + 1 };
                let flux = flux_solver.compute_flux(&faces[i], &elements[i], &elements[next_idx]);

                // Update momentum using the semi-implicit solver
                elements[i].momentum = semi_implicit_solver.semi_implicit_update(
                    -flux * (elements[i].momentum / elements[i].mass),
                    elements[i].momentum,
                    dt,
                );
                elements[next_idx].momentum = semi_implicit_solver.semi_implicit_update(
                    flux * (elements[next_idx].momentum / elements[next_idx].mass),
                    elements[next_idx].momentum,
                    dt,
                );

                // Assert positive momentum
                assert!(elements[i].momentum > 0.0, "Momentum should remain positive in element {}", i);
                assert!(elements[next_idx].momentum > 0.0, "Momentum should remain positive in element {}", next_idx);
            }
        }
    }

    #[test]
    fn test_oscillatory_flow_in_closed_channel() {
        // Define time step and total simulation time
        let dt = 0.01;
        let total_time = 1.0;

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Create the left and right elements
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 10.0, // Initial pressure (will oscillate)
            momentum: 0.0,
        };

        let right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            neighbor_ref: 0,
            pressure: 5.0, // Constant pressure on the right
            momentum: 0.0,
        };

        // Define the face between the two elements
        let face = Face {
            id: 0,
            nodes: (1, 2),  // Shared between the two elements
            velocity: (0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
        };

        // Loop over time steps to simulate oscillatory pressure
        for time in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Oscillate the pressure of the left element (e.g., sinusoidal variation)
            left_element.pressure = 10.0 + 5.0 * (2.0 * PI * time).sin(); // Amplitude of 5

            // Compute flux at this time step
            let flux = flux_solver.compute_flux(&face, &left_element, &right_element);

            // Test expectations: if left pressure > right pressure, flux should be positive
            if left_element.pressure > right_element.pressure {
                assert!(flux > 0.0, "Flux should be positive when left pressure is higher");
            } 
            // If right pressure > left pressure, flux should be negative
            else if right_element.pressure > left_element.pressure {
                assert!(flux < 0.0, "Flux should be negative when right pressure is higher");
            } 
            // Equal pressure means no flux
            else {
                assert_eq!(flux, 0.0, "Flux should be zero when pressures are equal");
            }

            // Output for debugging purposes
            println!(
                "Time: {:.3}, Left Pressure: {:.3}, Right Pressure: {:.3}, Flux: {:.3}",
                time, left_element.pressure, right_element.pressure, flux
            );
        }
    }
}
