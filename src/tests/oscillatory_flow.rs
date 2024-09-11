#[cfg(test)]
mod tests {
    use crate::domain::element::Element;
    use crate::domain::face::Face;
    use crate::solver::FluxSolver;
    use std::f64::consts::PI;

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
