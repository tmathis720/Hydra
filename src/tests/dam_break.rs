#[cfg(test)]
mod tests {
    use crate::domain::{ Element, Node};  // Assuming these exist in the domain module
    use crate::solver::FluxSolver;  // Assuming this exists for flux calculations
    use crate::timestep::{ExplicitEuler, TimeStepper};  // Time stepping method
    use crate::numerical::MeshGenerator;  // For generating rectangular meshes

    // Helper function to calculate the center x-position of an element
    fn calculate_center_x(element: &Element, nodes: &Vec<Node>) -> f64 {
        let mut x_sum = 0.0;
        for &node_id in &element.nodes {
            x_sum += nodes[node_id].position.0;  // Assume nodes are stored in the mesh by their IDs
        }
        x_sum / element.nodes.len() as f64  // Average the x-coordinates of the element’s nodes
    }

    // Test Case 1: Dam Break over Dry Bed
    #[test]
    fn test_dam_break_over_dry_bed() {
        let mesh_size_x = 100.0;  // Length of the domain (x-direction)
        let mesh_size_y = 10.0;   // Width of the domain (y-direction)
        let n_elements_x = 50;    // Number of elements in the x-direction
        let n_elements_y = 5;     // Number of elements in the y-direction

        // Step 1: Generate a rectangular mesh
        let mut mesh = MeshGenerator::generate_rectangle(mesh_size_x, mesh_size_y, n_elements_x, n_elements_y);

        // Step 2: Define initial conditions
        // We'll use pressure to simulate the water height, assuming a simple shallow water relationship.
        let initial_pressure_left = 1.0;  // Pressure representing water height on the left side
        let initial_pressure_right = 0.0; // Dry bed (no water, hence no pressure)

        for element in &mut mesh.elements {
            // Calculate center x-position of the element
            let center_x = calculate_center_x(&element, &mesh.nodes);

            // Set initial pressure: left half has water, right half is dry
            if center_x < mesh_size_x / 2.0 {
                element.pressure = initial_pressure_left;
            } else {
                element.pressure = initial_pressure_right;
            }
        }

        // Step 3: Time stepping and simulation loop
        let dt = 0.005;  // Time step size
        let total_time = 10.0;  // Simulate for 10 seconds
        let time_stepper = ExplicitEuler { dt };
        let mut flux_solver = FluxSolver { };  // Assumes some basic solver structure

        let mut max_right_pressure: f64 = 0.0;

        // Run the simulation over time
        for t in 0..((total_time / dt) as usize) {
            // Step forward in time
            time_stepper.step(&mut mesh, &mut flux_solver);

            
            // Debug: Check pressure on the right side after each step
            for element in &mesh.elements {
                let center_x = calculate_center_x(&element, &mesh.nodes);
                if center_x > mesh_size_x / 2.0 {
                    max_right_pressure = max_right_pressure.max(element.pressure);
                }
            }

            // Output simulation progress for debugging
            if t % 200 == 0 {
                println!("Time: {}, Max pressure on right side: {}", t as f64 * dt, max_right_pressure);
            }
        }

        // Debug: Print the maximum pressure on the right side
        println!("Maximum pressure on the right side: {}", max_right_pressure);

        // Step 4: Validate results
        // Check if the wavefront has propagated as expected
        // For simplicity, we check that pressure is non-zero in the right half of the domain after the wave propagates.
        let mut wave_reached_right_side = false;

        for element in &mesh.elements {
            // Calculate center x-position of the element
            let center_x = calculate_center_x(&element, &mesh.nodes);

            if center_x > mesh_size_x / 2.0 && element.pressure > 0.0 {
                wave_reached_right_side = true;
                break;
            }
        }
        println!("Final maximum pressure on the right side: {}", max_right_pressure);
        assert!(wave_reached_right_side, "Wave should propagate to the right side of the domain");
    }
}
