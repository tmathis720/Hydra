#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::domain::{Element, Face, Mesh};
    use crate::domain::FlowField;
    use crate::boundary::{BoundaryElement, BoundaryType};
    use crate::solver::{CrankNicolsonSolver, EddyViscositySolver, FluxSolver};
    use crate::timestep::{TimeStepper, ExplicitEuler};
    use std::rc::Rc;
    use std::cell::RefCell;


    #[test]
    fn test_3d_flux_solver() {
        let mut face = Face {
            id: 0,
            nodes: vec![0, 1, 2, 3],
            velocity: Vector3::new(1.0, 1.0, 0.0),  // Simple 2D flow in x and y
            area: 2.0,
            ..Face::default()
        };

        let mut left_element = Element {
            id: 0,
            pressure: 10.0,
            mass: 1.0,
            momentum: Vector3::new(1.0, 0.0, 0.0),
            velocity: Vector3::new(1.0, 1.0, 0.0),
            ..Default::default()
        };

        let mut right_element = Element {
            id: 1,
            pressure: 5.0,
            mass: 1.0,
            momentum: Vector3::new(0.5, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Default::default()
        };

        let flux_solver = FluxSolver {};

        // Compute 3D flux
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);
        assert!(flux > Vector3::new(0.0, 0.0, 0.0));

        // Apply the flux to update the face
        flux_solver.apply_flux_3d(&mut face, flux, 0.01);

        // Update the element properties based on the flux
        flux_solver.update_elements(&mut left_element, &mut right_element, (flux, flux, 0.0), 0.01);

        // Assert mass and momentum updates
        assert!(left_element.mass < 1.0);
        assert!(right_element.mass > 1.0);
    }


    #[test]
    fn test_complex_integration_with_horizontal_diffusion() {
        let dt = 0.01;
        let total_time = 10.0;
    
        // Define elements in the domain
        let elements = vec![
            Element {   id: 0, 
                        element_type: 2, 
                        nodes: vec![0, 1], 
                        faces: vec![0], 
                        mass: 1.0, 
                        neighbor_ref: 0, 
                        pressure: 10.0, 
                        height: 0.0, 
                        area: 1.0, 
                        momentum: Vector3::new(2.0, 0.0, 0.0), 
                        ..Element::default() },
            Element {   id: 1, 
                        element_type: 2, 
                        nodes: vec![1, 2], 
                        faces: vec![1], 
                        mass: 1.0, 
                        neighbor_ref: 0, 
                        pressure: 8.0, 
                        height: 0.0, 
                        area: 1.0, 
                        momentum: Vector3::new(1.5, 0.0, 0.0), 
                        ..Element::default() },
            Element {   id: 2, 
                        element_type: 2, 
                        nodes: vec![1, 2], 
                        faces: vec![1], 
                        mass: 1.0, 
                        neighbor_ref: 0, 
                        pressure: 8.0, 
                        height: 0.0, 
                        area: 1.0, 
                        momentum: Vector3::new(1.0, 0.0, 0.0), 
                        ..Element::default() },
        ];
    
        // Define faces between elements
        let faces = vec![
            Face { id: 0, nodes: vec![1, 2], area: 1.0, ..Face::default() },
            Face { id: 1, nodes: vec![2, 3], area: 1.0, ..Face::default() },
        ];
    
        // Create the mesh
        let mut mesh = Mesh {
            nodes: vec![],
            neighbors: vec![],
            elements: elements.clone(),
            faces,
            face_element_relations: vec![], // Populate as needed
        };
    
        // Instantiate FlowField with the elements
        let mut flow_field = FlowField::new(elements);
    
        // Set up boundary conditions
        let boundary_element: Vec<BoundaryElement> = mesh.elements
            .iter()
            .map(|element| BoundaryElement {
                element: Rc::new(RefCell::new(element.clone())),
                boundary_type: BoundaryType::Periodic, // Change this if testing other boundary types
            })
            .collect();
    
        let crank_nicolson_solver = CrankNicolsonSolver {};
        let eddy_viscosity_solver = EddyViscositySolver { nu_t: 0.1 }; // Eddy viscosity coefficient
        let mut flux_solver = FluxSolver {};
    
        // Define a time stepper
        let time_stepper = ExplicitEuler { solver: flux_solver };
    
        // Clone the boundary elements for manipulation
        let mut boundaries: Vec<BoundaryElement> = boundary_element.clone();
    
        // Time loop to run the simulation
        for _ in 0..(total_time / dt) as usize {
            // Apply boundary conditions to the elements
            for boundary in &mut boundaries {
                boundary.apply_boundary_condition(&mut flow_field, time_stepper.step);
            }
    
            // Flux and pressure updates between neighboring elements
            for i in 0..mesh.faces.len() {
                let (left_element, right_element) = mesh.elements.split_at_mut(i + 1);
                let left_element = &mut left_element[i];
                let right_element = &mut right_element[0];
    
                let pressure_diff = left_element.pressure - right_element.pressure;
                let flux = pressure_diff * mesh.faces[i].area;
    
                // Update momentum using Crank-Nicolson method
                left_element.momentum = crank_nicolson_solver.crank_nicolson_update(flux, left_element.momentum, dt);
                right_element.momentum = crank_nicolson_solver.crank_nicolson_update(-flux, right_element.momentum, dt);
    
                // Adjust pressures between elements
                let pressure_transfer = 0.01 * flux * dt;
                left_element.pressure = (left_element.pressure - pressure_transfer).max(0.0);
                right_element.pressure = (right_element.pressure + pressure_transfer).max(0.0);
            }
    
            // Apply horizontal eddy viscosity
            for i in 0..mesh.elements.len() - 1 {
                let (left_element, right_element) = mesh.elements.split_at_mut(i + 1);
                let left_element = &mut left_element[i];
                let right_element = &mut right_element[0];
    
                eddy_viscosity_solver.apply_diffusion(left_element, right_element, dt);
            }
    
            // Step forward in time using the updated mesh and flux solver
            time_stepper.step(&mut mesh, dt);
        }
    
        // Final assertions to ensure positive momentum and pressure
        for element in &mesh.elements {
            assert!(element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in all elements");
            assert!(element.pressure > 0.0, "Pressure should remain positive in all elements");
        }
    }

    use crate::domain::Node;

    // Helper function to calculate the center position of a 3D element
    fn calculate_center_position(element: &Element, nodes: &Vec<Node>) -> (f64, f64, f64) {
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut z_sum = 0.0;
        for &node_id in &element.nodes {
            let node = &nodes[node_id];
            x_sum += node.position[0];
            y_sum += node.position[1];
            z_sum += node.position[2];
        }
        (
            x_sum / element.nodes.len() as f64,
            y_sum / element.nodes.len() as f64,
            z_sum / element.nodes.len() as f64,
        )
    }

    use crate::numerical::MeshGenerator;

    // Test Case: 3D Wave Propagation
    #[test]
    fn test_3d_wave_propagation() {
        let domain_size_x = 100.0;  // Length of the domain (x-direction)
        let domain_size_y = 50.0;   // Width of the domain (y-direction)
        let domain_size_z = 30.0;   // Height of the domain (z-direction)
        let n_elements_x = 20;      // Number of elements in the x-direction
        let n_elements_y = 10;      // Number of elements in the y-direction
        let n_elements_z = 6;       // Number of elements in the z-direction

        // Step 1: Generate a 3D mesh for the rectangular domain
        let mut mesh = MeshGenerator::generate_rectangle_3d(
            domain_size_x, 
            domain_size_y, 
            domain_size_z, 
            n_elements_x, 
            n_elements_y, 
            n_elements_z
        );

        // Step 2: Define initial conditions for the 3D wave propagation
        let initial_pressure = 1.0;  // Initial pressure at the wave center
        let initial_velocity = Vector3::new(0.0, 0.0, 0.0); // Initial velocity (at rest)
        
        // Set initial pressure and velocity in the mesh elements
        for element in &mut mesh.elements {
            let (center_x, center_y, center_z) = calculate_center_position(&element, &mesh.nodes);

            // Set the pressure in the center of the domain (as a wave pulse)
            if (center_x - domain_size_x / 2.0).abs() < 10.0 && 
            (center_y - domain_size_y / 2.0).abs() < 10.0 && 
            (center_z - domain_size_z / 2.0).abs() < 10.0 {
                element.pressure = initial_pressure;
            } else {
                element.pressure = 0.0;  // Elsewhere, the domain is at rest (zero pressure)
            }

            element.velocity = initial_velocity;  // Initial velocity is zero everywhere
        }

        // Step 3: Set up solvers and time stepping
        let dt = 0.01;  // Time step size
        let total_time = 5.0;  // Simulate for 5 seconds
        
        let mut flux_solver = FluxSolver {};  // 3D flux solver
        let time_stepper = ExplicitEuler { solver: FluxSolver };

        // Track the maximum pressure in the domain for verification
        let mut max_pressure: f64 = 0.0;

        // Step 4: Run the simulation over time
        for _ in 0..((total_time / dt) as usize) {
            // Step forward in time
            time_stepper.step(&mut mesh, dt);

            // Check the maximum pressure in the domain
            for element in &mesh.elements {
                max_pressure = max_pressure.max(element.pressure);
            }

            // Output simulation progress for debugging
            if max_pressure > 0.0 {
                println!("Time: {}, Max pressure: {}", total_time, max_pressure);
            }
        }

        // Step 5: Validate the results
        // The test should check that the wave has propagated from the center outward.
        let mut wave_propagated = false;

        for element in &mesh.elements {
            let (center_x, center_y, center_z) = calculate_center_position(&element, &mesh.nodes);

            // Check if the wave has propagated to the outer regions of the domain
            if (center_x - domain_size_x / 2.0).abs() > 20.0 || 
            (center_y - domain_size_y / 2.0).abs() > 20.0 || 
            (center_z - domain_size_z / 2.0).abs() > 20.0 {
                if element.pressure > 0.0 {
                    wave_propagated = true;
                    break;
                }
            }
        }

        assert!(wave_propagated, "Wave should propagate to the edges of the domain.");
    }


    //use crate::domain::Node;  // Assuming these exist in the domain module
    //use crate::numerical::MeshGenerator;  // For generating rectangular meshes

    // Helper function to calculate the center x-position of an element
    /* fn calculate_center_x(element: &Element, nodes: &Vec<Node>) -> f64 {
        let mut x_sum = 0.0;
        for &node_id in &element.nodes {
            x_sum += nodes[node_id].position.0;  // Assume nodes are stored in the mesh by their IDs
        }
        x_sum / element.nodes.len() as f64  // Average the x-coordinates of the element’s nodes
    } */

/*     // Test Case 1: Dam Break over Dry Bed
    #[test]
    fn test_dam_break_over_dry_bed() {
        let mesh_size_x = 100.0;  // Length of the domain (x-direction)
        let mesh_size_y = 10.0;   // Width of the domain (y-direction)
        let n_elements_x = 50;    // Number of elements in the x-direction
        let n_elements_y = 5;     // Number of elements in the y-direction
    
        // Step 1: Generate a rectangular mesh
        let mut mesh = MeshGenerator::generate_rectangle(mesh_size_x, mesh_size_y, n_elements_x, n_elements_y);
    
        // Step 2: Define initial conditions
        let initial_pressure_left = 2.0;  // Increased pressure representing water height on the left side
        let initial_pressure_right = 0.0; // Dry bed (no water, hence no pressure)
    
        for element in &mut mesh.elements {
            let center_x = calculate_center_x(&element, &mesh.nodes);
            if center_x < mesh_size_x / 2.0 {
                element.pressure = initial_pressure_left;
            } else {
                element.pressure = initial_pressure_right;
            }
        }
    
        // Step 3: Time stepping and simulation loop
        let dt = 0.01;  // Slightly larger time step to see if it improves wave propagation
        let total_time = 10.0;  // Simulate for 10 seconds
        let time_stepper = ExplicitEuler { dt };
        let mut flux_solver = FluxSolver {};  // Ensure this solver handles flux correctly
    
        let mut max_right_pressure: f64 = 0.0;
        let mut total_mass_initial: f64 = mesh.elements.iter().map(|e| e.mass).sum();  // Initial mass
    
        for t in 0..((total_time / dt) as usize) {
            // Step forward in time
            time_stepper.step(&mut mesh, &mut flux_solver);
    
            // Debug: Check pressure and mass conservation
            let mut total_mass: f64 = 0.0;
            for element in &mesh.elements {
                let center_x = calculate_center_x(&element, &mesh.nodes);
                total_mass += element.mass;  // Track mass conservation
                if center_x > mesh_size_x / 2.0 {
                    max_right_pressure = max_right_pressure.max(element.pressure);
                }
            }
    
            // Output simulation progress for debugging
            if t % 200 == 0 {
                println!("Time: {}, Max pressure on right side: {}, Total mass: {}", t as f64 * dt, max_right_pressure, total_mass);
            }
    
            // Check mass conservation
            assert!((total_mass - total_mass_initial).abs() < 1e-6, "Mass conservation violated at time {}", t as f64 * dt);
        }
    
        // Step 4: Validate results
        let mut wave_reached_right_side = false;
    
        for element in &mesh.elements {
            let center_x = calculate_center_x(&element, &mesh.nodes);
            if center_x > mesh_size_x / 2.0 && element.pressure > 0.0 {
                wave_reached_right_side = true;
                break;
            }
        }
    
        println!("Final maximum pressure on the right side: {}", max_right_pressure);
        assert!(wave_reached_right_side, "Wave should propagate to the right side of the domain");
    } */
}
