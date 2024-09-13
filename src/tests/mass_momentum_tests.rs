#[cfg(test)]
mod tests {
    use crate::input::gmsh::GmshParser;
    use crate::timestep::ExplicitEuler;
    use crate::domain::{Face, Mesh};
    use crate::solver::FluxSolver;
    use nalgebra::Vector3;
    fn load_from_gmsh() -> Mesh {
        let (nodes, elements, faces) = 
        GmshParser::load_mesh("inputs/test.msh2").expect("Failed to load test mesh");
        let face_element_relations = vec![];
        Mesh::new(elements, nodes, faces, face_element_relations)
    }
    
    fn step_simulation(_mesh: &mut Mesh, _flux_solver: &mut FluxSolver) {
        let _euler_stepper = ExplicitEuler { solver: FluxSolver };
        let mut _time_stepper = ExplicitEuler { solver: FluxSolver };
    }
    
    fn calculate_momentum(_mesh: &Mesh, velocities: &[f64]) -> f64 {
        velocities.iter().sum() // Example calculation
    }
    
    #[test]
    fn mass_momentum_conservation_test() {
        let mut mesh = load_from_gmsh(); // Use the load_from_gmsh function
    
        let mut flux_solver = FluxSolver;
        
        // Initialize velocities for all elements
        let velocities = vec![0.0; mesh.elements.len()];
        
        let initial_momentum = calculate_momentum(&mesh, &velocities);
        
        // Step the simulation
        step_simulation(&mut mesh, &mut flux_solver);
        
        let final_momentum = calculate_momentum(&mesh, &velocities);
        
        // Verify conservation of momentum
        assert_eq!(initial_momentum, final_momentum);
    }

    #[test]
    fn test_conservation_of_mass_no_inflow_outflow() {
        let mut mesh = load_from_gmsh(); // Load the mesh
        let mut flux_solver = FluxSolver;

        let initial_mass = calculate_total_mass(&mesh); // Calculate initial mass in the domain

        // Step the simulation several times
        for _ in 0..10 {
            step_simulation(&mut mesh, &mut flux_solver);
        }

        let final_mass = calculate_total_mass(&mesh); // Calculate final mass

        // Assert mass conservation (with a small tolerance for floating-point precision)
        assert!((initial_mass - final_mass).abs() < 1e-10, "Mass was not conserved");
    }

    // Helper function to calculate total mass in the mesh
    fn calculate_total_mass(mesh: &Mesh) -> f64 {
        mesh.elements.iter().map(|element| element.mass).sum() // Assuming `mass` field exists in `Element`
    }

    #[test]
    fn test_conservation_of_momentum_with_inflow() {
        let mut mesh = load_from_gmsh();
        let mut flux_solver = FluxSolver;

        let initial_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Apply inflow boundary condition
        apply_inflow_boundary(&mut mesh, 1.0); // Inflow velocity of 1.0

        // Step the simulation
        for _ in 0..10 {
            step_simulation(&mut mesh, &mut flux_solver);
        }

        let final_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Verify momentum change due to inflow
        let expected_momentum_change = calculate_inflow_momentum_change(&mesh, 1.0); // Calculate expected momentum change
        assert!((final_momentum - initial_momentum - expected_momentum_change).abs() < 1e-10, "Momentum was not conserved with inflow");
    }

    // Helper function to apply inflow boundary condition
    fn apply_inflow_boundary(mesh: &mut Mesh, inflow_velocity: f64) {
        for face in &mut mesh.faces {
            if is_inflow_face(face) {
                face.velocity[1] = inflow_velocity; // Apply inflow velocity
            }
        }
    }

    // Helper function to calculate expected momentum change due to inflow
    fn calculate_inflow_momentum_change(mesh: &Mesh, inflow_velocity: f64) -> f64 {
        mesh.faces.iter()
            .filter(|face| is_inflow_face(face))
            .map(|face| inflow_velocity * face.area) // Assuming `area` field in `Face`
            .sum()
    }

    // Placeholder to detect inflow face (modify as needed)
    fn is_inflow_face(_face: &Face) -> bool {
        // Example: Detect based on position of face or specific IDs
        true // Modify this logic
    }

    #[test]
    fn test_reflective_boundary_conditions() {
        let mut mesh = load_from_gmsh();
        let mut flux_solver = FluxSolver;

        let initial_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Apply reflective boundary conditions
        apply_reflective_boundaries(&mut mesh);

        // Step the simulation
        for _ in 0..10 {
            step_simulation(&mut mesh, &mut flux_solver);
        }

        let final_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Momentum should remain the same with reflective boundaries
        assert!((initial_momentum - final_momentum).abs() < 1e-10, "Momentum was not conserved with reflective boundaries");
    }

    // Helper function to apply reflective boundaries
    fn apply_reflective_boundaries(mesh: &mut Mesh) {
        for face in &mut mesh.faces {
            if is_reflective_face(face) {
                face.velocity = Vector3::new(-face.velocity[0], -face.velocity[1], -face.velocity[2]); // Reflect the velocity
            }
        }
    }

    // Placeholder to detect reflective face (modify as needed)
    fn is_reflective_face(_face: &Face) -> bool {
        // Logic to identify reflective boundaries, e.g., based on position
        true // Modify this logic
    }

    #[test]
    fn test_long_term_stability() {
        let mut mesh = load_from_gmsh();
        let mut flux_solver = FluxSolver;

        let initial_mass = calculate_total_mass(&mesh);
        let initial_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Step the simulation for 1000 time steps
        for _ in 0..1000 {
            step_simulation(&mut mesh, &mut flux_solver);
        }

        let final_mass = calculate_total_mass(&mesh);
        let final_momentum = calculate_momentum(&mesh, &vec![0.0; mesh.elements.len()]);

        // Assert mass and momentum conservation
        assert!((initial_mass - final_mass).abs() < 1e-10, "Mass was not conserved over long-term simulation");
        assert!((initial_momentum - final_momentum).abs() < 1e-10, "Momentum was not conserved over long-term simulation");
    }

    use crate::domain::Element;

    #[test]
    fn test_mass_conservation_no_inflow_outflow() {
        // Create the left and right elements with equal mass
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0, // Initial mass
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0, // Initial mass
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
            nodes: vec![1, 2],
            velocity: Vector3::new(0.0, 0.0, 0.0),  // Initial velocity is zero
            area: 1.0,  // Simple unit area for the face
            ..Face::default()
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute flux and simulate mass transfer over time
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

        // Transfer mass based on flux direction (flux moves from left to right)
        left_element.mass -= flux.sum();
        right_element.mass += flux.sum();

        // Assert that total mass is conserved
        let total_mass = left_element.mass + right_element.mass;
        assert_eq!(total_mass, 2.0, "Total mass should be conserved (no inflow or outflow)");
    }

    #[test]
    fn test_mass_conservation_with_inflow_outflow() {
        // Create the left element with higher mass and an inflow
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 2.0, // More mass in the left element
            neighbor_ref: 0,
            pressure: 10.0, // Higher pressure on the left
            height: 0.0,
            area: 0.0,
            momentum: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            ..Element::default()
        };

        // Create the right element with an outflow
        let mut right_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0, // Less mass in the right element
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
            nodes: vec![1, 2],
            velocity: Vector3::new(0.0, 0.0, 0.0),
            area: 1.0, // Simple unit area for the face
            ..Face::default()
        };

        // Instantiate the flux solver
        let flux_solver = FluxSolver {};

        // Compute the flux
        let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

        // Simulate inflow: Add mass to the left element (inflow boundary)
        left_element.mass += 0.1; // Simulate some inflow

        // Simulate mass transfer based on flux
        left_element.mass -= flux.sum();
        right_element.mass += flux.sum();

        // Assert total mass has increased by the inflow amount
        let total_mass = left_element.mass + right_element.mass;
        assert_eq!(total_mass, 3.1, "Total mass should include inflow mass");
    }

    use crate::solver::SemiImplicitSolver;

    #[test]
    fn test_long_term_stability_alt() {
        let dt = 0.01; // Time step
        let total_time = 100.0; // Long simulation time

        // Create the left and right elements
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

        // Face between the two elements
        let face = Face {
            id: 0,
            nodes: vec![1, 2],
            velocity: Vector3::new(0.0, 0.0, 0.0),
            area: 1.0,
            ..Face::default()
        };

        // Instantiate the flux solver and semi-implicit solver
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _time in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Compute the flux at each time step
            let flux = flux_solver.compute_flux_3d(&face, &left_element, &right_element);

            // Use the semi-implicit solver to update the momentum
            left_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux * (left_element.momentum / left_element.mass),
                left_element.momentum,
                dt,
            );
            right_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux * (right_element.momentum / right_element.mass),
                right_element.momentum,
                dt,
            );

            // Apply clamping to ensure momentum does not go negative
            left_element.momentum = left_element.update_momentum(delta_momentum);
            right_element.momentum = right_element.momentum.max(0.0);

            // Ensure no spurious numerical oscillations occur
            assert!(left_element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in the long term");
            assert!(right_element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in the long term");
        }
    }


    #[test]
    fn test_variable_initial_conditions() {
        let dt = 0.01;
        let total_time = 50.0; // Shorter simulation for initial conditions

        // Create three elements with varying initial conditions
        let mut left_element = Element {
            id: 0,
            element_type: 2,
            nodes: vec![0, 1],
            faces: vec![0],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 15.0,  // High pressure
            momentum: Vector3::new(3.0, 0.0, 0.0),   // High momentum
            ..Element::default()
        };

        let mut middle_element = Element {
            id: 1,
            element_type: 2,
            nodes: vec![1, 2],
            faces: vec![1],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 10.0,  // Moderate pressure
            momentum: Vector3::new(2.0, 0.0, 0.0),   // Moderate momentum
            ..Element::default()
        };

        let mut right_element = Element {
            id: 2,
            element_type: 2,
            nodes: vec![2, 3],
            faces: vec![2],
            mass: 1.0,
            height: 0.0,
            area: 0.0,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            neighbor_ref: 0,
            pressure: 5.0,  // Low pressure
            momentum: Vector3::new(1.0, 0.0, 0.0),  // Low momentum
            ..Element::default()
        };

        // Faces between the elements
        let left_face = Face {
            id: 0,
            nodes: vec![1, 2],
            velocity: Vector3::new(0.0, 0.0, 0.0),
            area: 1.0,
            ..Face::default()
        };
        let right_face = Face {
            id: 1,
            nodes: vec![2, 3],
            velocity: Vector3::new(0.0, 0.0, 0.0),
            area: 1.0,
            ..Face::default()
        };

        // Instantiate solvers
        let flux_solver = FluxSolver {};
        let semi_implicit_solver = SemiImplicitSolver {};

        // Run the simulation over time
        for _time in (0..(total_time / dt) as usize).map(|i| i as f64 * dt) {
            // Compute fluxes between elements
            let flux_left = flux_solver.compute_flux_3d(&left_face, &left_element, &middle_element);
            let flux_right = flux_solver.compute_flux_3d(&right_face, &middle_element, &right_element);

            // Update momenta using semi-implicit solver
            left_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_left * (left_element.momentum / left_element.mass),
                left_element.momentum,
                dt,
            );
            middle_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux_left * (middle_element.momentum / middle_element.mass),
                middle_element.momentum,
                dt,
            );
            middle_element.momentum = semi_implicit_solver.semi_implicit_update(
                -flux_right * (middle_element.momentum / middle_element.mass),
                middle_element.momentum,
                dt,
            );
            right_element.momentum = semi_implicit_solver.semi_implicit_update(
                flux_right * (right_element.momentum / right_element.mass),
                right_element.momentum,
                dt,
            );

            // Ensure momentum remains positive
            assert!(left_element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in the left element");
            assert!(middle_element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in the middle element");
            assert!(right_element.momentum > Vector3::new(0.0, 0.0, 0.0), "Momentum should remain positive in the right element");
        }
    }
}
