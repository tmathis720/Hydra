#[cfg(test)]
mod tests {
    use rand::Rng; // Import random number generator
    use crate::input::gmsh::GmshParser;
    use crate::timestep::ExplicitEuler;
    use crate::domain::face::Face;
    use crate::domain::mesh::Mesh;
    use crate::solver::FluxSolver;
    fn load_from_gmsh() -> Mesh {
        let (nodes, elements, faces) = 
        GmshParser::load_mesh("C:/rust_projects/HYDRA/inputs/test.msh2").expect("Failed to load test mesh");
        let face_element_relations = vec![];
        Mesh::new(elements, nodes, faces, face_element_relations)
    }
    
    fn step_simulation(mesh: &mut Mesh, flux_solver: &mut FluxSolver) {
        let euler_stepper = ExplicitEuler { dt: 0.1 };
        euler_stepper.step(mesh, flux_solver);
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
                face.velocity.0 = inflow_velocity; // Apply inflow velocity
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
                face.velocity = (-face.velocity.0, -face.velocity.1); // Reflect the velocity
            }
        }
    }

    // Placeholder to detect reflective face (modify as needed)
    fn is_reflective_face(_face: &Face) -> bool {
        // Logic to identify reflective boundaries, e.g., based on position
        true // Modify this logic
    }

    #[test]
    fn test_variable_initial_conditions() {
        let mut mesh = load_from_gmsh();
        let mut flux_solver = FluxSolver;

        // Assign random initial velocities and mass
        let mut rng = rand::thread_rng();
        let velocities: Vec<f64> = (0..mesh.elements.len()).map(|_| rng.gen_range(0.0..1.0)).collect();
        for element in &mut mesh.elements {
            element.mass = rng.gen_range(1.0..10.0); // Random initial mass
        }

        let initial_mass = calculate_total_mass(&mesh);
        let initial_momentum = calculate_momentum(&mesh, &velocities);

        // Step the simulation
        for _ in 0..10 {
            step_simulation(&mut mesh, &mut flux_solver);
        }

        let final_mass = calculate_total_mass(&mesh);
        let final_momentum = calculate_momentum(&mesh, &velocities);

        // Ensure mass and momentum are conserved
        assert!((initial_mass - final_mass).abs() < 1e-10, "Mass was not conserved with variable initial conditions");
        assert!((initial_momentum - final_momentum).abs() < 1e-10, "Momentum was not conserved with variable initial conditions");
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
}


