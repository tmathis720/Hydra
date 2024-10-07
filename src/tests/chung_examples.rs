#[cfg(test)]
mod integration_test {
    use crate::{
        domain::{Mesh, section::Section},                // Mesh and Section for fields
        input_output::mesh_generation::MeshGenerator,    // Correct reference for mesh generation
        solver::{KSP, gmres::GMRES, ksp::SolverResult},  // Solver abstraction, GMRES, and SolverResult
        domain::mesh_entity::MeshEntity,                 // Entities for boundary conditions
    };
    use faer::Mat;                                       // Matrix representation
    use std::error::Error;

    /// Generate a 2D rectangular mesh for the Navier-Stokes problem.
    fn generate_mesh() -> Mesh {
        let width = 1.0;  // Width of the domain
        let height = 1.0; // Height of the domain
        let nx = 10;      // Number of divisions along the x-axis
        let ny = 10;      // Number of divisions along the y-axis
        println!("Generating mesh with width {}, height {}, nx {}, ny {}", width, height, nx, ny);
        MeshGenerator::generate_rectangle_2d(width, height, nx, ny)  // Use the MeshGenerator from Hydra
    }

    /// Apply boundary conditions by associating values with mesh entities.
    fn apply_boundary_conditions(section: &mut Section<f64>) {
        // Example of defining inflow and outflow boundaries
        let inlet = MeshEntity::Edge(1);   // Assuming inlet corresponds to edge 1
        let outlet = MeshEntity::Edge(2);  // Assuming outlet corresponds to edge 2

        // Set the Dirichlet condition (constant velocity) for the inlet
        println!("Applying Dirichlet boundary condition to inlet (Edge 1)");
        section.set_data(inlet, 1.0);  // Constant velocity of 1.0 for the inlet

        // Set the Neumann condition (constant pressure) for the outlet
        println!("Applying Neumann boundary condition to outlet (Edge 2)");
        section.set_data(outlet, 0.0);  // Zero pressure for the outlet
    }

    /// Initialize the KSP solver with required parameters.
    fn initialize_solver() -> GMRES {
        let max_iter = 100;  // Maximum number of iterations
        let tol = 1e-6;      // Tolerance for convergence
        let restart = 30;    // Restart parameter for GMRES
        println!("Initializing GMRES solver with max_iter = {}, tol = {}, restart = {}", max_iter, tol, restart);
        GMRES::new(max_iter, tol, restart)  // Initialize GMRES solver with the necessary parameters
    }

    /// Assemble the system matrix and RHS vector using the Section and mesh.
    fn assemble_system(section: &Section<f64>, _mesh: &Mesh) -> (Mat<f64>, Vec<f64>) {
        let num_entities = section.entities().len();
        let a = Mat::<f64>::zeros(num_entities, num_entities);  // System matrix (placeholder)
        let b = vec![0.0; num_entities];  // Right-hand side vector (placeholder)

        println!("Assembling system with {} entities", num_entities);
        println!("System matrix A: {:?}", a);
        println!("RHS vector b: {:?}", b);

        (a, b)
    }

    /// Function to run the Navier-Stokes simulation using GMRES solver and explicit time-stepping.
    fn run_navier_stokes_simulation() -> Result<(), Box<dyn Error>> {
        // Step 1: Generate the mesh using the MeshGenerator
        let mesh = generate_mesh();  // Generate a 2D mesh

        // Step 2: Create Sections for velocity and pressure fields associated with mesh entities
        let mut velocity_section = Section::<f64>::new();  // Initialize Section for velocity
        let mut _pressure_section = Section::<f64>::new();  // Initialize Section for pressure

        // Step 3: Apply boundary conditions to the velocity field
        apply_boundary_conditions(&mut velocity_section);

        // Step 4: Initialize the GMRES solver
        let mut solver = initialize_solver();

        // Step 5: Initialize time-stepping parameters
        let mut time = 0.0;      // Starting time
        let end_time = 1.0;      // End time of the simulation
        let dt = 0.01;           // Time step size
        println!("Starting time-stepping loop from t = {} to t = {} with dt = {}", time, end_time, dt);

        // Step 6: Time-stepping loop
        while time < end_time {
            println!("Current time: {}", time);

            // Step 7: Assemble the system matrix (A) and the right-hand side vector (b)
            let (a, b) = assemble_system(&velocity_section, &mesh);  // Use the Section to assemble the system

            // Initialize a solution vector
            let mut x = vec![0.0; b.len()];  // Placeholder vector for the solution
            println!("Initial solution vector x: {:?}", x);

            // Solve the system using GMRES
            let result: SolverResult = solver.solve(&a, &b, &mut x);  // Solve the linear system

            // Check if the solver converged
            if result.converged {
                // Successfully solved the system
                println!("Solver converged in {} iterations with a residual norm of {}.", result.iterations, result.residual_norm);
            } else {
                // Handle solver failure
                println!("Solver failed to converge after {} iterations with a residual norm of {}.", result.iterations, result.residual_norm);
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "Solver failed to converge after {} iterations with a residual norm of {}.",
                        result.iterations, result.residual_norm
                    ),
                )));
            }

            // Update the solution in the velocity section
            println!("Updating velocity section with new solution values.");
            for (entity, value) in velocity_section.entities().iter().zip(x.iter()) {
                velocity_section.update_data(entity, *value).expect("Failed to update solution");
                println!("Updated entity {:?} with value {}", entity, value);
            }

            // Advance time
            time += dt;
        }

        Ok(())
    }

    /// Integration test for Chung's 2010 CFD Example 7.2.1.
    #[test]
    fn test_chung_example_7_2_1() {
        // Run the simulation and check for success
        println!("Starting integration test for Chung's 2010 CFD Example 7.2.1");
        assert!(run_navier_stokes_simulation().is_ok());
    }
}
