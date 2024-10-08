#[cfg(test)]
mod integration_test {
    use crate::{
        geometry::{Geometry, FaceShape},
        domain::{Mesh, section::Section},                // Mesh and Section for fields
        input_output::mesh_generation::MeshGenerator,    // Correct reference for mesh generation
        solver::{KSP, gmres::GMRES, ksp::SolverResult},  // Solver abstraction, GMRES, and SolverResult
        domain::mesh_entity::MeshEntity,                 // Entities for boundary conditions
    };
    use faer::Mat;                                       // Matrix representation
    use std::error::Error;
    use rustc_hash::FxHashMap;

    /// Generate a 2D rectangular mesh for the Navier-Stokes problem.
    fn generate_mesh() -> Mesh {
        let width = 1.0;
        let height = 1.0;
        let nx = 3;
        let ny = 2;
        println!("Generating mesh with width {}, height {}, nx {}, ny {}", width, height, nx, ny);
        let mesh = MeshGenerator::generate_rectangle_2d(width, height, nx, ny);
    
        // Debug: Print vertex coordinates
        println!("Vertex Coordinates:");
        for (vertex_id, coords) in &mesh.vertex_coordinates {
            println!("Vertex ID: {}, Coordinates: {:?}", vertex_id, coords);
        }
    
        // Debug: Print all entities
        println!("Mesh Entities:");
        for entity in &mesh.entities {
            println!("{:?}", entity);
        }
    
        mesh
    }

    /// Apply boundary conditions by associating values with mesh entities.
    fn apply_boundary_conditions(mesh: &Mesh, section: &mut Section<f64>) {
        // Loop over the mesh entities and apply Dirichlet boundary conditions
        for entity in mesh.entities.iter() {
            match entity {
                MeshEntity::Vertex(id) => {
                    let coords = mesh.get_vertex_coordinates(*id).unwrap();
                    let x = coords[0];
                    let y = coords[1];
    
                    // Check if the vertex is on the boundary
                    if x == 0.0 || x == 1.0 || y == 0.0 || y == 1.0 {
                        // Calculate the boundary value from the exact solution
                        let u = 2.0 * x.powi(2) * y.powi(2);
                        section.set_data(*entity, u);
                    }
                }
                _ => {}
            }
        }
    }

    /// Initialize the KSP solver with required parameters.
    fn initialize_solver() -> GMRES {
        let max_iter = 100;  // Maximum number of iterations
        let tol = 1e-6;      // Tolerance for convergence
        let restart = 30;    // Restart parameter for GMRES
        println!("Initializing GMRES solver with max_iter = {}, tol = {}, restart = {}", max_iter, tol, restart);
        GMRES::new(max_iter, tol, restart)  // Initialize GMRES solver with the necessary parameters
    }

    fn compute_s_coefficients(mesh: &Mesh) -> FxHashMap<(usize, usize), f64> {
        let mut s_coeffs = FxHashMap::default();
    
        // For each vertex in the mesh
        for vertex in mesh.entities.iter().filter(|e| matches!(e, MeshEntity::Vertex(_))) {
            let node_id = vertex.id();
    
            // Get neighboring vertices
            let neighbors = mesh.get_neighboring_vertices(vertex);
    
            for neighbor in &neighbors {
                let neighbor_id = neighbor.id();
                // Avoid duplicate entries
                if node_id < neighbor_id {
                    // Compute S_{i,j}
                    let s_ij = compute_s_ij(mesh, vertex, neighbor);
    
                    // Store the coefficient
                    s_coeffs.insert((node_id, neighbor_id), s_ij);
                    s_coeffs.insert((neighbor_id, node_id), s_ij); // Since S_{i,j} = S_{j,i}
                }
            }
        }
        s_coeffs
    }

    fn compute_s_ij(mesh: &Mesh, entity_i: &MeshEntity, entity_j: &MeshEntity) -> f64 {
        // Compute the surface parameter S_{i,j} for the edge between nodes i and j
    
        // Get the coordinates of nodes i and j
        let coords_i = mesh.get_vertex_coordinates(entity_i.id()).expect("Coordinates not found for vertex_i");
        let coords_j = mesh.get_vertex_coordinates(entity_j.id()).expect("Coordinates not found for vertex_j");
    
        // Compute the vector between nodes i and j
        let dx = coords_j[0] - coords_i[0];
        let dy = coords_j[1] - coords_i[1];
    
        // Compute the length of the edge between nodes i and j
        let length = (dx.powi(2) + dy.powi(2)).sqrt();
    
        // For simplicity, set S_{i,j} = length
        let s_ij = length;
    
        s_ij
    }

    fn compute_control_volume_area(mesh: &Mesh, entity: &MeshEntity) -> f64 {
        // We assume 'entity' is a MeshEntity::Vertex
        // Compute the control volume area around this vertex using the geometry module
    
        // Get the neighboring cells (faces in 2D)
        let connected_faces = mesh.sieve.support(entity);
    
        if connected_faces.is_empty() {
            panic!("No connected faces found for entity {:?}", entity);
        }
    
        let mut control_volume_area = 0.0;
    
        for face in &connected_faces {
            // Get the face vertices
            let face_vertices_coords = mesh.get_face_vertices(face);
    
            // Determine the face shape
            let face_shape = match face_vertices_coords.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => panic!("Unsupported face shape with {} vertices", face_vertices_coords.len()),
            };
    
            // Compute the area of the face
            let geometry = Geometry::new();
            let face_area = geometry.compute_face_area(face_shape, &face_vertices_coords);
    
            // The control volume area contribution from this face is a fraction of the face area
            // For a vertex shared by 'n' vertices in the face, the fraction is 1 / n
            let fraction = 1.0 / (face_vertices_coords.len() as f64);
            control_volume_area += face_area * fraction;
        }
    
        control_volume_area
    }

    fn compute_source_term(coords: [f64; 3]) -> f64 {
        let x = coords[0];
        let y = coords[1];
        // From the exact solution u = 2x^2 y^2, compute ∇²u to get f(x,y)
        // Compute the Laplacian of u analytically:
        // ∇²u = ∂²u/∂x² + ∂²u/∂y² = 4y² + 4x²
    
        let f = 4.0 * y.powi(2) + 4.0 * x.powi(2);
        f
    }

    /// Assemble the system matrix and RHS vector using the Section and mesh.
    fn assemble_system(section: &Section<f64>, mesh: &Mesh) -> (Mat<f64>, Vec<f64>) {
        use std::collections::HashMap;
        let num_entities = section.entities().len();
        let mut a = Mat::<f64>::zeros(num_entities, num_entities);  // System matrix
        let mut b = vec![0.0; num_entities];  // Right-hand side vector
    
        // Map MeshEntity to index
        let entity_indices: HashMap<MeshEntity, usize> = section.entities()
            .iter()
            .enumerate()
            .map(|(i, e)| (*e, i))
            .collect();
    
        // Compute S coefficients
        let s_coeffs = compute_s_coefficients(mesh);
    
        // For each node (vertex)
        for entity in section.entities() {
            if let MeshEntity::Vertex(node_id) = entity {
                let i = entity_indices[&entity];
    
                // Check if the node is on the boundary
                let coords = mesh.get_vertex_coordinates(node_id).unwrap();
                let x = coords[0];
                let y = coords[1];
    
                if x == 0.0 || x == 1.0 || y == 0.0 || y == 1.0 {
                    // Boundary node, Dirichlet condition
                    a.write(i, i, 1.0);
                    b[i] = *section.restrict(&entity).unwrap();
                } else {
                    // Interior node
                    // Sum over neighboring nodes
                    let mut sum_s = 0.0;

                    // Get neighboring vertices
                    let neighbors = mesh.get_neighboring_vertices(&entity);

                    if neighbors.is_empty() {
                        panic!("No neighboring nodes found for entity {:?}", entity);
                    }

                    for neighbor in &neighbors {
                        let j = entity_indices[neighbor];
                        let s_ij = *s_coeffs.get(&(node_id, neighbor.id())).unwrap_or(&0.0);
                        sum_s += s_ij;

                        a.write(i, j, -s_ij);
                    }

                    // Diagonal term
                    a.write(i, i, sum_s);
    
                    // Compute the area A_i (control volume area)
                    let a_i = compute_control_volume_area(mesh, &entity);
    
                    // Source term f_i (from exact solution)
                    let f_i = compute_source_term(coords);
    
                    // Right-hand side
                    b[i] = f_i * a_i;
                }
            }
        }
    
        (a, b)
    }

    fn run_poisson_equation_simulation() -> Result<(), Box<dyn Error>> {
        // Step 1: Generate the mesh
        let mesh = generate_mesh();  // Generate a 2D mesh
    
        // Step 2: Create a Section for the solution field
        let mut solution_section = Section::<f64>::new();  // Initialize Section for the solution
    
        // Step 3: Apply boundary conditions
        apply_boundary_conditions(&mesh, &mut solution_section);
    
        // Step 4: Assemble the system matrix (A) and the right-hand side vector (b)
        let (a, b) = assemble_system(&solution_section, &mesh);
    
        // Initialize a solution vector
        let mut x = vec![0.0; b.len()];  // Initial guess for the solution
    
        // Step 5: Initialize the GMRES solver
        let mut solver = initialize_solver();
    
        // Solve the system using GMRES
        let result: SolverResult = solver.solve(&a, &b, &mut x);
    
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
    
        // Update the solution section with new solution values
        println!("Updating solution section with new solution values.");
        for (entity, &value) in solution_section.entities().iter().zip(x.iter()) {
            solution_section.update_data(entity, value).expect("Failed to update solution");
            println!("Updated entity {:?} with value {}", entity, value);
        }
    
        // Optional: Compare the numerical solution with the exact solution
        for entity in solution_section.entities() {
            let numerical = *solution_section.restrict(&entity).unwrap();
            if let MeshEntity::Vertex(node_id) = entity {
                let coords = mesh.get_vertex_coordinates(node_id).unwrap();
                let x = coords[0];
                let y = coords[1];
                let exact = 2.0 * x.powi(2) * y.powi(2);
                println!("Node {:?}: Numerical = {}, Exact = {}, Error = {}", node_id, numerical, exact, (numerical - exact).abs());
            }
        }
    
        Ok(())
    }

    /// Integration test for Chung's 2010 CFD Example 7.2.1.
    #[test]
    fn test_chung_example_7_2_1() {
        // Run the simulation and check for success
        println!("Starting integration test for Chung's 2010 CFD Example 7.2.1");
        assert!(run_poisson_equation_simulation().is_ok());
    }
}
