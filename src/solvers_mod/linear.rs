
use crate::mesh_mod::mesh_ops::Mesh;
use crate::transport_mod::flux_ops::FluxCalculator;

pub struct LinearSolver {
    pub mesh: Mesh,         // The mesh containing elements
    pub tol: f64,           // Convergence tolerance
    pub max_iter: usize,    // Maximum number of iterations
    //pub preconditioner: Option<Preconditioner>,// Placeholder for future use
}

impl LinearSolver {
    // Create a new LinearSolver with a given mesh
    pub fn new(mesh: Mesh, tol: f64, max_iter: usize) -> Self {
        LinearSolver {
            mesh,
            tol,
            max_iter,
            //preconditioner: None, // Set a default preconditioner here
        }
    }

    pub fn solve(&mut self) {
        let mut iter = 0;
        let mut residual = self.compute_residual();

        // Iterate until convergence or maximum iterations
        while residual > self.tol && iter < self.max_iter {
            self.step();
            residual = self.compute_residual(); // Update residual
            iter += 1;

            // Optionally print iteration details
            println!("Iteration: {}, Residual: {}", iter, residual);
        }

        if residual <= self.tol {
            println!("Solver converged in {} iterations", iter);
        } else {
            println!("Solver did not converge within the maximum iterations.");
        }
    }
    
    pub fn compute_residual(&self) -> f64 {
        // Dummy residual calculation
        let mut residual = 0.0;
        for element in &self.mesh.elements {
            residual += (element.state - 0.0).abs();
        }
        residual
    }

    pub fn step(&mut self) {
        for element in &self.mesh.elements {
            for &neighbor_id in &element.neighbors {
                let neighbor = &self.mesh.elements[neighbor_id - 1];  // Assuming 1-based indexing
                let flux = FluxCalculator::compute_element_flux(element, neighbor);
                // Use the flux for something (e.g., update element state)
            }
        }
    }

    /// Perform a flux computation and return the updated state without advancing time
    pub fn compute_fluxes(&self) -> Vec<f64> {
        let mut new_states = vec![0.0; self.mesh.elements.len()];  // Initialize the new_states vector

        for (i, element) in self.mesh.elements.iter().enumerate() {
            let mut total_flux = 0.0;

            for &neighbor_id in &element.neighbors {
                // Correct neighbor indexing
                let neighbor = &self.mesh.elements[neighbor_id - 1];  // Assuming 1-based indexing
                let flux = FluxCalculator::compute_element_flux(element, neighbor);
                total_flux += flux;
            }

            new_states[i] = total_flux;  // Save the computed flux for this element
        }

        new_states  // Return the new state vector
    }

    pub fn update_states(&mut self, new_states: Vec<f64>, dt: f64) {
        for (i, element) in self.mesh.elements.iter_mut().enumerate() {
            element.state += new_states[i] * dt;  // state update based on flux and time step
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_mod::mesh_ops::Mesh;


    /* #[test]
    fn test_linear_solver_convergence() {
        let mut mesh = Mesh::new();

        // Create a basic 2-element mesh
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);

        mesh.add_element(1, [1, 2, 3], vec![1, 2]);
        mesh.add_element(2, [2, 3, 4], vec![1, 2]);

        mesh.elements[0].state = 10.0;
        mesh.elements[1].state = 5.0;

        mesh.build_neighbors();

        // Create a solver with tolerance and max iterations
        let mut solver = LinearSolver::new(mesh, 1e-6, 100);
        
        // Perform the solve
        solver.solve();

        // Check that the solver converges (or doesn't exceed max iterations)
        assert!(solver.compute_residual() < solver.tol, "Solver did not converge");
    } */


    #[test]
    fn test_flux_computation() {
        let mut mesh = Mesh::new();
        let mut tol = 0.01;
        let mut max_iter = 100;
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        mesh.add_element(2, [2, 4, 3], vec![99, 2]);
        mesh.elements[0].state = 10.0;
        mesh.elements[1].state = 5.0;
        mesh.build_neighbors();

        let solver = LinearSolver::new(mesh, tol, max_iter);
        let fluxes = solver.compute_fluxes();

        assert_eq!(fluxes.len(), 2);
    }

    /* #[test]
    fn test_linear_solver_flux_computation() {
        let mut mesh = Mesh::new();

        let mut tol = 0.01;
        let mut max_iter = 100;
        // Adding nodes to the mesh (forming two triangles)
        mesh.add_node(1, 0.0, 0.0, 0.0); // Node 1
        mesh.add_node(2, 1.0, 0.0, 0.0); // Node 2
        mesh.add_node(3, 0.0, 1.0, 0.0); // Node 3
        mesh.add_node(4, 1.0, 1.0, 0.0); // Node 4

        // Adding two elements (triangles)
        mesh.add_element(1, [1, 2, 3], vec![1, 2]); // Element 1
        mesh.add_element(2, [2, 3, 4], vec![1, 2]); // Element 2

        // Setting states for each element
        mesh.elements[0].state = 10.0; // Element 1 state
        mesh.elements[1].state = 5.0;  // Element 2 state

        // Building neighbor relationships between elements
        mesh.build_neighbors();

        // Initialize the solver
        let solver = LinearSolver::new(mesh, tol, max_iter);

        // Compute fluxes
        let fluxes = solver.compute_fluxes();

        // Output flux values for debugging
        println!("Fluxes: {:?}", fluxes);

        // Verify the size of the flux array
        assert_eq!(fluxes.len(), 2, "Expected fluxes for two elements.");

        // Validate the expected fluxes based on a diffusion-like formula
        // Assuming a simple model where flux is proportional to the state difference and area
        // State difference between Element 1 and Element 2 is 10.0 - 5.0 = 5.0
        let expected_flux_element_1 = -5.0 * solver.mesh.elements[0].area; // Flux from element 1 to element 2
        let expected_flux_element_2 = 5.0 * solver.mesh.elements[1].area;  // Flux from element 2 to element 1

        assert!((fluxes[0] - expected_flux_element_1).abs() < 1e-6, 
                "Flux mismatch for element 1. Expected: {}, Got: {}", 
                expected_flux_element_1, fluxes[0]);
        assert!((fluxes[1] - expected_flux_element_2).abs() < 1e-6, 
                "Flux mismatch for element 2. Expected: {}, Got: {}", 
                expected_flux_element_2, fluxes[1]);
    } */

}

