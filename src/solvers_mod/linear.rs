
use crate::mesh_mod::mesh_ops::Mesh;
use crate::transport_mod::flux_ops::FluxCalculator;

pub struct LinearSolver {
    pub mesh: Mesh,
}

impl LinearSolver {
    // Create a new LinearSolver with a given mesh
    pub fn new(mesh: Mesh) -> Self {
        LinearSolver { mesh }
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

    #[test]
    fn test_flux_computation() {
        let mut mesh = Mesh::new();
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);
        mesh.add_element(1, [1, 2, 3], vec![99, 2]);
        mesh.add_element(2, [2, 4, 3], vec![99, 2]);
        mesh.elements[0].state = 10.0;
        mesh.elements[1].state = 5.0;
        mesh.build_neighbors();

        let solver = LinearSolver::new(mesh);
        let fluxes = solver.compute_fluxes();

        assert_eq!(fluxes.len(), 2);
    }

    #[test]
    fn test_linear_solver_flux_computation() {
        let mut mesh = Mesh::new();
    
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);
    
        mesh.add_element(1, [1, 2, 3], vec![1, 2]);
        mesh.add_element(2, [2, 3, 4], vec![1, 2]);
    
        mesh.elements[0].state = 10.0;
        mesh.elements[1].state = 5.0;
    
        mesh.build_neighbors();
    
        let solver = LinearSolver::new(mesh);
        let fluxes = solver.compute_fluxes();
    
        // Debugging output to help diagnose issues
        println!("Fluxes: {:?}", fluxes);
        println!("Element 1 Neighbors: {:?}", solver.mesh.elements[0].neighbors);
        println!("Element 2 Neighbors: {:?}", solver.mesh.elements[1].neighbors);
    
        assert_eq!(fluxes.len(), 2);
        assert!((fluxes[0] - -2.5).abs() < 1e-6, "Flux mismatch for element 1");
        assert!((fluxes[1] - 2.5).abs() < 1e-6, "Flux mismatch for element 2");
    }
}

