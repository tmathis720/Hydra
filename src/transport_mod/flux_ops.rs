
use crate::mesh_mod::mesh_ops::Mesh;
use crate::mesh_mod::element_ops::Element;

pub struct FluxCalculator;

impl FluxCalculator {
    // Computes the flux between elements and stores it in a vector
    pub fn compute_fluxes(mesh: &Mesh) -> Vec<f64> {
        let mut fluxes = vec![0.0; mesh.elements.len()];

        for (i, element) in mesh.elements.iter().enumerate() {
            let mut total_flux = 0.0;
            
            for &neighbor_id in &element.neighbors {
                let neighbor = &mesh.elements[neighbor_id - 1];
                let flux = Self::compute_element_flux(element, neighbor);
                total_flux += flux;
            }

            fluxes[i] = total_flux;
        }

        fluxes
    }

    // Simple diffusion-based flux calculation between two elements
    pub fn compute_element_flux(element: &Element, neighbor: &Element) -> f64 {
        let diff = element.state - neighbor.state;
        let avg_area = (element.area + neighbor.area) / 2.0;
        
        // Flux is proportional to the difference in state and the average area
        diff * avg_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_mod::mesh_ops::Mesh;
    use crate::transport_mod::flux_ops::FluxCalculator;

    #[test]
    fn test_flux_calculation() {
        // Create a simple mock mesh with 2 elements
        let mut mesh = Mesh::new();

        // Add nodes (for reference in element creation)
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 0.0, 1.0, 0.0);
        mesh.add_node(4, 1.0, 1.0, 0.0);

        // Add two triangular elements
        mesh.add_element(1, [1, 2, 3], vec![1]);
        mesh.add_element(2, [2, 4, 3], vec![1]);

        // Assign some arbitrary state values to each element
        mesh.elements[0].state = 10.0;
        mesh.elements[1].state = 5.0;

        // Set up neighbor relationships manually for this test
        mesh.elements[0].neighbors = vec![2]; // Element 1's neighbor is Element 2
        mesh.elements[1].neighbors = vec![1]; // Element 2's neighbor is Element 1

        // Compute fluxes using the FluxCalculator
        let fluxes = FluxCalculator::compute_fluxes(&mesh);

        // Check that the computed flux matches the expected value (based on the state difference)
        assert!((fluxes[0] - 7.5).abs() < 1e-6, "Flux mismatch for element 1");
        assert!((fluxes[1] + 7.5).abs() < 1e-6, "Flux mismatch for element 2");
    }
}
