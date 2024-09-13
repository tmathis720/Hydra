// src/timestep/euler.rs

use crate::domain::Mesh;
use crate::solver::flux_solver::FluxSolver;

/// Explicit Euler time-stepping method.
pub struct ExplicitEuler {
    pub solver: FluxSolver,  // Holds the specific solver for flux computations
}

impl ExplicitEuler {
    fn step(&self, mesh: &mut Mesh, dt: f64) {
        for face in &mut mesh.faces {
            // Get mutable references to the connected elements
            let (left_element, right_element) = mesh.get_connected_elements(face);

            if let (Some(left), Some(right)) = (left_element, right_element) {
                // Use the specific solver to compute fluxes
                let flux_3d = self.solver.compute_flux_3d(face, left, right);

                // Apply the flux to update the face velocity
                self.solver.apply_flux_3d(face, flux_3d, dt);

                // Update momentum of the elements explicitly
                left.update_momentum(flux_3d * dt);
                right.update_momentum(-flux_3d * dt);  // Opposite direction
            }
        }

        // Update velocities of all elements based on the new momentum
        for element in &mut mesh.elements {
            element.update_velocity_from_momentum();
        }
    }
}
