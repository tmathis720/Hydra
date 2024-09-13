// src/timestep/cranknicolson.rs

use crate::domain::Mesh;
use crate::solver::flux_solver::FluxSolver;

/// Crank-Nicolson time-stepping method.
pub struct CrankNicolson {
    pub solver: FluxSolver,  // Holds the specific solver for flux computations
}

impl CrankNicolson {
    fn step(&self, mesh: &mut Mesh, dt: f64) {
        for face in &mut mesh.faces {
            // Get mutable references to the connected elements
            let (left_element, right_element) = mesh.get_connected_elements(face);

            if let (Some(left), Some(right)) = (left_element, right_element) {
                // Compute the flux at the current time step
                let flux_3d_old = self.solver.compute_flux_3d(face, left, right);

                // Apply half the time step using the old flux (semi-implicit)
                self.solver.apply_flux_3d(face, flux_3d_old, dt / 2.0);

                // Recompute the flux with the updated values (new time step)
                let flux_3d_new = self.solver.compute_flux_3d(face, left, right);

                // Apply the full time step using the average of the old and new fluxes
                let flux_3d_avg = (flux_3d_old + flux_3d_new) / 2.0;
                self.solver.apply_flux_3d(face, flux_3d_avg, dt);

                // Update momentum of the elements using the average flux
                left.update_momentum(flux_3d_avg * dt);
                right.update_momentum(-flux_3d_avg * dt);  // Opposite direction
            }
        }

        // Update velocities of all elements based on the new momentum
        for element in &mut mesh.elements {
            element.update_velocity_from_momentum();
        }
    }
}
