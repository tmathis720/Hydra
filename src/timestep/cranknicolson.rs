use crate::domain::Mesh;
use crate::solver::flux_solver::FluxSolver;

/// Crank-Nicolson time-stepping method.
pub struct CrankNicolson {
    pub solver: FluxSolver,  // Holds the specific solver for flux computations
}

impl CrankNicolson {
    fn _step(&self, mesh: &mut Mesh, dt: f64) {
        // First, collect all the data we need from the face-element relations
        let relations_data: Vec<(u32, u32, u32)> = mesh.face_element_relations
            .iter()
            .map(|relation| (relation.face_id, relation.connected_elements[0], relation.connected_elements[1]))
            .collect();

        // Now, iterate over the collected data and apply the necessary changes
        for (face_id, left_id, right_id) in relations_data {
            // First, borrow elements and faces immutably to compute the average flux
            let flux_3d_avg = {
                let left = mesh.get_element_by_id(left_id).expect("element");
                let right = mesh.get_element_by_id(right_id).expect("element");
                let face = mesh.get_face_by_id(face_id).expect("face");

                // Compute old and new fluxes and average them
                let flux_3d_old = self.solver.compute_flux_3d(face, left, right);
                let flux_3d_new = self.solver.compute_flux_3d(face, left, right);
                (flux_3d_old + flux_3d_new) / 2.0
            };

            // Now, borrow the face and elements mutably and apply the flux
            if let Some(face) = mesh.get_face_by_id_mut(face_id) {
                self.solver.apply_flux_3d(face, flux_3d_avg, dt);
            }
            if let Some(left) = mesh.get_element_by_id_mut(left_id) {
                left.update_momentum(flux_3d_avg * dt);
            }
            if let Some(right) = mesh.get_element_by_id_mut(right_id) {
                right.update_momentum(-flux_3d_avg * dt);  // Opposite direction
            }
        }

        /* // This is responsible for solving the momentum equation
        for element in &mut mesh.elements {
            let old_velocity = element.velocity;

            // Compute the new velocity using the Crank-Nicolson scheme
            let momentum_flux = self.solver.compute_flux(element, flow_field);
            let new_velocity = old_velocity + (momentum_flux * dt);

            element.velocity = new_velocity;
        } */

        // Update velocities of all elements based on the new momentum
        for element in &mut mesh.elements {
            element.update_velocity_from_momentum();
        }
    }
}

