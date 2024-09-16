use crate::domain::Mesh;
use crate::solver::flux_solver::FluxSolver;

/// Explicit Euler time-stepping method.
pub struct ExplicitEuler {
    pub solver: FluxSolver,  // Holds the specific solver for flux computations
}

impl ExplicitEuler {
    fn _step(&self, mesh: &mut Mesh, dt: f64) {
        // First, collect all the data we need from the face-element relations
        let relations_data: Vec<(u32, u32, u32)> = mesh.face_element_relations
            .iter()
            .map(|relation| (relation.face_id, relation.connected_elements[0], relation.connected_elements[1]))
            .collect();

        // Now, iterate over the collected data and apply the necessary changes
        for (face_id, left_id, right_id) in relations_data {
            // First, borrow elements and faces immutably to compute the flux
            let flux_3d = {
                let left = mesh.get_element_by_id(left_id).expect("element");
                let right = mesh.get_element_by_id(right_id).expect("element");
                let face = mesh.get_face_by_id(face_id).expect("face");
                self.solver.compute_flux_3d(face, left, right)
            };

            // Now, borrow the face and elements mutably and apply the flux
            if let Some(face) = mesh.get_face_by_id_mut(face_id) {
                self.solver.apply_flux_3d(face, flux_3d, dt);
            }
            if let Some(left) = mesh.get_element_by_id_mut(left_id) {
                left.update_momentum(flux_3d * dt);
            }
            if let Some(right) = mesh.get_element_by_id_mut(right_id) {
                right.update_momentum(-flux_3d * dt);  // Opposite direction
            }
        }

        // Update velocities of all elements based on the new momentum
        for element in &mut mesh.elements {
            let _ = element.update_velocity_from_momentum();
        }
    }
}
