use crate::domain::mesh::Mesh;
use crate::solver::FluxSolver;

pub struct ExplicitEuler {
    pub dt: f64,  // Time step size
}

impl ExplicitEuler {
    pub fn step(&self, mesh: &mut Mesh, flux_solver: &mut FluxSolver) {
        // Step 1: Collect all face-element references from the relationship table
        for relation in &mesh.face_element_relations {
            let left_element = &mesh.elements[relation.left_element_id as usize];
            let right_element = &mesh.elements[relation.right_element_id as usize];

            // Step 2: Borrow the face mutably
            let face = &mut mesh.faces[relation.face_id as usize];

            // Step 3: Compute and apply flux
            let flux = flux_solver.compute_flux(face, left_element, right_element);
            flux_solver.apply_flux(face, flux, self.dt);
        }
    }
}
