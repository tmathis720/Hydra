use crate::domain::Mesh;
use crate::solver::Solver;
use crate::timestep::TimeStepper;

pub struct ExplicitEuler {
    pub dt: f64,  // Time step size
}

impl TimeStepper for ExplicitEuler {
    fn step(&self, mesh: &mut Mesh, solver: &mut dyn Solver) {
        // Step 1: Collect all face-element references from the relationship table
        for relation in &mesh.face_element_relations {
            let left_element = &mesh.elements[relation.left_element_id as usize];
            let right_element = &mesh.elements[relation.right_element_id as usize];

            // Step 2: Borrow the face mutably
            let face = &mut mesh.faces[relation.face_id as usize];

            // Step 3: Compute and apply flux
            let flux = solver.compute_flux(face, left_element, right_element);
            solver.apply_flux(face, flux, self.dt);
        }
    }
}
