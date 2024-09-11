use crate::timestep::TimeStepper;
use crate::domain::mesh::Mesh;
use crate::solver::Solver;

pub struct CrankNicolson {
    pub dt: f64,
}

impl TimeStepper for CrankNicolson {
    fn step(&self, mesh: &mut Mesh, solver: &mut dyn Solver) {
        for relation in &mesh.face_element_relations {
            let left_element = &mesh.elements[relation.left_element_id as usize];
            let right_element = &mesh.elements[relation.right_element_id as usize];

            let face = &mut mesh.faces[relation.face_id as usize];

            let flux = solver.compute_flux(face, left_element, right_element);
            solver.apply_flux(face, flux, self.dt);
        }
    }
}
