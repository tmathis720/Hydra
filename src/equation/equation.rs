use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::BoundaryConditionHandler;
use crate::domain::section::{Vector3, Scalar};

pub struct Equation {}

impl Equation {
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        velocity_field: &Section<Vector3>,
        pressure_field: &Section<Scalar>,
        fluxes: &mut Section<Vector3>,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64, // Accept current_time as a parameter
    ) {
        let _ = pressure_field;
        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                let velocity_dot_normal = velocity_field
                    .restrict(&face)
                    .map(|vel| vel.iter().zip(normal.iter()).map(|(v, n)| v * n).sum::<f64>())
                    .unwrap_or(0.0);

                let flux = Vector3([velocity_dot_normal * area, 0.0, 0.0]);
                fluxes.set_data(face.clone(), flux);

                // Boundary condition logic
                let mut matrix_storage = faer::Mat::<f64>::zeros(1, 1);
                let mut rhs_storage = faer::Mat::<f64>::zeros(1, 1);
                let mut matrix = matrix_storage.as_mut();
                let mut rhs = rhs_storage.as_mut();
                let boundary_entities = boundary_handler.get_boundary_faces();
                let entity_to_index = domain.get_entity_to_index();

                boundary_handler.apply_bc(
                    &mut matrix,
                    &mut rhs,
                    &boundary_entities,
                    &entity_to_index,
                    current_time, // Pass current_time
                );
            }
        }
    }
}
