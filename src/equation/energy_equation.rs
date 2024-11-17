use crate::equation::PhysicalEquation;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::geometry::{Geometry, FaceShape};
use crate::domain::section::{Scalar, Vector3};
use crate::Mesh;

use super::fields::{Fields, Fluxes};

pub struct EnergyEquation {
    pub thermal_conductivity: f64, // Coefficient for thermal conduction
}

impl PhysicalEquation for EnergyEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_energy_fluxes(
            domain,
            fields,
            fluxes,
            boundary_handler,
            current_time,
        );
    }
}

impl EnergyEquation {
    pub fn new(thermal_conductivity: f64) -> Self {
        EnergyEquation { thermal_conductivity }
    }

    pub fn calculate_energy_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        _current_time: f64,
    ) {
        let mut geometry = Geometry::new();

        for face in domain.get_faces() {
            let face_vertices = domain.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue, // Skip unsupported face shapes
            };
            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

            let cells = domain.get_cells_sharing_face(&face);
            let cell_a = cells
                .iter()
                .next()
                .map(|entry| entry.key().clone())
                .expect("Face should have at least one associated cell.");

            let temp_a = fields.get_scalar_field_value("temperature", &cell_a)
                .expect("Temperature not found for cell");
            let grad_temp_a = fields.get_vector_field_value("temperature_gradient", &cell_a)
                .expect("Temperature gradient not found for cell");

            let mut face_temperature = self.reconstruct_face_value(
                temp_a,
                grad_temp_a,
                geometry.compute_cell_centroid(domain, &cell_a),
                face_center,
            );

            let velocity = fields.get_vector_field_value("velocity", &face)
                .expect("Velocity not found at face");
            let face_normal = geometry
                .compute_face_normal(domain, &face, &cell_a)
                .expect("Normal not found for face");

            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);

            let total_flux;

            if cells.len() == 1 {
                // Boundary face
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            face_temperature = Scalar(value);

                            // Recompute conductive flux based on temperature difference
                            let cell_centroid = geometry.compute_cell_centroid(domain, &cell_a);
                            let distance =
                                Geometry::compute_distance(&cell_centroid, &face_center);

                            let temp_gradient_normal = (face_temperature.0 - temp_a.0) / distance;
                            let face_normal_length = face_normal.0
                                .iter()
                                .map(|n| n * n)
                                .sum::<f64>()
                                .sqrt();

                            let conductive_flux = -self.thermal_conductivity
                                * temp_gradient_normal
                                * face_normal_length;

                            // Compute convective flux
                            let vel_dot_n = velocity.0
                                .iter()
                                .zip(&face_normal.0)
                                .map(|(v, n)| v * n)
                                .sum::<f64>();
                            let rho = 1.0;
                            let convective_flux = rho * face_temperature.0 * vel_dot_n;

                            total_flux = Scalar((conductive_flux + convective_flux) * face_area);
                        }
                        BoundaryCondition::Neumann(flux) => {
                            total_flux = Scalar(flux * face_area);
                        }
                        _ => {
                            total_flux = self.compute_flux(
                                temp_a,
                                face_temperature,
                                &grad_temp_a,
                                &face_normal,
                                &velocity,
                                face_area,
                            );
                        }
                    }
                } else {
                    total_flux = self.compute_flux(
                        temp_a,
                        face_temperature,
                        &grad_temp_a,
                        &face_normal,
                        &velocity,
                        face_area,
                    );
                }
            } else {
                // Internal face
                total_flux = self.compute_flux(
                    temp_a,
                    face_temperature,
                    &grad_temp_a,
                    &face_normal,
                    &velocity,
                    face_area,
                );
            }

            fluxes.add_energy_flux(face, total_flux);
        }
    }

    fn reconstruct_face_value(
        &self,
        cell_value: Scalar,
        cell_gradient: Vector3,
        cell_centroid: [f64; 3],
        face_center: [f64; 3],
    ) -> Scalar {
        let dx = face_center[0] - cell_centroid[0];
        let dy = face_center[1] - cell_centroid[1];
        let dz = face_center[2] - cell_centroid[2];

        Scalar(
            cell_value.0 + cell_gradient.0[0] * dx + cell_gradient.0[1] * dy + cell_gradient.0[2] * dz,
        )
    }

    fn compute_flux(
        &self,
        _temp_a: Scalar,
        face_temperature: Scalar,
        grad_temp_a: &Vector3,
        face_normal: &Vector3,
        velocity: &Vector3,
        face_area: f64,
    ) -> Scalar {
        let conductive_flux = -self.thermal_conductivity
            * (grad_temp_a.0[0] * face_normal.0[0]
                + grad_temp_a.0[1] * face_normal.0[1]
                + grad_temp_a.0[2] * face_normal.0[2]);

        let rho = 1.0;
        let convective_flux = rho
            * face_temperature.0
            * (velocity.0[0] * face_normal.0[0]
                + velocity.0[1] * face_normal.0[1]
                + velocity.0[2] * face_normal.0[2]);

        Scalar((conductive_flux + convective_flux) * face_area)
    }
}