use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    geometry::Geometry, Mesh,
};
use super::{
    fields::{Fields, Fluxes},
    PhysicalEquation,
};
use crate::domain::section::{Vector3, Scalar};

pub struct MomentumParameters {
    pub density: f64,
    pub viscosity: f64,
}

pub struct MomentumEquation {
    pub params: MomentumParameters,
}

impl PhysicalEquation for MomentumEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_momentum_fluxes(domain, fields, fluxes, boundary_handler, current_time);
    }
}

impl MomentumEquation {
    pub fn calculate_momentum_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        let _ = current_time;
        let mut _geometry = Geometry::new();

        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                // Get the cells adjacent to the face
                let cells = domain.get_cells_sharing_face(&face);

                // Initialize variables
                let mut velocity_a = Vector3([0.0; 3]);
                let mut pressure_a = Scalar(0.0);
                let mut velocity_b = Vector3([0.0; 3]);
                let mut pressure_b = Scalar(0.0);

                let mut has_cell_b = false;

                // Iterate over adjacent cells
                let mut iter = cells.iter();
                if let Some(cell_entry) = iter.next() {
                    let cell_a = cell_entry.key().clone();
                    if let Some(vel) = fields.get_vector_field_value("velocity_field", &cell_a) {
                        velocity_a = vel;
                    }
                    if let Some(pres) = fields.get_scalar_field_value("pressure_field", &cell_a) {
                        pressure_a = pres;
                    }
                }
                if let Some(cell_entry) = iter.next() {
                    let cell_b = cell_entry.key().clone();
                    has_cell_b = true;
                    if let Some(vel) = fields.get_vector_field_value("velocity_field", &cell_b) {
                        velocity_b = vel;
                    }
                    if let Some(pres) = fields.get_scalar_field_value("pressure_field", &cell_b) {
                        pressure_b = pres;
                    }
                }

                // Compute convective flux
                let avg_velocity = if has_cell_b {
                    Vector3([
                        0.5 * (velocity_a.0[0] + velocity_b.0[0]),
                        0.5 * (velocity_a.0[1] + velocity_b.0[1]),
                        0.5 * (velocity_a.0[2] + velocity_b.0[2]),
                    ])
                } else {
                    velocity_a
                };

                let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                let convective_flux = self.params.density * velocity_dot_normal * area;

                // Compute pressure flux
                let pressure_flux = if has_cell_b {
                    0.5 * (pressure_a.0 + pressure_b.0) * area
                } else {
                    pressure_a.0 * area
                };

                // Compute diffusive flux (simplified for demonstration)
                // In practice, this would involve gradients of velocity
                let diffusive_flux = self.params.viscosity * area;

                // Total flux vector (assuming 3D for demonstration)
                let total_flux = Vector3([
                    convective_flux - pressure_flux + diffusive_flux,
                    0.0,
                    0.0,
                ]);

                // Update fluxes
                fluxes.add_momentum_flux(face.clone(), total_flux);

                // Apply boundary conditions
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(_value) => {
                            // Apply Dirichlet condition
                            // Adjust fluxes or impose values as necessary
                        }
                        BoundaryCondition::Neumann(_value) => {
                            // Apply Neumann condition
                            // Modify fluxes accordingly
                        }
                        _ => (),
                    }
                }
            }
        }
    }
}