// src/equation/equation.rs

use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::equation::gradient::gradient_calc::Gradient;
use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
use crate::equation::reconstruction::reconstruct::reconstruct_face_value;
use crate::geometry::Geometry;

pub struct Equation {
    // Define any necessary fields, such as parameters or constants
}

impl Equation {
    /// Calculates fluxes at cell faces using TVD upwinding
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        field: &Section<f64>,
        gradient: &Section<[f64; 3]>,
        velocity_field: &Section<[f64; 3]>,
        fluxes: &mut Section<f64>,
    ) {
        let mesh = &domain.mesh;
        let boundary_handler = &domain.boundary_handler;
        let geometry = Geometry::new();

        for face in mesh.get_faces() {
            // Get the cells sharing this face
            let neighbor_cells = mesh.get_cells_sharing_face(&face);
            let cells: Vec<_> = neighbor_cells.iter().map(|entry| entry.key().clone()).collect();

            // Get the face normal and area
            let face_vertices = mesh.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => crate::FaceShape::Triangle,
                4 => crate::FaceShape::Quadrilateral,
                _ => continue, // Skip unsupported face shapes
            };

            let face_normal = geometry.compute_face_normal(mesh, &face, &cells[0]).unwrap();
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);

            // Compute the unit normal vector
            let mut normal = face_normal;
            let normal_length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            for i in 0..3 {
                normal[i] /= normal_length;
            }

            // Initialize variables
            let mut left_value = 0.0;
            let mut right_value = 0.0;
            let mut velocity = 0.0;

            // Handle internal faces
            if cells.len() == 2 {
                let cell_left = &cells[0];
                let cell_right = &cells[1];

                // Get cell values and gradients
                let phi_left = field.restrict(cell_left).unwrap();
                let grad_left = gradient.restrict(cell_left).unwrap();
                let phi_right = field.restrict(cell_right).unwrap();
                let grad_right = gradient.restrict(cell_right).unwrap();

                // Get cell centers and face center
                let cell_left_center = geometry.compute_cell_centroid(mesh, cell_left);
                let cell_right_center = geometry.compute_cell_centroid(mesh, cell_right);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face values from both sides
                left_value = reconstruct_face_value(*phi_left, *grad_left, cell_left_center, face_center);
                right_value = reconstruct_face_value(*phi_right, *grad_right, cell_right_center, face_center);

                // Compute velocity at the face (e.g., average of cell velocities)
                let vel_left = velocity_field.restrict(cell_left).unwrap();
                let vel_right = velocity_field.restrict(cell_right).unwrap();

                // Project velocity onto face normal to get normal component
                let vel_normal_left = vel_left[0] * normal[0] + vel_left[1] * normal[1] + vel_left[2] * normal[2];
                let vel_normal_right = vel_right[0] * normal[0] + vel_right[1] * normal[1] + vel_right[2] * normal[2];

                // Average velocity normal component
                velocity = 0.5 * (vel_normal_left + vel_normal_right);
            } else if cells.len() == 1 {
                // Boundary face
                let cell_left = &cells[0];
                let phi_left = field.restrict(cell_left).unwrap();
                let grad_left = gradient.restrict(cell_left).unwrap();
                let cell_left_center = geometry.compute_cell_centroid(mesh, cell_left);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face value from inside the domain
                left_value = reconstruct_face_value(*phi_left, *grad_left, cell_left_center, face_center);

                // Handle boundary condition to get right_value
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            right_value = *value;
                        }
                        BoundaryCondition::Neumann(_) => {
                            // For Neumann BC, we need special handling (omitted for brevity)
                            right_value = left_value; // Simplification; may need adjustment
                        }
                        _ => {
                            // Handle other boundary conditions as needed
                            right_value = left_value; // Default to left_value
                        }
                    }
                } else {
                    // No boundary condition specified; default to left_value
                    right_value = left_value;
                }

                // Get cell velocity and project onto face normal
                let vel_left = velocity_field.restrict(cell_left).unwrap();
                velocity = vel_left[0] * normal[0] + vel_left[1] * normal[1] + vel_left[2] * normal[2];
            } else {
                // No cells associated with face; skip
                continue;
            }

            // Compute upwind flux
            let upwind_value = Self::compute_upwind_flux(left_value, right_value, velocity);

            // Compute flux (e.g., flux = upwind_value * velocity * face_area)
            let flux = upwind_value * velocity * face_area;

            // Store or accumulate fluxes as needed
            fluxes.set_data(face.clone(), flux);
        }
    }

    /// Computes the upwind flux based on the flow direction (velocity).
    pub fn compute_upwind_flux(left_value: f64, right_value: f64, velocity: f64) -> f64 {
        if velocity >= 0.0 {
            left_value
        } else {
            right_value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_upwind_flux() {
        let left_value = 1.0;
        let right_value = 2.0;

        // Positive velocity (flow from left to right)
        let velocity = 1.0;
        let upwind_value = Equation::compute_upwind_flux(left_value, right_value, velocity);
        assert_eq!(upwind_value, left_value);

        // Negative velocity (flow from right to left)
        let velocity = -1.0;
        let upwind_value = Equation::compute_upwind_flux(left_value, right_value, velocity);
        assert_eq!(upwind_value, right_value);

        // Zero velocity (stationary flow)
        let velocity = 0.0;
        let upwind_value = Equation::compute_upwind_flux(left_value, right_value, velocity);
        assert_eq!(upwind_value, left_value);
    }
}
