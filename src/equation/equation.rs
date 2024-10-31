// src/equation/equation.rs

use crate::domain::{mesh::Mesh, Section};
use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
use crate::equation::reconstruction::reconstruct::reconstruct_face_value;
use crate::geometry::{FaceShape, Geometry};
use crate::domain::mesh_entity::MeshEntity;

/// `Equation` is a struct representing the primary fluid flow equations (momentum and continuity)
/// for use in the finite volume method. It calculates fluxes at the faces of control volumes
/// in the domain mesh, using methods such as TVD (Total Variation Diminishing) upwinding,
/// to achieve stable and accurate flux approximations at cell interfaces.
pub struct Equation {
    // Define any necessary fields, such as parameters or constants.
    // Fields may include solver parameters, constants, or other data required by the equation.
}

impl Equation {
    /// Calculates fluxes at each face of the cells in a mesh using TVD upwinding.
    /// This method iterates over each face in the mesh and applies the upwinding scheme to compute
    /// fluxes, which are stored in the `fluxes` section. For boundary faces, it applies the 
    /// boundary conditions accordingly.
    ///
    /// # Parameters
    /// - `domain`: Reference to the domain mesh, containing mesh geometry and topology.
    /// - `field`: Section with scalar field values (e.g., pressure) for each cell in the mesh.
    /// - `gradient`: Section with gradient vectors for each cell, aiding flux reconstruction.
    /// - `velocity_field`: Section containing velocity vectors for each cell.
    /// - `fluxes`: Mutable section to store computed flux values at each face.
    /// - `boundary_handler`: Reference to boundary condition handler, managing BCs for faces.
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        field: &Section<f64>,
        gradient: &Section<[f64; 3]>,
        velocity_field: &Section<[f64; 3]>,
        fluxes: &mut Section<f64>,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        let mut geometry = Geometry::new();

        // Iterate only over face entities in the mesh
        for face in domain.entities.read().unwrap().iter().filter_map(|e| {
            if let MeshEntity::Face(_) = e {
                Some(e)
            } else {
                None
            }
        }) {
            // Identify cells sharing this face
            let neighbor_cells = domain.get_cells_sharing_face(face);
            let cells: Vec<_> = neighbor_cells.iter().map(|entry| entry.key().clone()).collect();

            // Ensure face geometry is valid by checking vertex count
            let face_vertices = domain.get_face_vertices(face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue, // Unsupported face shape; skip processing
            };

            // Compute face normal and area for flux calculation
            let face_normal = geometry.compute_face_normal(domain, face, &cells[0]).unwrap();
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);

            // Normalize face normal vector for consistent flux computation
            let mut normal = face_normal;
            let normal_length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            for i in 0..3 {
                normal[i] /= normal_length;
            }

            // Initialize variables for flux computation at this face
            let mut left_value = 0.0;
            let mut right_value = 0.0;
            let mut velocity = 0.0;

            if cells.len() == 2 {
                // Internal face (shared by two cells)
                let cell_left = &cells[0];
                let cell_right = &cells[1];

                // Retrieve field values and gradients for left and right cells
                let phi_left = field.restrict(cell_left).unwrap();
                let grad_left = gradient.restrict(cell_left).unwrap();
                let phi_right = field.restrict(cell_right).unwrap();
                let grad_right = gradient.restrict(cell_right).unwrap();

                // Retrieve centers for left and right cells, and face center
                let cell_left_center = geometry.compute_cell_centroid(domain, cell_left);
                let cell_right_center = geometry.compute_cell_centroid(domain, cell_right);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face values from left and right cell data
                let mut _left_value = reconstruct_face_value(phi_left, grad_left, cell_left_center, face_center);
                let mut _right_value = reconstruct_face_value(phi_right, grad_right, cell_right_center, face_center);

                // Compute normal component of velocity at the face by averaging
                let vel_left = velocity_field.restrict(cell_left).unwrap();
                let vel_right = velocity_field.restrict(cell_right).unwrap();
                let vel_normal_left = vel_left[0] * normal[0] + vel_left[1] * normal[1] + vel_left[2] * normal[2];
                let vel_normal_right = vel_right[0] * normal[0] + vel_right[1] * normal[1] + vel_right[2] * normal[2];

                // Average the normal component of velocity across the face
                let mut _velocity = 0.5 * (vel_normal_left + vel_normal_right);

            } else if cells.len() == 1 {
                // Boundary face (shared by a single cell)
                let cell_left = &cells[0];
                let phi_left = field.restrict(cell_left).unwrap();
                let grad_left = gradient.restrict(cell_left).unwrap();
                let cell_left_center = geometry.compute_cell_centroid(domain, cell_left);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face value from internal cell data
                left_value = reconstruct_face_value(phi_left, grad_left, cell_left_center, face_center);

                // Apply boundary condition on face to determine right value
                if let Some(bc) = boundary_handler.get_bc(face) {
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            // Dirichlet BC: set the right value as the specified boundary value
                            right_value = value;
                        }
                        BoundaryCondition::Neumann(_) => {
                            // Neumann BC: no flux adjustment here; assign left value as default
                            right_value = left_value;
                        }
                        _ => {
                            // Other BCs not specifically handled default to left value
                            right_value = left_value;
                        }
                    }
                } else {
                    // Default if no boundary condition is specified
                    right_value = left_value;
                }

                // Calculate velocity normal component at face from left cell data
                let vel_left = velocity_field.restrict(cell_left).unwrap();
                velocity = vel_left[0] * normal[0] + vel_left[1] * normal[1] + vel_left[2] * normal[2];
            } else {
                // Skip processing if no associated cells for the face
                continue;
            }

            // Compute the upwind flux value based on the velocity direction
            let upwind_value = Self::compute_upwind_flux(left_value, right_value, velocity);

            // Calculate the flux across the face as flux = upwind_value * velocity * area
            let flux = upwind_value * velocity * face_area;

            // Store the computed flux in the `fluxes` section for this face
            fluxes.set_data(face.clone(), flux);
        }
    }

    /// Determines the upwind flux based on the flow direction (sign of velocity).
    /// If the velocity is positive, the upwind value is the `left_value` (upwind cell);
    /// otherwise, it is the `right_value` (downwind cell). This method is central to
    /// implementing the upwinding scheme, ensuring numerical stability.
    ///
    /// # Parameters
    /// - `left_value`: The scalar field value from the upwind (left) cell.
    /// - `right_value`: The scalar field value from the downwind (right) cell.
    /// - `velocity`: The normal velocity component at the face.
    ///
    /// # Returns
    /// The selected upwind flux value based on the flow direction.
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
