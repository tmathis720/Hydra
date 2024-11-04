// src/equation/equation.rs

use crate::domain::{mesh::Mesh, Section};
use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
use crate::equation::reconstruction::reconstruct::reconstruct_face_value;
use crate::geometry::{FaceShape, Geometry};
use crate::domain::mesh_entity::MeshEntity;

use super::fields::{Fields, Fluxes};
use super::PhysicalEquation;

/// `Equation` is a struct representing the primary fluid flow equations (momentum and continuity)
/// for use in the finite volume method. It calculates fluxes at the faces of control volumes
/// in the domain mesh, using methods such as TVD (Total Variation Diminishing) upwinding,
/// to achieve stable and accurate flux approximations at cell interfaces.
pub struct Equation {
    // Define any necessary fields, such as parameters or constants.
    // Fields may include solver parameters, constants, or other data required by the equation.
}

// For Equation (momentum and continuity)
impl PhysicalEquation for Equation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        self.calculate_fluxes(
            domain,
            &fields.field,
            &fields.gradient,
            &fields.velocity_field,
            &mut fluxes.momentum_fluxes,
            boundary_handler,
        );
    }
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
        // Removed solver and preconditioner from parameters
    ) {
        let mut geometry = Geometry::new();
    
        // Iterate over face entities in the mesh
        for face in domain.entities.read().unwrap().iter().filter_map(|e| {
            if let MeshEntity::Face(_) = e {
                Some(e)
            } else {
                None
            }
        }) {
            // Identify cells sharing this face
            let neighbor_cells = domain.sieve.cone(face).unwrap_or_default();
            let cells: Vec<_> = neighbor_cells.iter().cloned().collect();
    
            // Retrieve the face vertices
            let face_vertices_entities = domain.get_vertices_of_face(face);
            let face_vertices: Vec<[f64; 3]> = face_vertices_entities.iter()
                .filter_map(|vertex_entity| {
                    if let MeshEntity::Vertex(vertex_id) = vertex_entity {
                        domain.get_vertex_coordinates(*vertex_id)
                    } else {
                        None
                    }
                }).collect();
    
            // Ensure face geometry is valid by checking vertex count
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue, // Unsupported face shape; skip processing
            };
    
            // Compute face normal and area for flux calculation
            let face_normal = geometry.compute_face_normal(domain, face, &cells[0]);
            if face_normal.is_none() {
                continue; // Skip if normal computation fails
            }
            let face_normal = face_normal.unwrap();
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
    
            // Normalize face normal vector for consistent computation
            let mut normal = face_normal;
            let normal_length = normal.iter().map(|&n| n * n).sum::<f64>().sqrt();
            if normal_length.abs() < 1e-12 {
                continue; // Skip if normal is zero for numerical stability
            }
            normal.iter_mut().for_each(|n| *n /= normal_length);

            // Initialize variables for flux computation
            let mut left_value = 0.0;
            let mut right_value = 0.0;
            let mut velocity = 0.0;

            if cells.len() == 2 {
                // Internal face (shared by two cells)
                let (cell_left, cell_right) = (&cells[0], &cells[1]);

                // Retrieve field and gradient data with validation
                let phi_left = field.restrict(cell_left).unwrap_or_default();
                let grad_left = gradient.restrict(cell_left).unwrap_or_default();
                let phi_right = field.restrict(cell_right).unwrap_or_default();
                let grad_right = gradient.restrict(cell_right).unwrap_or_default();

                // Compute cell and face centers
                let cell_left_center = geometry.compute_cell_centroid(domain, cell_left);
                let cell_right_center = geometry.compute_cell_centroid(domain, cell_right);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face values
                let _left_value = reconstruct_face_value(phi_left, grad_left, cell_left_center, face_center);
                let _right_value = reconstruct_face_value(phi_right, grad_right, cell_right_center, face_center);

                // Compute normal velocity components
                let vel_left = velocity_field.restrict(cell_left).unwrap_or([0.0, 0.0, 0.0]);
                let vel_right = velocity_field.restrict(cell_right).unwrap_or([0.0, 0.0, 0.0]);
                let vel_normal_left = vel_left.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();
                let vel_normal_right = vel_right.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                // Average normal velocity at the face
                let _velocity = 0.5 * (vel_normal_left + vel_normal_right);

            } else if cells.len() == 1 {
                // Boundary face (shared by a single cell)
                let cell_left = &cells[0];

                // Retrieve field and gradient data
                let phi_left = field.restrict(cell_left).unwrap_or_default();
                let grad_left = gradient.restrict(cell_left).unwrap_or_default();
                let cell_left_center = geometry.compute_cell_centroid(domain, cell_left);
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Reconstruct face value from internal cell data
                left_value = reconstruct_face_value(phi_left, grad_left, cell_left_center, face_center);

                // Apply boundary conditions
                right_value = match boundary_handler.get_bc(face) {
                    Some(BoundaryCondition::Dirichlet(value)) => value,
                    Some(BoundaryCondition::Neumann(_)) => left_value,
                    _ => left_value,
                };

                // Compute velocity component at the face
                let vel_left = velocity_field.restrict(cell_left).unwrap_or([0.0, 0.0, 0.0]);
                velocity = vel_left.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

            } else {
                // Skip if no associated cells
                continue;
            }

            // Compute the upwind flux value based on the velocity direction
            let upwind_value = Self::compute_upwind_flux(left_value, right_value, velocity);

            // Calculate the flux through the face
            let flux = upwind_value * velocity * face_area;

            // Store the computed flux in the output section
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
