use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition};
use crate::geometry::{Geometry, FaceShape};
use std::error::Error;

/// Struct for calculating gradients of a scalar field across a mesh.
/// 
/// # Purpose
/// The `Gradient` struct computes the spatial gradient of a scalar field,
/// often needed in fluid dynamics simulations to evaluate fluxes and
/// advective transport terms in finite volume methods. It interfaces with
/// mesh and geometry structures to handle cell and face details, while
/// incorporating boundary conditions where applicable.
///
/// # Fields
/// - `mesh`: Reference to the mesh data structure.
/// - `boundary_handler`: Handler for managing boundary conditions.
/// - `geometry`: Used for geometrical calculations like volume and face area.
pub struct Gradient<'a> {
    mesh: &'a Mesh,
    boundary_handler: &'a BoundaryConditionHandler,
    geometry: Geometry,
}

impl<'a> Gradient<'a> {
    /// Constructs a new `Gradient` calculator with the given mesh and boundary handler.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure containing cell and face connectivity.
    /// - `boundary_handler`: Reference to a handler that manages boundary conditions.
    ///
    /// # Returns
    /// Returns an initialized `Gradient` struct ready to compute gradients.
    pub fn new(mesh: &'a Mesh, boundary_handler: &'a BoundaryConditionHandler) -> Self {
        Self {
            mesh,
            boundary_handler,
            geometry: Geometry::new(),
        }
    }

    /// Computes the gradient of a scalar field across each cell in the mesh.
    ///
    /// # Parameters
    /// - `field`: A section containing scalar field values for each cell in the mesh.
    /// - `gradient`: A mutable section where the computed gradient vectors `[f64; 3]` will be stored.
    /// - `time`: Current simulation time, passed to boundary condition functions as required.
    ///
    /// # Returns
    /// - `Ok(())`: If gradients are successfully computed for all cells.
    /// - `Err(Box<dyn Error>)`: If any issue arises, such as missing values or zero cell volume.
    ///
    /// # Description
    /// This function iterates through each cell, computes the gradient by
    /// summing flux contributions from each face, and applies the volume to
    /// finalize the result. If a face lacks a neighboring cell, boundary conditions
    /// are applied as needed.
    pub fn compute_gradient(
        &mut self,
        field: &Section<f64>,
        gradient: &mut Section<[f64; 3]>,
        time: f64,
    ) -> Result<(), Box<dyn Error>> {
        for cell in self.mesh.get_cells() {
            // Retrieve the field value for the current cell
            let phi_c = field.restrict(&cell).ok_or("Field value not found for cell")?;
            let mut grad_phi = [0.0; 3];

            // Retrieve vertices for cell volume computation, ensure it is non-empty
            let cell_vertices = self.mesh.get_cell_vertices(&cell);
            if cell_vertices.is_empty() {
                return Err(format!(
                    "Cell {:?} has 0 vertices; cannot compute volume or gradient.",
                    cell
                )
                .into());
            }

            // Calculate cell volume to scale gradient contributions
            let volume = self.geometry.compute_cell_volume(self.mesh, &cell);
            if volume == 0.0 {
                return Err("Cell volume is zero; cannot compute gradient.".into());
            }

            // Sum face flux contributions for gradient
            if let Some(faces) = self.mesh.get_faces_of_cell(&cell) {
                for face_entry in faces.iter() {
                    let face = face_entry.key();

                    // Determine face shape and retrieve vertices
                    let face_vertices = self.mesh.get_face_vertices(face);
                    let face_shape = match face_vertices.len() {
                        3 => FaceShape::Triangle,
                        4 => FaceShape::Quadrilateral,
                        _ => {
                            return Err(format!(
                                "Unsupported face shape with {} vertices for gradient computation",
                                face_vertices.len()
                            )
                            .into());
                        }
                    };

                    // Compute face area and normal
                    let area = self.geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
                    let normal = self
                        .geometry
                        .compute_face_normal(self.mesh, face, &cell)
                        .ok_or("Face normal not found")?;

                    let flux_vector = [
                        normal[0] * area,
                        normal[1] * area,
                        normal[2] * area,
                    ];

                    // Find neighboring cell or apply boundary condition
                    let neighbor_cells = self.mesh.get_cells_sharing_face(face);
                    let nb_cell = neighbor_cells
                        .iter()
                        .find(|neighbor| *neighbor.key() != cell)
                        .map(|entry| entry.key().clone());

                    if let Some(nb_cell) = nb_cell {
                        // Neighboring cell found, add flux contribution
                        let phi_nb = field.restrict(&nb_cell).ok_or("Field value not found for neighbor cell")?;
                        let delta_phi = phi_nb - phi_c;
                        for i in 0..3 {
                            grad_phi[i] += delta_phi * flux_vector[i];
                        }
                    } else {
                        // No neighboring cell: apply boundary condition
                        self.apply_boundary_condition(
                            face,
                            phi_c,
                            flux_vector,
                            time,
                            &mut grad_phi,
                        )?;
                    }
                }

                // Finalize gradient by dividing by cell volume
                for i in 0..3 {
                    grad_phi[i] /= volume;
                }

                // Store computed gradient
                gradient.set_data(cell, grad_phi);
            }
        }

        Ok(())
    }

    /// Applies boundary conditions for a face without a neighboring cell.
    ///
    /// # Parameters
    /// - `face`: The face entity for which boundary conditions are applied.
    /// - `phi_c`: Scalar field value at the current cell.
    /// - `flux_vector`: Scaled normal vector representing face flux direction.
    /// - `time`: Simulation time, required for time-dependent boundary functions.
    /// - `grad_phi`: Accumulator array to which boundary contributions will be added.
    ///
    /// # Returns
    /// - `Ok(())`: Boundary condition successfully applied.
    /// - `Err(Box<dyn Error>)`: If the boundary condition type is unsupported.
    ///
    /// # Supported Boundary Conditions
    /// - `Dirichlet`: Sets a fixed value on the face.
    /// - `Neumann`: Applies a constant flux across the face.
    /// - `DirichletFn`: Dirichlet with a time-dependent function.
    /// - `NeumannFn`: Neumann with a time-dependent function.
    fn apply_boundary_condition(
        &self,
        face: &MeshEntity,
        phi_c: f64,
        flux_vector: [f64; 3],
        time: f64,
        grad_phi: &mut [f64; 3],
    ) -> Result<(), Box<dyn Error>> {
        if let Some(bc) = self.boundary_handler.get_bc(face) {
            match bc {
                BoundaryCondition::Dirichlet(value) => {
                    // Fixed value at boundary face; add flux contribution
                    let delta_phi = value - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                }
                BoundaryCondition::Neumann(flux) => {
                    // Constant flux boundary condition
                    for i in 0..3 {
                        grad_phi[i] += flux * flux_vector[i];
                    }
                }
                BoundaryCondition::Robin { alpha: _, beta: _ } => {
                    return Err("Robin boundary condition not implemented for gradient computation".into());
                }
                BoundaryCondition::DirichletFn(fn_bc) => {
                    // Time-dependent Dirichlet condition
                    let coords = self.geometry.compute_face_centroid(FaceShape::Triangle, &self.mesh.get_face_vertices(face));
                    let phi_nb = fn_bc(time, &coords);
                    let delta_phi = phi_nb - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                }
                BoundaryCondition::NeumannFn(fn_bc) => {
                    // Time-dependent Neumann condition
                    let coords = self.geometry.compute_face_centroid(FaceShape::Triangle, &self.mesh.get_face_vertices(face));
                    let flux = fn_bc(time, &coords);
                    for i in 0..3 {
                        grad_phi[i] += flux * flux_vector[i];
                    }
                }
            }
        }
        Ok(())
    }
}
