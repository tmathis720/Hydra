use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition};
use crate::geometry::{Geometry, FaceShape};
use std::error::Error;

pub struct Gradient<'a> {
    mesh: &'a Mesh,
    boundary_handler: &'a BoundaryConditionHandler,
    geometry: Geometry,
}

impl<'a> Gradient<'a> {
    pub fn new(mesh: &'a Mesh, boundary_handler: &'a BoundaryConditionHandler) -> Self {
        Self {
            mesh,
            boundary_handler,
            geometry: Geometry::new(),
        }
    }

    pub fn compute_gradient(
        &mut self,
        field: &Section<f64>,
        gradient: &mut Section<[f64; 3]>,
        time: f64,
    ) -> Result<(), Box<dyn Error>> {
        for cell in self.mesh.get_cells() {
            let phi_c = field.restrict(&cell).ok_or("Field value not found for cell")?;
            let mut grad_phi = [0.0; 3];

            // Ensure valid cell vertices for computing cell volume
            let cell_vertices = self.mesh.get_cell_vertices(&cell);
            if cell_vertices.is_empty() {
                return Err(format!(
                    "Cell {:?} has 0 vertices; cannot compute volume or gradient.",
                    cell
                )
                .into());
            }

            // Obtain cell volume with error handling
            let volume = self.geometry.compute_cell_volume(self.mesh, &cell);
            if volume == 0.0 {
                return Err("Cell volume is zero; cannot compute gradient.".into());
            }

            if let Some(faces) = self.mesh.get_faces_of_cell(&cell) {
                for face_entry in faces.iter() {
                    let face = face_entry.key();

                    // Retrieve face vertices and determine shape
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

                    // Find neighboring cell or apply boundary conditions
                    let neighbor_cells = self.mesh.get_cells_sharing_face(face);
                    let nb_cell = neighbor_cells
                        .iter()
                        .find(|neighbor| *neighbor.key() != cell)
                        .map(|entry| entry.key().clone());

                    if let Some(nb_cell) = nb_cell {
                        // Neighbor cell found, compute flux contribution
                        let phi_nb = field.restrict(&nb_cell).ok_or("Field value not found for neighbor cell")?;
                        let delta_phi = phi_nb - phi_c;
                        for i in 0..3 {
                            grad_phi[i] += delta_phi * flux_vector[i];
                        }
                    } else {
                        // Apply boundary conditions if no neighbor cell
                        self.apply_boundary_condition(
                            face,
                            phi_c,
                            flux_vector,
                            time,
                            &mut grad_phi,
                        )?;
                    }
                }

                // Apply cell volume to finalize the gradient
                for i in 0..3 {
                    grad_phi[i] /= volume;
                }

                // Store the computed gradient
                gradient.set_data(cell, grad_phi);
            }
        }

        Ok(())
    }

    /// Applies boundary conditions for faces without a neighboring cell.
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
                    let delta_phi = value - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                }
                BoundaryCondition::Neumann(flux) => {
                    for i in 0..3 {
                        grad_phi[i] += flux * flux_vector[i];
                    }
                }
                BoundaryCondition::Robin { alpha: _, beta: _ } => {
                    return Err("Robin boundary condition not implemented for gradient computation".into());
                }
                BoundaryCondition::DirichletFn(fn_bc) => {
                    let coords = self.geometry.compute_face_centroid(FaceShape::Triangle, &self.mesh.get_face_vertices(face));
                    let phi_nb = fn_bc(time, &coords);
                    let delta_phi = phi_nb - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                }
                BoundaryCondition::NeumannFn(fn_bc) => {
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
