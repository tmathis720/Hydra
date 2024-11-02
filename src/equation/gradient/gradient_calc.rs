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
            let phi_c = field.restrict(&cell).ok_or("Field value not found for cell")?;
            let mut grad_phi = [0.0; 3];
            let cell_vertices = self.mesh.get_cell_vertices(&cell);

            if cell_vertices.is_empty() {
                return Err(format!(
                    "Cell {:?} has 0 vertices; cannot compute volume or gradient.",
                    cell
                )
                .into());
            }

            let volume = self.geometry.compute_cell_volume(self.mesh, &cell);
            if volume == 0.0 {
                return Err("Cell volume is zero; cannot compute gradient.".into());
            }

            if let Some(faces) = self.mesh.get_faces_of_cell(&cell) {
                for face_entry in faces.iter() {
                    let face = face_entry.key();
                    let face_vertices = self.mesh.get_face_vertices(face);
                    let face_shape = self.determine_face_shape(face_vertices.len())?;
                    let area = self.geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
                    let normal = self.geometry.compute_face_normal(self.mesh, face, &cell)
                        .ok_or("Face normal not found")?;
                    let flux_vector = [normal[0] * area, normal[1] * area, normal[2] * area];
                    let neighbor_cells = self.mesh.get_cells_sharing_face(face);
                    
                    let nb_cell = neighbor_cells.iter()
                        .find(|neighbor| *neighbor.key() != cell)
                        .map(|entry| entry.key().clone());

                    if let Some(nb_cell) = nb_cell {
                        let phi_nb = field.restrict(&nb_cell).ok_or("Field value not found for neighbor cell")?;
                        let delta_phi = phi_nb - phi_c;
                        for i in 0..3 {
                            grad_phi[i] += delta_phi * flux_vector[i];
                        }
                    } else {
                        self.apply_boundary_condition(face, phi_c, flux_vector, time, &mut grad_phi)?;
                    }
                }

                for i in 0..3 {
                    grad_phi[i] /= volume;
                }

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
                    self.apply_dirichlet_boundary(value, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Neumann(flux) => {
                    self.apply_neumann_boundary(flux, flux_vector, grad_phi);
                }
                BoundaryCondition::Robin { alpha: _, beta: _ } => {
                    return Err("Robin boundary condition not implemented for gradient computation".into());
                }
                BoundaryCondition::DirichletFn(fn_bc) => {
                    let coords = self.geometry.compute_face_centroid(FaceShape::Triangle, &self.mesh.get_face_vertices(face));
                    let phi_nb = fn_bc(time, &coords);
                    self.apply_dirichlet_boundary(phi_nb, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::NeumannFn(fn_bc) => {
                    let coords = self.geometry.compute_face_centroid(FaceShape::Triangle, &self.mesh.get_face_vertices(face));
                    let flux = fn_bc(time, &coords);
                    self.apply_neumann_boundary(flux, flux_vector, grad_phi);
                }
                BoundaryCondition::Mixed { gamma, delta } => {
                    self.apply_mixed_boundary(gamma, delta, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Cauchy { lambda, mu } => {
                    self.apply_cauchy_boundary(lambda, mu, flux_vector, grad_phi);
                }
            }
        }
        Ok(())
    }

    /// Applies a Dirichlet boundary condition by adding flux contribution.
    fn apply_dirichlet_boundary(&self, value: f64, phi_c: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        let delta_phi = value - phi_c;
        for i in 0..3 {
            grad_phi[i] += delta_phi * flux_vector[i];
        }
    }

    /// Applies a Neumann boundary condition by adding constant flux.
    fn apply_neumann_boundary(&self, flux: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += flux * flux_vector[i];
        }
    }

    /// Applies a Mixed boundary condition by combining field value and flux.
    fn apply_mixed_boundary(&self, gamma: f64, delta: f64, phi_c: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        let mixed_contrib = gamma * phi_c + delta;
        for i in 0..3 {
            grad_phi[i] += mixed_contrib * flux_vector[i];
        }
    }

    /// Applies a Cauchy boundary condition by adding lambda to flux and mu to field.
    fn apply_cauchy_boundary(&self, lambda: f64, mu: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += lambda * flux_vector[i] + mu;
        }
    }

    /// Determines face shape based on vertex count.
    fn determine_face_shape(&self, vertex_count: usize) -> Result<FaceShape, Box<dyn Error>> {
        match vertex_count {
            3 => Ok(FaceShape::Triangle),
            4 => Ok(FaceShape::Quadrilateral),
            _ => Err(format!(
                "Unsupported face shape with {} vertices for gradient computation",
                vertex_count
            )
            .into()),
        }
    }
}
