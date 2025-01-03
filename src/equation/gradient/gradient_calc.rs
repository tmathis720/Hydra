use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::geometry::{FaceShape, Geometry};
use crate::equation::gradient::GradientMethod;
use crate::domain::section::{scalar::Scalar, vector::Vector3};
use std::error::Error;

/// Struct for the finite volume gradient calculation method.
///
/// This struct implements the `GradientMethod` trait for finite volume
/// computations of gradient.
pub struct FiniteVolumeGradient;

impl GradientMethod for FiniteVolumeGradient {
    /// Computes the gradient of a scalar field across each cell in the mesh.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure containing cell and face connectivity.
    /// - `boundary_handler`: Reference to a handler that manages boundary conditions.
    /// - `geometry`: Geometry utilities for computing areas, volumes, etc.
    /// - `field`: Scalar field values for each cell.
    /// - `cell`: The current cell for which the gradient is computed.
    /// - `time`: Current simulation time.
    ///
    /// # Returns
    /// - `Ok([f64; 3])`: Computed gradient vector.
    /// - `Err(Box<dyn Error>)`: If any error occurs during computation.
    fn calculate_gradient(
        &self,
        mesh: &Mesh,
        boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        field: &Section<Scalar>,
        cell: &MeshEntity,
        time: f64,
    ) -> Result<[f64; 3], Box<dyn Error>> {
        let phi_c = field.restrict(cell).ok_or("Field value not found for cell")?.0;
        let mut grad_phi = [0.0; 3];
        let cell_vertices = mesh.get_cell_vertices(cell);

        if cell_vertices.is_empty() {
            return Err(format!("Cell {:?} has 0 vertices; cannot compute volume or gradient.", cell).into());
        }

        let volume = geometry.compute_cell_volume(mesh, cell);
        if volume == 0.0 {
            return Err("Cell volume is zero; cannot compute gradient.".into());
        }

        if let Some(faces) = mesh.get_faces_of_cell(cell) {
            for face_entry in faces.iter() {
                let face = face_entry.key();
                let face_vertices = mesh.get_face_vertices(face);
                let face_shape = self.determine_face_shape(face_vertices.len())?;
                let area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
                let normal = geometry.compute_face_normal(mesh, face, cell)
                    .ok_or("Face normal not found")?;
                let flux_vector = Vector3([normal[0] * area, normal[1] * area, normal[2] * area]);
                let neighbor_cells = mesh.get_cells_sharing_face(face);

                let nb_cell = neighbor_cells.iter()
                    .find(|neighbor| *neighbor.key() != *cell)
                    .map(|entry| entry.key().clone());

                if let Some(nb_cell) = nb_cell {
                    let phi_nb = field.restrict(&nb_cell).ok_or("Field value not found for neighbor cell")?.0;
                    let delta_phi = phi_nb - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                } else {
                    // Pass boundary_handler directly to the function
                    self.apply_boundary_condition(face, phi_c, flux_vector, time, &mut grad_phi, boundary_handler, geometry, mesh)?;
                }
            }

            for i in 0..3 {
                grad_phi[i] /= volume;
            }
        }

        Ok(grad_phi)
    }
}

impl FiniteVolumeGradient {
    /// Applies boundary conditions for a face without a neighboring cell.
    ///
    /// # Parameters
    /// - `face`: The face entity for which boundary conditions are applied.
    /// - `phi_c`: Scalar field value at the current cell.
    /// - `flux_vector`: Scaled normal vector representing face flux direction.
    /// - `time`: Simulation time, required for time-dependent boundary functions.
    /// - `grad_phi`: Accumulator array to which boundary contributions will be added.
    /// - `boundary_handler`: Boundary condition handler.
    /// - `geometry`: Geometry utility for calculations.
    /// - `mesh`: Mesh structure to access cell and face data.
    ///
    /// # Returns
    /// - `Ok(())`: Boundary condition successfully applied.
    /// - `Err(Box<dyn Error>)`: If the boundary condition type is unsupported.
    fn apply_boundary_condition(
        &self,
        face: &MeshEntity,
        phi_c: f64,
        flux_vector: Vector3,
        time: f64,
        grad_phi: &mut [f64; 3],
        boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        mesh: &Mesh,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(bc) = boundary_handler.get_bc(face) {
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
                BoundaryCondition::DirichletFn(wrapper) => {
                    let coords = geometry.compute_face_centroid(FaceShape::Triangle, &mesh.get_face_vertices(face));
                    let phi_nb = (wrapper.function)(time, &coords);
                    self.apply_dirichlet_boundary(phi_nb, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::NeumannFn(wrapper) => {
                    let coords = geometry.compute_face_centroid(FaceShape::Triangle, &mesh.get_face_vertices(face));
                    let flux = (wrapper.function)(time, &coords);
                    self.apply_neumann_boundary(flux, flux_vector, grad_phi);
                }
                BoundaryCondition::Mixed { gamma, delta } => {
                    self.apply_mixed_boundary(gamma, delta, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Cauchy { lambda, mu } => {
                    self.apply_cauchy_boundary(lambda, mu, flux_vector, grad_phi);
                }
                BoundaryCondition::SolidWallInviscid => {
                    self.apply_solid_wall_inviscid_boundary(phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::SolidWallViscous { normal_velocity } => {
                    self.apply_solid_wall_viscous_boundary(normal_velocity, flux_vector, grad_phi);
                }
                BoundaryCondition::FarField(value) => {
                    self.apply_dirichlet_boundary(value, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Injection(value) => {
                    self.apply_injection_boundary(value, flux_vector, grad_phi);
                }
                BoundaryCondition::InletOutlet => {
                    self.apply_inlet_outlet_boundary(phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Symmetry => {
                    self.apply_symmetry_boundary(phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Periodic { .. } => {
                    return Err("Periodic boundary condition not applicable for gradient computation".into());
                }
            }
        }
        Ok(())
    }
    
    /// Applies a Dirichlet boundary condition by adding flux contribution.
    fn apply_dirichlet_boundary(&self, value: f64, phi_c: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        let delta_phi = value - phi_c;
        for i in 0..3 {
            grad_phi[i] += delta_phi * flux_vector[i];
        }
    }
    
    /// Applies a Neumann boundary condition by adding constant flux.
    fn apply_neumann_boundary(&self, flux: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += flux * flux_vector[i];
        }
    }
    
    /// Applies a Mixed boundary condition by combining field value and flux.
    fn apply_mixed_boundary(&self, gamma: f64, delta: f64, phi_c: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        let mixed_contrib = gamma * phi_c + delta;
        for i in 0..3 {
            grad_phi[i] += mixed_contrib * flux_vector[i];
        }
    }
    
    /// Applies a Cauchy boundary condition by adding lambda to flux and mu to field.
    fn apply_cauchy_boundary(&self, lambda: f64, mu: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += lambda * flux_vector[i] + mu;
        }
    }

    /// Applies an inviscid solid wall boundary condition by ensuring no normal flux.
    fn apply_solid_wall_inviscid_boundary(&self, _phi_c: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        // For inviscid walls, the normal flux is zero; no contribution to grad_phi.
        let no_flux = 0.0;
        for i in 0..3 {
            grad_phi[i] += no_flux * flux_vector[i];
        }
    }

    /// Applies a viscous solid wall boundary condition by enforcing no-slip velocity.
    fn apply_solid_wall_viscous_boundary(&self, normal_velocity: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        // For viscous walls, the velocity at the wall is equal to the normal velocity.
        for i in 0..3 {
            grad_phi[i] += normal_velocity * flux_vector[i];
        }
    }

    /// Applies an injection boundary condition by adding injected flux.
    fn apply_injection_boundary(&self, value: f64, flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += value * flux_vector[i];
        }
    }

    /// Applies an inlet-outlet boundary condition by assuming zero gradient.
    fn apply_inlet_outlet_boundary(&self, _phi_c: f64, _flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        // Assume no contribution to the gradient from inlet/outlet
        for i in 0..3 {
            grad_phi[i] += 0.0;
        }
    }

    /// Applies a symmetry boundary condition by ensuring no flux contribution.
    fn apply_symmetry_boundary(&self, _phi_c: f64, _flux_vector: Vector3, grad_phi: &mut [f64; 3]) {
        // For symmetry, the flux vector normal to the boundary does not contribute.
        for i in 0..3 {
            grad_phi[i] += 0.0;
        }
    }
    
    /// Determines face shape based on vertex count.
    fn determine_face_shape(&self, vertex_count: usize) -> Result<FaceShape, Box<dyn Error>> {
        match vertex_count {
            2 => Ok(FaceShape::Edge),
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
