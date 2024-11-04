### Detailed Approach for Integrating Krylov Subspace Methods in the `Equation` Module

**Objective**: Integrate efficient solvers for large-scale sparse linear systems within the `Equation` module using Krylov methods, including Generalized Minimal Residual (GMRES) for non-symmetric matrices and Conjugate Gradient (CG) for symmetric positive-definite (SPD) matrices.

### Step 1: Update `Equation::calculate_fluxes` to Use Solvers
   - **Objective**: Modify `calculate_fluxes` in `src/equation/equation.rs` to interface with the `Solver` trait for solving linear systems during flux calculation.
   
   - **Integration**:
     - Replace direct linear solver calls with a call to `self.solver.solve(...)`, allowing the solver configuration (GMRES or CG) to be dynamically selected based on the matrix properties.
     - Integrate preconditioning by passing the configured preconditioner to the solver if applicable.

   - **Example Code**:
     ```rust
     pub fn calculate_fluxes(
         &self,
         domain: &Mesh,
         field: &Section<f64>,
         gradient: &Section<[f64; 3]>,
         velocity_field: &Section<[f64; 3]>,
         fluxes: &mut Section<f64>,
         boundary_handler: &BoundaryConditionHandler,
     ) {
         let result = self.solver.solve(&matrix, &rhs, &mut solution);
         assert!(result.converged, "Solver did not converge within tolerance");
         // Use solution to update flux values
     }
     ```

---

### Step 2: Testing and Validation
   
   - **Integration Tests**:
     - Develop end-to-end tests to verify the correct operation of `calculate_fluxes` in conjunction with the solver and preconditioners.
     - Compare computed results against known solutions or analytical benchmarks where possible.

   - **Performance Testing**:
     - Benchmark solver performance with and without preconditioning to measure convergence rate improvements and solver efficiency, referencing guidelines in *Iterative Methods for Sparse Linear Systems*【21†source】.

---

### Additional Enhancements
   - **Generic Scalar Types**: Consider extending the `KSP` traits to support scalar types beyond `f64` in the future for greater flexibility.
   - **Solver Configuration**: Develop a high-level configuration interface within the `KSP` module to facilitate dynamic selection and configuration of solvers based on problem requirements.
   - **Documentation**: Include detailed inline comments and module documentation explaining the design choices, configurations, and example usage for both solvers and preconditioners.

---

This approach systematically incorporates Krylov subspace methods into the `Equation` module, aligning with HYDRA’s modular architecture and ensuring flexibility for solving a wide range of sparse linear systems in geophysical simulations.

Below you will find the source code that is essential to include in your memory as we complete the steps above. Please provide complete revised/upgraded code with unit tests wherever possible. You can analyze the problem in whole, but for now, only output the required code to complete Step 1 above.

---

`src/equation/mod.rs`

```rust
pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;
```

---

`src/equation/equation.rs`

```rust
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
```

---

`src/equation/flux_limiter/mod.rs`

```rust
pub mod flux_limiters;

#[cfg(test)]
pub mod tests;
```

---

`src/equation/flux_limiter/flux_limiters.rs`

```rust
/// Trait defining a generic Flux Limiter, which adjusts flux values
/// to prevent numerical oscillations, crucial for Total Variation Diminishing (TVD) schemes.
/// 
/// # Purpose
/// This trait provides a method `limit` to calculate a modified value
/// based on neighboring values, which helps in maintaining the stability
/// and accuracy of the finite volume method by applying flux limiters.
/// 
/// # Method
/// - `limit`: Takes left and right flux values and returns a constrained value
/// to mitigate oscillations at cell interfaces.
pub trait FluxLimiter {
    /// Applies the limiter to two neighboring values to prevent oscillations.
    ///
    /// # Parameters
    /// - `left_value`: The flux value on the left side of the interface.
    /// - `right_value`: The flux value on the right side of the interface.
    ///
    /// # Returns
    /// A modified value that limits oscillations, ensuring TVD compliance.
    fn limit(&self, left_value: f64, right_value: f64) -> f64;
}

/// Implementation of the Minmod flux limiter.
///
/// # Characteristics
/// The Minmod limiter is a simple, commonly used limiter that chooses the minimum
/// absolute value of the left and right values while preserving the sign. It is effective
/// for handling sharp gradients without introducing non-physical oscillations.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0
///   to avoid oscillations.
/// - Otherwise, it selects the smaller absolute value, retaining the original sign.
pub struct Minmod;

/// Implementation of the Superbee flux limiter.
///
/// # Characteristics
/// The Superbee limiter provides higher resolution compared to Minmod and is more aggressive,
/// capturing sharp gradients while preserving stability. This limiter is suitable
/// for problems where capturing steep gradients is essential.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0,
///   preventing oscillations.
/// - Otherwise, it calculates two options based on twice the left and right values,
///   clamping them within the original range, and selects the larger of the two.
pub struct Superbee;

impl FluxLimiter for Minmod {
    /// Applies the Minmod flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs (indicating an oscillation).
    /// - Otherwise, returns the value with the smaller magnitude, preserving the sign.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Minmod: Different signs or zero - returning 0.0");
            0.0 // Different signs or zero: prevent oscillations by returning zero
        } else {
            // Take the minimum magnitude value, maintaining its original sign
            let result = if left_value.abs() < right_value.abs() {
                left_value
            } else {
                right_value
            };
            println!("Minmod: left_value = {}, right_value = {}, result = {}", left_value, right_value, result);
            result
        }
    }
}

impl FluxLimiter for Superbee {
    /// Applies the Superbee flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs, to prevent oscillations.
    /// - Otherwise, calculates two possible limited values and returns the maximum
    ///   to ensure higher resolution while maintaining stability.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Superbee: Different signs or zero - returning 0.0");
            0.0 // Different signs: prevent oscillations by returning zero
        } else {
            // Calculate two limited values and return the maximum to capture sharp gradients
            let option1 = (2.0 * left_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let option2 = (2.0 * right_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let result = option1.max(option2);

            println!(
                "Superbee: left_value = {}, right_value = {}, option1 = {}, option2 = {}, result = {}",
                left_value, right_value, option1, option2, result
            );

            result
        }
    }
}
```

---

`src/equation/flux_limiter/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use crate::equation::flux_limiter::flux_limiters::{FluxLimiter, Minmod, Superbee};

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_minmod_limiter() {
        let minmod = Minmod;

        // Test with same signs
        assert!(approx_eq(minmod.limit(1.0, 0.5), 0.5, 1e-6));
        assert!(approx_eq(minmod.limit(0.5, 1.0), 0.5, 1e-6));
        
        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(minmod.limit(1.0, -0.5), 0.0, 1e-6));
        assert!(approx_eq(minmod.limit(-1.0, 0.5), 0.0, 1e-6));

        // Test with zero values
        assert!(approx_eq(minmod.limit(0.0, 1.0), 0.0, 1e-6));
        assert!(approx_eq(minmod.limit(1.0, 0.0), 0.0, 1e-6));
        
        // Test edge cases with very high and low values
        assert!(approx_eq(minmod.limit(1e6, 1e6), 1e6, 1e-6));
        assert!(approx_eq(minmod.limit(-1e6, -1e6), -1e6, 1e-6));
    }

    #[test]
    fn test_superbee_limiter() {
        let superbee = Superbee;

        // Test with same signs
        assert!(approx_eq(superbee.limit(1.0, 0.5), 1.0, 1e-6));
        assert!(approx_eq(superbee.limit(0.5, 1.0), 1.0, 1e-6));

        // Test with opposite signs (expect zero to prevent oscillations)
        assert!(approx_eq(superbee.limit(1.0, -0.5), 0.0, 1e-6));
        assert!(approx_eq(superbee.limit(-1.0, 0.5), 0.0, 1e-6));

        // Test with zero values
        assert!(approx_eq(superbee.limit(0.0, 1.0), 0.0, 1e-6));
        assert!(approx_eq(superbee.limit(1.0, 0.0), 0.0, 1e-6));

        // Test edge cases with very high and low values
        assert!(approx_eq(superbee.limit(1e6, 1e6), 1e6, 1e-6));
        assert!(approx_eq(superbee.limit(-1e6, -1e6), -1e6, 1e-6));
    }
}
```

---

`src/equation/gradient/mod.rs`

```rust
pub mod gradient_calc;

#[cfg(test)]
pub mod tests;
```

---

`src/equation/gradient/gradient_calc.rs`

```rust
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
```

---

`src/equation/gradient/tests.rs`

```rust
// src/equation/gradient/tests.rs

use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::boundary::{bc_handler::BoundaryConditionHandler, bc_handler::BoundaryCondition};
use crate::equation::gradient::gradient_calc::Gradient;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a simple mesh used by all tests to ensure consistency.
    fn create_simple_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Create vertices
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);

        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);

        // Set vertex coordinates
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);

        // Create face
        let face = MeshEntity::Face(1);
        mesh.add_entity(face);

        // Create cells
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        mesh.add_entity(cell1);
        mesh.add_entity(cell2);

        // Build relationships
        // Cells to face
        mesh.add_relationship(cell1, face.clone());
        mesh.add_relationship(cell2, face.clone());
        // Cells to vertices
        for &cell in &[cell1, cell2] {
            mesh.add_relationship(cell, vertex1);
            mesh.add_relationship(cell, vertex2);
            mesh.add_relationship(cell, vertex3);
            mesh.add_relationship(cell, vertex4);
        }
        // Face to vertices
        mesh.add_relationship(face.clone(), vertex1);
        mesh.add_relationship(face.clone(), vertex2);
        mesh.add_relationship(face.clone(), vertex3);

        mesh
    }

    #[test]
    fn test_gradient_simple_mesh() {
        let mesh = create_simple_mesh();

        // Initialize field values
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);
        field.set_data(MeshEntity::Cell(2), 2.0);

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(
            result.is_ok(),
            "Gradient calculation failed with result: {:?}",
            result
        );

        let grad_cell1 = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed for cell1");
        println!("Computed gradient for cell1: {:?}", grad_cell1);

        // Expected gradient based on computations
        let expected_grad = [0.0, 0.0, 3.0];
        for i in 0..3 {
            assert!(
                (grad_cell1[i] - expected_grad[i]).abs() < 1e-6,
                "Gradient component {} does not match expected value",
                i
            );
        }
    }

    #[test]
    fn test_gradient_dirichlet_boundary() {
        let mesh = create_simple_mesh();

        // Remove cell2 and its relationships to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        // Initialize field
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Dirichlet(2.0));

        // Initialize gradient calculator
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(
            result.is_ok(),
            "Gradient calculation failed with result: {:?}",
            result
        );

        let grad = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient: {:?}", grad);

        // Expected gradient based on computations
        let expected_grad = [0.0, 0.0, 3.0];
        for i in 0..3 {
            assert!(
                (grad[i] - expected_grad[i]).abs() < 1e-6,
                "Gradient component {} does not match expected value",
                i
            );
        }
    }

    #[test]
    fn test_gradient_neumann_boundary() {
        let mesh = create_simple_mesh();

        // Remove cell2 and its relationships to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        // Initialize field
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Neumann(2.0));

        // Initialize gradient calculator
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Compute the gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(
            result.is_ok(),
            "Gradient calculation failed with result: {:?}",
            result
        );

        let grad = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient: {:?}", grad);

        // Corrected expected gradient based on computations
        let expected_grad = [0.0, 0.0, 6.0];
        for i in 0..3 {
            assert!(
                (grad[i] - expected_grad[i]).abs() < 1e-6,
                "Gradient component {} does not match expected value",
                i
            );
        }
    }

    #[test]
    fn test_gradient_functional_boundary() {
        let mesh = create_simple_mesh();

        // Remove cell2 and its relationships to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        // Initialize field
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::DirichletFn(Arc::new(|time, _| 1.0 + time)),
        );

        // Initialize gradient calculator
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Compute the gradient at time = 2.0
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 2.0);
        assert!(
            result.is_ok(),
            "Gradient calculation failed with result: {:?}",
            result
        );

        let grad = gradient
            .restrict(&MeshEntity::Cell(1))
            .expect("Gradient not computed");
        println!("Computed gradient: {:?}", grad);

        // Expected gradient based on computations
        let expected_grad = [0.0, 0.0, 6.0];
        for i in 0..3 {
            assert!(
                (grad[i] - expected_grad[i]).abs() < 1e-6,
                "Gradient component {} does not match expected value",
                i
            );
        }
    }

    #[test]
    fn test_gradient_missing_data() {
        let mesh = Mesh::new();

        // Create cell without any faces
        let cell = MeshEntity::Cell(1);
        mesh.add_entity(cell);

        // Initialize empty field
        let field = Section::<f64>::new();

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Attempt to compute gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(
            result.is_err(),
            "Expected error due to missing field values"
        );

        println!("Expected error: {:?}", result.err());
    }

    #[test]
    fn test_gradient_unimplemented_robin_condition() {
        let mesh = create_simple_mesh();

        // Remove cell2 and its relationships to simulate a boundary
        mesh.entities.write().unwrap().remove(&MeshEntity::Cell(2));
        mesh.sieve.adjacency.remove(&MeshEntity::Cell(2));

        // Initialize field
        let field = Section::<f64>::new();
        field.set_data(MeshEntity::Cell(1), 1.0);

        // Initialize gradient
        let mut gradient = Section::<[f64; 3]>::new();
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(1),
            BoundaryCondition::Robin {
                alpha: 1.0,
                beta: 2.0,
            },
        );

        // Initialize gradient calculator
        let mut gradient_calculator = Gradient::new(&mesh, &boundary_handler);

        // Attempt to compute gradient
        let result = gradient_calculator.compute_gradient(&field, &mut gradient, 0.0);
        assert!(
            result.is_err(),
            "Expected error due to unimplemented Robin condition"
        );

        println!("Expected error: {:?}", result.err());
    }
}
```

---

`src/equation/reconstruction/mod.rs`

```rust
pub mod reconstruct;
```

---

`src/equation/reconstruction/reconstruct.rs`

```rust
// src/equation/reconstruction/reconstruct.rs

/// Reconstructs the solution at a face center by extrapolating from the cell value 
/// and its gradient. This approach is critical for finite volume methods as it
/// provides a face-centered scalar value, which is essential for flux calculations.
///
/// # Arguments
///
/// * `cell_value` - The scalar field value at the cell center, representing the primary
///                  field quantity (e.g., temperature, pressure).
/// * `gradient` - The gradient vector `[f64; 3]` representing the rate of change of the
///                scalar field within the cell in each spatial direction. This gradient
///                allows for a linear approximation of the field near the cell center.
/// * `cell_center` - Coordinates `[f64; 3]` of the cell center, where `cell_value` and
///                   `gradient` are defined.
/// * `face_center` - Coordinates `[f64; 3]` of the face center where the scalar field 
///                   value is to be reconstructed.
///
/// # Returns
///
/// The reconstructed scalar field value at the face center, determined by linearly 
/// extrapolating from the cell center using the gradient.
///
/// # Example
///
/// ```rust
/// use hydra::equation::reconstruction::reconstruct::reconstruct_face_value;
///
/// let cell_value = 1.0;
/// let gradient = [2.0, 0.0, 0.0];
/// let cell_center = [0.0, 0.0, 0.0];
/// let face_center = [0.5, 0.0, 0.0];
///
/// let reconstructed_value = reconstruct_face_value(cell_value, gradient, cell_center, face_center);
/// assert_eq!(reconstructed_value, 2.0);
/// ```
pub fn reconstruct_face_value(
    cell_value: f64,
    gradient: [f64; 3],
    cell_center: [f64; 3],
    face_center: [f64; 3],
) -> f64 {
    let delta = [
        face_center[0] - cell_center[0],
        face_center[1] - cell_center[1],
        face_center[2] - cell_center[2],
    ];
    cell_value + gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2]
}

/// Unit tests for `reconstruct_face_value`.
///
/// Tests a variety of scenarios to ensure correct reconstruction of values
/// at face centers, verifying the linear extrapolation approach based on the
/// provided gradient and cell/face positions.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct_face_value() {
        // Test case 1: Gradient in the x-direction
        let cell_value = 1.0;
        let gradient = [2.0, 0.0, 0.0]; // Gradient along x-axis
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [0.5, 0.0, 0.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: cell_value + gradient_x * delta_x = 1.0 + 2.0 * 0.5 = 2.0
        let expected_value = 2.0;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );

        // Test case 2: Gradient in the y-direction
        let cell_value = 3.0;
        let gradient = [0.0, -1.0, 0.0]; // Gradient along negative y-axis
        let cell_center = [1.0, 1.0, 0.0];
        let face_center = [1.0, 0.5, 0.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: 3.0 + (-1.0) * (-0.5) = 3.0 + 0.5 = 3.5
        let expected_value = 3.5;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );

        // Test case 3: Gradient in all directions
        let cell_value = 0.0;
        let gradient = [1.0, 2.0, 3.0];
        let cell_center = [0.0, 0.0, 0.0];
        let face_center = [1.0, 1.0, 1.0];

        let reconstructed_value = reconstruct_face_value(
            cell_value,
            gradient,
            cell_center,
            face_center,
        );

        // Expected value: 0.0 + 1*1 + 2*1 + 3*1 = 6.0
        let expected_value = 6.0;
        assert!(
            (reconstructed_value - expected_value).abs() < 1e-6,
            "Reconstructed value does not match expected value. Expected {}, got {}",
            expected_value,
            reconstructed_value
        );
    }
}
```

---

`src/solver/mod.rs`

```rust
//! Main module for the solver interface in Hydra.
//!
//! This module houses the Krylov solvers and preconditioners,
//! facilitating flexible solver selection.
//! 
pub mod ksp;
pub mod cg;
pub mod preconditioner;
pub mod gmres;

pub use ksp::KSP;
pub use cg::ConjugateGradient;
pub use gmres::GMRES;

#[cfg(test)]
mod tests;
```

---

`src/solver/ksp.rs`

```rust
//! Defines the Krylov Subspace Method (KSP) trait for solver implementation in Hydra.
//!
//! This trait standardizes methods for solving linear systems across different Krylov methods.

use crate::linalg::{Matrix, Vector};

#[derive(Debug)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// KSP trait for Krylov solvers, encompassing solvers like CG and GMRES.
pub trait KSP {
    fn solve(
        &mut self, 
        a: &dyn Matrix<Scalar = f64>, 
        b: &dyn Vector<Scalar = f64>, 
        x: &mut dyn Vector<Scalar = f64>
    ) -> SolverResult;
}

```

---

`src/solver/preconditioner/mod.rs`

```rust
pub mod jacobi;
pub mod lu;
pub mod ilu;
pub mod cholesky;

pub use jacobi::Jacobi;
pub use lu::LU;
pub use ilu::ILU;
pub use cholesky::CholeskyPreconditioner;

use crate::linalg::{Matrix, Vector};

// Preconditioner trait
pub trait Preconditioner {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}
```