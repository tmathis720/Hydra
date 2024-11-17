<<<<<<< HEAD
=======
I am attempting to troubleshoot some problems in the implementation of the energy equation for the FVM code I am developing, called Hydra. I will provide you with the test failure outputs first, and then provide source code to help clarify and provide the necessary information to formulate a complete solution.

Here is the failure output:

```bash 
failures:

---- equation::energy_equation::tests::test_flux_calculation_with_dirichlet_boundary_condition stdout ----
Face: Face(1), Cell: Cell(1), Temp: 300, Grad Temp: [10.0, 0.0, 0.0], Face Temp: 300.8333333333333
Face: Face(1), Normal: [0.0, 0.0, 1.0], Velocity: [2.0, 0.0, 0.0]
Face: Face(1), Conductive Flux: -0, Convective Flux: 0
Storing total flux: 0 for face Face(1)
thread 'equation::energy_equation::tests::test_flux_calculation_with_dirichlet_boundary_condition' panicked at src\equation\energy_equation.rs:262:9:
assertion `left == right` failed: Flux should match Dirichlet boundary value.
  left: 0.0
 right: 100.0
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- equation::energy_equation::tests::test_flux_calculation_with_neumann_boundary_condition stdout ----
Face: Face(1), Cell: Cell(2), Temp: 310, Grad Temp: [10.0, 0.0, 0.0], Face Temp: 310.8333333333333
Face: Face(1), Normal: [0.0, 0.0, 1.0], Velocity: [2.0, 0.0, 0.0]
Face: Face(1), Conductive Flux: -0, Convective Flux: 0
Storing total flux: 0 for face Face(1)
thread 'equation::energy_equation::tests::test_flux_calculation_with_neumann_boundary_condition' panicked at src\equation\energy_equation.rs:301:9:
Flux should be adjusted by Neumann boundary condition.

---- equation::energy_equation::tests::test_flux_calculation_with_robin_boundary_condition stdout ----
Face: Face(1), Cell: Cell(2), Temp: 310, Grad Temp: [10.0, 0.0, 0.0], Face Temp: 310.8333333333333
Face: Face(1), Normal: [0.0, 0.0, 1.0], Velocity: [2.0, 0.0, 0.0]
Face: Face(1), Conductive Flux: -0, Convective Flux: 0
Storing total flux: 0 for face Face(1)
thread 'equation::energy_equation::tests::test_flux_calculation_with_robin_boundary_condition' panicked at src\equation\energy_equation.rs:340:9:
Flux should be affected by Robin boundary conditions.


failures:
    equation::energy_equation::tests::test_flux_calculation_with_dirichlet_boundary_condition
    equation::energy_equation::tests::test_flux_calculation_with_neumann_boundary_condition
    equation::energy_equation::tests::test_flux_calculation_with_robin_boundary_condition
```

Here is the source code for `src/equation/equation.rs` :

```rust
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
```

---

`src/equation/fields.rs`

```rust
// src/equation/fields.rs

use crate::domain::Section;

pub struct Fields {
    pub field: Section<f64>, // For primary variables like pressure
    pub gradient: Section<[f64; 3]>,
    pub velocity_field: Section<[f64; 3]>,

    // Additional fields for energy and turbulence
    pub temperature_field: Section<f64>,
    pub temperature_gradient: Section<[f64; 3]>,
    pub k_field: Section<f64>,         // Turbulent kinetic energy
    pub epsilon_field: Section<f64>,   // Turbulent dissipation rate
}

pub struct Fluxes {
    pub momentum_fluxes: Section<f64>,
    pub energy_fluxes: Section<f64>,
    pub turbulence_fluxes: Section<f64>,
}
```

---

`src/equation/manager.rs`

```rust
// src/equation/manager.rs

use super::{PhysicalEquation, Fields, Fluxes};
use crate::domain::mesh::Mesh;
use crate::boundary::bc_handler::BoundaryConditionHandler;

pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
}

impl EquationManager {
    pub fn new() -> Self {
        EquationManager {
            equations: Vec::new(),
        }
    }

    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        for equation in &self.equations {
            equation.assemble(domain, fields, fluxes, boundary_handler);
        }
    }
}
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

`src/domain/mesh/mod.rs`

```rust
pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};

// Delegate methods to corresponding modules

/// Represents the mesh structure, which is composed of a sieve for entity management,  
/// a set of mesh entities, vertex coordinates, and channels for boundary data.  
/// 
/// The `Mesh` struct is the central component for managing mesh entities and  
/// their relationships. It stores entities such as vertices, edges, faces,  
/// and cells, along with their geometric data and boundary-related information.  
/// 
/// Example usage:
/// 
///    let mesh = Mesh::new();  
///    let entity = MeshEntity::Vertex(1);  
///    mesh.entities.write().unwrap().insert(entity);  
/// 
#[derive(Clone, Debug)]
pub struct Mesh {
    /// The sieve structure used for organizing the mesh entities' relationships.  
    pub sieve: Arc<Sieve>,  
    /// A thread-safe, read-write lock for managing mesh entities.  
    /// This set contains all `MeshEntity` objects in the mesh.  
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,  
    /// A map from vertex indices to their 3D coordinates.  
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,  
    /// An optional channel sender for transmitting boundary data related to mesh entities.  
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// An optional channel receiver for receiving boundary data related to mesh entities.  
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,  
}

impl Mesh {
    /// Creates a new instance of the `Mesh` struct with initialized components.  
    /// 
    /// This method sets up the sieve, entity set, vertex coordinate map,  
    /// and a channel for boundary data communication between mesh components.  
    ///
    /// The `Sender` and `Receiver` are unbounded channels used to pass boundary  
    /// data between mesh modules asynchronously.
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    assert!(mesh.entities.read().unwrap().is_empty());  
    /// 
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }
}

#[cfg(test)]
pub mod tests;
```

---

`src/domain/sieve.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity`  
/// elements, organized in an adjacency map.
///
/// The adjacency map tracks directed relations between entities in the mesh.  
/// It supports operations such as adding relationships, querying direct  
/// relations (cones), and computing closure and star sets for entities.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// A thread-safe adjacency map where each key is a `MeshEntity`,  
    /// and the value is a set of `MeshEntity` objects related to the key.  
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new empty `Sieve` instance with an empty adjacency map.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.  
    /// The relationship is stored in the adjacency map from the `from` entity  
    /// to the `to` entity.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).  
    /// This operation is referred to as retrieving the cone of the entity.  
    /// Returns `None` if there are no related entities.
    pub fn cone(&self, point: &MeshEntity) -> Option<Vec<MeshEntity>> {
        self.adjacency.get(point).map(|cone| {
            cone.iter().map(|entry| entry.key().clone()).collect()
        })
    }

    /// Computes the closure of a given `MeshEntity`.  
    /// The closure includes the entity itself and all entities it covers (cones) recursively.
    pub fn closure(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());

        while !stack.is_empty() {
            let keys: Vec<MeshEntity> = stack.iter().map(|entry| entry.key().clone()).collect();
            for p in keys {
                if result.insert(p.clone(), ()).is_none() {
                    if let Some(cones) = self.cone(&p) {
                        for q in cones {
                            stack.insert(q, ());
                        }
                    }
                }
                stack.remove(&p);
            }
        }
        result
    }

    /// Computes the star of a given `MeshEntity`.  
    /// The star includes the entity itself and all entities that directly cover it (supports).
    pub fn star(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        result.insert(point.clone(), ());
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        result
    }

    /// Retrieves all entities that support the given entity (`point`).  
    /// These are the entities that have an arrow pointing to `point`.
    pub fn support(&self, point: &MeshEntity) -> Vec<MeshEntity> {
        let mut supports = Vec::new();
        self.adjacency.iter().for_each(|entry| {
            let from = entry.key();
            if entry.value().contains_key(point) {
                supports.push(from.clone());
            }
        });
        supports
    }

    /// Computes the meet operation for two entities, `p` and `q`.  
    /// This is the intersection of their closures.
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        let result = DashMap::new();

        closure_p.iter().for_each(|entry| {
            let key = entry.key();
            if closure_q.contains_key(key) {
                result.insert(key.clone(), ());
            }
        });

        result
    }

    /// Computes the join operation for two entities, `p` and `q`.  
    /// This is the union of their stars.
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let star_p = self.star(p);
        let star_q = self.star(q);
        let result = DashMap::new();

        star_p.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });
        star_q.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });

        result
    }

    /// Applies a given function in parallel to all adjacency map entries.  
    /// This function is executed concurrently over each entity and its  
    /// corresponding set of related entities.
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, Vec<MeshEntity>)) + Sync + Send,
    {
        // Collect entries from DashMap to avoid borrow conflicts
        let entries: Vec<_> = self.adjacency.iter().map(|entry| {
            let key = entry.key().clone();
            let values: Vec<MeshEntity> = entry.value().iter().map(|e| e.key().clone()).collect();
            (key, values)
        }).collect();

        // Execute in parallel over collected entries
        entries.par_iter().for_each(|entry| {
            func((&entry.0, entry.1.clone()));
        });
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies adding an arrow between two entities and querying  
    /// the cone of an entity works as expected.
    fn test_add_arrow_and_cone() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        sieve.add_arrow(vertex, edge);
        let cone_result = sieve.cone(&vertex).unwrap();
        assert!(cone_result.contains(&edge));
    }

    #[test]
    /// Test that verifies the closure of a vertex correctly includes  
    /// all transitive relationships and the entity itself.
    fn test_closure() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);
        sieve.add_arrow(vertex, edge);
        sieve.add_arrow(edge, face);
        let closure_result = sieve.closure(&vertex);
        assert!(closure_result.contains_key(&vertex));
        assert!(closure_result.contains_key(&edge));
        assert!(closure_result.contains_key(&face));
        assert_eq!(closure_result.len(), 3);
    }

    #[test]
    /// Test that verifies the support of an entity includes the  
    /// correct supporting entities.
    fn test_support() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let support_result = sieve.support(&edge);

        assert!(support_result.contains(&vertex));
        assert_eq!(support_result.len(), 1);
    }

    #[test]
    /// Test that verifies the star of an entity includes both the entity itself and  
    /// its immediate supports.
    fn test_star() {
        let sieve = Sieve::new();
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(edge, face);

        let star_result = sieve.star(&face);

        assert!(star_result.contains_key(&face));
        assert!(star_result.contains_key(&edge));
        assert_eq!(star_result.len(), 2);
    }

    #[test]
    /// Test that verifies the meet operation between two entities returns  
    /// the correct intersection of their closures.
    fn test_meet() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex1, edge);
        sieve.add_arrow(vertex2, edge);

        let meet_result = sieve.meet(&vertex1, &vertex2);

        assert!(meet_result.contains_key(&edge));
        assert_eq!(meet_result.len(), 1);
    }

    #[test]
    /// Test that verifies the join operation between two entities returns  
    /// the correct union of their stars.
    fn test_join() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        let join_result = sieve.join(&vertex1, &vertex2);

        assert!(join_result.contains_key(&vertex1), "Join result should contain vertex1");
        assert!(join_result.contains_key(&vertex2), "Join result should contain vertex2");
        assert_eq!(join_result.len(), 2);
    }
}
```

---

`src/boundary/bc_handler.rs`

```rust
use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// BoundaryCondition represents various types of boundary conditions
/// that can be applied to mesh entities.
#[derive(Clone)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}

/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl BoundaryConditionHandler {
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    /// Applies the boundary conditions to the system matrices and right-hand side vectors.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                let index = *entity_to_index.get(entity).unwrap();
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(matrix, rhs, index, alpha, beta);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, value);
                    }
                    BoundaryCondition::Mixed { gamma, delta } => {
                        let mixed_bc = MixedBC::new();
                        mixed_bc.apply_mixed(matrix, rhs, index, gamma, delta);
                    }
                    BoundaryCondition::Cauchy { lambda, mu } => {
                        let cauchy_bc = CauchyBC::new();
                        cauchy_bc.apply_cauchy(matrix, rhs, index, lambda, mu);
                    }
                }
            }
        }
    }
}

/// The BoundaryConditionApply trait defines the `apply` method, which is used to apply 
/// a boundary condition to a given mesh entity.
pub trait BoundaryConditionApply {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    );
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let index = *entity_to_index.get(entity).unwrap();
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, *flux);
            }
            BoundaryCondition::Robin { alpha, beta } => {
                let robin_bc = RobinBC::new();
                robin_bc.apply_robin(matrix, rhs, index, *alpha, *beta);
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, value);
            }
            BoundaryCondition::Mixed { gamma, delta } => {
                let mixed_bc = MixedBC::new();
                mixed_bc.apply_mixed(matrix, rhs, index, *gamma, *delta);
            }
            BoundaryCondition::Cauchy { lambda, mu } => {
                let cauchy_bc = CauchyBC::new();
                cauchy_bc.apply_cauchy(matrix, rhs, index, *lambda, *mu);
            }
        }
    }
}
```

---

`src/geometry/mod.rs`

```rust
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use std::sync::Mutex;

// Module for handling geometric data and computations
// 2D Shape Modules
pub mod quadrilateral;
pub mod triangle;
// 3D Shape Modules
pub mod tetrahedron;
pub mod hexahedron;
pub mod prism;
pub mod pyramid;

/// The `Geometry` struct stores geometric data for a mesh, including vertex coordinates, 
/// cell centroids, and volumes. It also maintains a cache of computed properties such as 
/// volume and centroid for reuse, optimizing performance by avoiding redundant calculations.
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>,  // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,         // Volumes of each cell
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>, // Cache for computed properties, with thread safety
}

/// The `GeometryCache` struct stores computed properties of geometric entities, 
/// including volume, centroid, and area, with an optional "dirty" flag for lazy evaluation.
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,  // Stores a precomputed normal vector for a face
}

/// `CellShape` enumerates the different cell shapes in a mesh, including:
/// * Tetrahedron
/// * Hexahedron
/// * Prism
/// * Pyramid
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// `FaceShape` enumerates the different face shapes in a mesh, including:
/// * Triangle
/// * Quadrilateral
#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}

impl Geometry {
    /// Initializes a new `Geometry` instance with empty data.
    pub fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            cell_centroids: Vec::new(),
            cell_volumes: Vec::new(),
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// Adds or updates a vertex in the geometry. If the vertex already exists,
    /// it updates its coordinates; otherwise, it adds a new vertex.
    ///
    /// # Arguments
    /// * `vertex_index` - The index of the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) {
        if vertex_index >= self.vertices.len() {
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }
        self.vertices[vertex_index] = coords;
        self.invalidate_cache();
    }

    /// Computes and returns the centroid of a specified cell using the cell's shape and vertices.
    /// Caches the result for reuse.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3] {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let centroid = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(&cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().centroid = Some(centroid);
        centroid
    }

    /// Computes the volume of a given cell using its shape and vertex coordinates.
    /// The computed volume is cached for efficiency.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64 {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let volume = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(&cell_vertices),
            CellShape::Prism => self.compute_prism_volume(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().volume = Some(volume);
        volume
    }

    /// Calculates Euclidean distance between two points in 3D space.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt()
    }

    /// Computes the area of a 2D face based on its shape, caching the result.
    pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.area) {
            return cached;
        }

        let area = match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        };

        self.cache.lock().unwrap().entry(face_id).or_default().area = Some(area);
        area
    }

    /// Computes the centroid of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the face centroid.
    pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        match face_shape {
            FaceShape::Triangle => self.compute_triangle_centroid(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_centroid(face_vertices),
        }
    }

    /// Computes and caches the normal vector for a face based on its shape.
    ///
    /// This function determines the face shape and calls the appropriate 
    /// function to compute the normal vector.
    ///
    /// # Arguments
    /// * `mesh` - A reference to the mesh.
    /// * `face` - The face entity for which to compute the normal.
    /// * `cell` - The cell associated with the face, used to determine the orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector, or `None` if it could not be computed.
    pub fn compute_face_normal(
        &mut self,
        mesh: &Mesh,
        face: &MeshEntity,
        _cell: &MeshEntity,
    ) -> Option<[f64; 3]> {
        let face_id = face.get_id();

        // Check if the normal is already cached
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal) {
            return Some(cached);
        }

        let face_vertices = mesh.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let normal = match face_shape {
            FaceShape::Triangle => self.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_normal(&face_vertices),
        };

        // Cache the normal vector for future use
        self.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal);

        Some(normal)
    }

    /// Invalidate the cache when geometry changes (e.g., vertex updates).
    fn invalidate_cache(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    /// Computes the total volume of all cells.
    pub fn compute_total_volume(&self) -> f64 {
        self.cell_volumes.par_iter().sum()
    }

    /// Updates all cell volumes in parallel using mesh information.
    pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
        let new_volumes: Vec<f64> = mesh
            .get_cells()
            .par_iter()
            .map(|cell| {
                let mut temp_geometry = Geometry::new();
                temp_geometry.compute_cell_volume(mesh, cell)
            })
            .collect();

        self.cell_volumes = new_volumes;
    }

    /// Computes the total centroid of all cells.
    pub fn compute_total_centroid(&self) -> [f64; 3] {
        let total_centroid: [f64; 3] = self.cell_centroids
            .par_iter()
            .cloned()
            .reduce(
                || [0.0, 0.0, 0.0],
                |acc, centroid| [
                    acc[0] + centroid[0],
                    acc[1] + centroid[1],
                    acc[2] + centroid[2],
                ],
            );

        let num_centroids = self.cell_centroids.len() as f64;
        [
            total_centroid[0] / num_centroids,
            total_centroid[1] / num_centroids,
            total_centroid[2] / num_centroids,
        ]
    }
}


#[cfg(test)]
mod tests {
    use crate::geometry::{Geometry, CellShape, FaceShape};
    use crate::domain::{MeshEntity, mesh::Mesh, Sieve};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_set_vertex() {
        let mut geometry = Geometry::new();

        // Set vertex at index 0
        geometry.set_vertex(0, [1.0, 2.0, 3.0]);
        assert_eq!(geometry.vertices[0], [1.0, 2.0, 3.0]);

        // Update vertex at index 0
        geometry.set_vertex(0, [4.0, 5.0, 6.0]);
        assert_eq!(geometry.vertices[0], [4.0, 5.0, 6.0]);

        // Set vertex at a higher index
        geometry.set_vertex(3, [7.0, 8.0, 9.0]);
        assert_eq!(geometry.vertices[3], [7.0, 8.0, 9.0]);

        // Ensure the intermediate vertices are initialized to [0.0, 0.0, 0.0]
        assert_eq!(geometry.vertices[1], [0.0, 0.0, 0.0]);
        assert_eq!(geometry.vertices[2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_distance() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 4.0, 0.0];

        let distance = Geometry::compute_distance(&p1, &p2);

        // The expected distance is 5 (Pythagoras: sqrt(3^2 + 4^2))
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_compute_cell_centroid_tetrahedron() {
        // Create a new Sieve and Mesh.
        let sieve = Sieve::new();
        let mut mesh = Mesh {
            sieve: Arc::new(sieve),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: None,
            boundary_data_receiver: None,
        };

        // Define vertices and a cell.
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
        let cell = MeshEntity::Cell(1);

        // Set vertex coordinates.
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);

        // Add entities to the mesh.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.add_entity(cell);

        // Establish relationships between the cell and vertices.
        mesh.add_arrow(cell, vertex1);
        mesh.add_arrow(cell, vertex2);
        mesh.add_arrow(cell, vertex3);
        mesh.add_arrow(cell, vertex4);

        // Verify that `get_cell_vertices` retrieves the correct vertices.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        assert_eq!(cell_vertices.len(), 4, "Expected 4 vertices for a tetrahedron cell.");

        // Validate the shape before computing.
        assert_eq!(mesh.get_cell_shape(&cell), Ok(CellShape::Tetrahedron));

        // Create a Geometry instance and compute the centroid.
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(&mesh, &cell);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_compute_face_area_triangle() {
        let mut geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_face_area(1, FaceShape::Triangle, &triangle_vertices);

        // Expected area: 0.5 * base * height = 0.5 * 3.0 * 4.0 = 6.0
        assert_eq!(area, 6.0);
    }

    #[test]
    fn test_compute_face_centroid_quadrilateral() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let centroid = geometry.compute_face_centroid(FaceShape::Quadrilateral, &quad_vertices);

        // Expected centroid is the geometric center: (0.5, 0.5, 0.0)
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_compute_total_volume() {
        let mut geometry = Geometry::new();
        let _mesh = Mesh::new();

        // Example setup: Define cells with known volumes
        // Here, you would typically define several cells and their volumes for the test
        geometry.cell_volumes = vec![1.0, 2.0, 3.0];

        // Expected total volume is the sum of individual cell volumes: 1.0 + 2.0 + 3.0 = 6.0
        assert_eq!(geometry.compute_total_volume(), 6.0);
    }

    #[test]
    fn test_compute_face_normal_triangle() {
        let geometry = Geometry::new();
        
        // Define vertices for a triangular face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        // Define the face as a triangle
        let _face = MeshEntity::Face(1);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal without setting up mesh connectivity
        let normal = geometry.compute_triangle_normal(&vertices);

        // Expected normal for a triangle in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_quadrilateral() {
        let geometry = Geometry::new();

        // Define vertices for a quadrilateral face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        // Define the face as a quadrilateral
        let _face = MeshEntity::Face(2);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal for quadrilateral
        let normal = geometry.compute_quadrilateral_normal(&vertices);

        // Expected normal for a quadrilateral in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_caching() {
        let geometry = Geometry::new();

        // Define vertices for a triangular face
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        let face_id = 3; // Unique identifier for caching
        let _face = MeshEntity::Face(face_id);
        let _cell = MeshEntity::Cell(1);

        // First computation to populate the cache
        let normal_first = geometry.compute_triangle_normal(&vertices);

        // Manually retrieve from cache to verify caching behavior
        geometry.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal_first);
        let cached_normal = geometry.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal);

        // Verify that the cached value matches the first computed value
        assert_eq!(Some(normal_first), cached_normal);
    }

    #[test]
    fn test_compute_face_normal_unsupported_shape() {
        let geometry = Geometry::new();

        // Define vertices for a pentagon (unsupported)
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.5, 0.5, 0.0], // vertex 5
        ];

        let _face = MeshEntity::Face(4);
        let _cell = MeshEntity::Cell(1);

        // Since compute_face_normal expects only triangles or quadrilaterals, it should return None
        let face_shape = match vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return, // Unsupported shape, skip test
        };

        // Attempt to compute the normal for an unsupported shape
        let normal = match face_shape {
            FaceShape::Triangle => Some(geometry.compute_triangle_normal(&vertices)),
            FaceShape::Quadrilateral => Some(geometry.compute_quadrilateral_normal(&vertices)),
        };

        // Assert that the function correctly handles unsupported shapes by skipping normal computation
        assert!(normal.is_none());
    }
}
```

---

`src/solver/ksp.rs`

```rust
//! Enhancements to the KSP module to introduce an interface adapter for flexible usage.
//!
//! This adds the `SolverManager` for high-level integration of solvers and preconditioners.

use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::sync::Arc;

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
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}

/// Struct representing a high-level interface for managing solver configuration.
pub struct SolverManager {
    solver: Box<dyn KSP>,
    preconditioner: Option<Arc<dyn Preconditioner>>,
}

impl SolverManager {
    /// Creates a new `SolverManager` instance with a specified solver.
    ///
    /// # Arguments
    /// - `solver`: The Krylov solver to be used.
    ///
    /// # Returns
    /// A new `SolverManager` instance.
    pub fn new(solver: Box<dyn KSP>) -> Self {
        SolverManager {
            solver,
            preconditioner: None,
        }
    }

    /// Sets a preconditioner for the solver.
    ///
    /// # Arguments
    /// - `preconditioner`: The preconditioner to be used.
    pub fn set_preconditioner(&mut self, preconditioner: Arc<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }

    /// Solves a system `Ax = b` using the configured solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `a`: The system matrix `A`.
    /// - `b`: The right-hand side vector `b`.
    /// - `x`: The solution vector `x`, which will be updated with the computed solution.
    ///
    /// # Returns
    /// A `SolverResult` containing convergence information and the final residual norm.
    pub fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(a, b, x);
        }
        self.solver.solve(a, b, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::Jacobi;
    use crate::solver::cg::ConjugateGradient;
    use faer::{mat, Mat};

    #[test]
    fn test_solver_manager_with_jacobi_preconditioner() {
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];
        let b = mat![
            [1.0],
            [2.0],
        ];
        let mut x = Mat::<f64>::zeros(2, 1);

        // Initialize CG solver and solver manager
        let cg_solver = ConjugateGradient::new(100, 1e-6);
        let mut solver_manager = SolverManager::new(Box::new(cg_solver));

        // Set Jacobi preconditioner
        let jacobi_preconditioner = Arc::new(Jacobi::default());
        solver_manager.set_preconditioner(jacobi_preconditioner);

        // Solve the system
        let result = solver_manager.solve(&a, &b, &mut x);

        // Validate results
        assert!(result.converged, "Solver did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
        assert!(
            !crate::linalg::vector::traits::Vector::as_slice(&x).contains(&f64::NAN),
            "Solution contains NaN values"
        );
    }
}
```
>>>>>>> 4c1a47574ec839f290abebf543fee85826ad06c9
