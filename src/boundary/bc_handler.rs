//! Boundary Condition Module
//!
//! This module provides functionality for defining and applying various types of boundary
//! conditions (Dirichlet, Neumann, Robin, etc.) to mesh entities in a computational fluid
//! dynamics (CFD) simulation.
//!
//! # Overview
//! - `BoundaryCondition`: Enum representing supported boundary condition types.
//! - `BoundaryConditionHandler`: Manages boundary conditions for mesh entities.
//! - `BoundaryConditionApply`: Trait for applying boundary conditions to system matrices.
//!
//! # Computational Context
//! Boundary conditions play a crucial role in solving partial differential equations (PDEs)
//! in CFD. This module ensures compatibility with Hydra's unstructured grid and time-stepping
//! framework.

use dashmap::DashMap;
use log::{error, info};
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use crate::boundary::solid_wall::SolidWallBC;
use crate::boundary::far_field::FarFieldBC;
use crate::boundary::injection::InjectionBC;
use crate::boundary::inlet_outlet::InletOutletBC;
use crate::boundary::periodic::PeriodicBC;
use crate::boundary::symmetry::SymmetryBC;
use faer::MatMut;

use super::{log_boundary_error, log_boundary_info, log_boundary_warning, BoundaryError};

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// Wrapper for function-based boundary conditions, allowing metadata for equality and debug.
#[derive(Clone)]
pub struct FunctionWrapper {
    pub description: String, // Metadata to identify the function
    pub function: BoundaryConditionFn,
}

impl PartialEq for FunctionWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.description == other.description
    }
}

impl std::fmt::Debug for FunctionWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FunctionWrapper {{ description: {} }}", self.description)
    }
}

/// Enum defining various boundary condition types.
///
/// # Variants
/// - `Dirichlet(f64)`: Specifies a fixed value at the boundary.
/// - `Neumann(f64)`: Specifies a fixed flux at the boundary.
/// - `Robin { alpha, beta }`: Combines Dirichlet and Neumann conditions.
/// - `Mixed { gamma, delta }`: A hybrid boundary condition.
/// - `Cauchy { lambda, mu }`: Used in fluid-structure interaction problems.
/// - `SolidWallInviscid`: Represents a slip wall boundary condition.
/// - `SolidWallViscous { normal_velocity: f64 }`: Represents a no-slip wall boundary condition.
/// - `DirichletFn(FunctionWrapper)`: Functional Dirichlet condition with metadata.
/// - `NeumannFn(FunctionWrapper)`: Functional Neumann condition with metadata.
/// - `Periodic { pairs }`: Specifies a Periodic boundary condition between pairs.
/// - `FarField(f64)`: Represents far-field boundary conditions for simulating infinite domains.
/// - `Injection(f64)`: Represents injection of a specified property (e.g., mass, momentum, or energy).
/// - `InletOutlet`: Represents combined inlet and outlet conditions.
/// - `Symmetry`: Represents symmetry plane conditions.
///
/// # Notes
/// Functional boundary conditions allow time-dependent or spatially varying constraints.
#[derive(Clone, PartialEq, Debug)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    SolidWallInviscid,
    SolidWallViscous { normal_velocity: f64 },
    DirichletFn(FunctionWrapper),
    NeumannFn(FunctionWrapper),
    Periodic { pairs: Vec<(MeshEntity, MeshEntity)> },
    FarField(f64),
    Injection(f64),
    InletOutlet,
    Symmetry,
}



/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

lazy_static! {
    static ref GLOBAL_BC_HANDLER: Arc<RwLock<BoundaryConditionHandler>> =
        Arc::new(RwLock::new(BoundaryConditionHandler::new()));
}

impl BoundaryConditionHandler {
    /// Retrieves the coordinates of a mesh entity.
    fn get_coordinates(&self, entity: &MeshEntity) -> Result<Vec<f64>, BoundaryError> {
        // Placeholder implementation, replace with actual logic to get coordinates
        Ok(vec![0.0, 0.0, 0.0])
    }
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    pub fn global() -> Arc<RwLock<BoundaryConditionHandler>> {
        GLOBAL_BC_HANDLER.clone()
    }

    /// Sets a boundary condition for a mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) -> Result<(), BoundaryError> {
        if self.conditions.contains_key(&entity) {
            log_boundary_warning(&format!(
                "Overwriting existing boundary condition for entity {:?}.",
                entity
            ));
        }
        self.conditions.insert(entity.clone(), condition);
        log_boundary_info(&format!("Boundary condition set for {:?}", entity));
        Ok(())
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    pub fn get_boundary_faces(&self) -> Vec<MeshEntity> {
        self.conditions.iter()
            .map(|entry| entry.key().clone()) // Extract the keys (MeshEntities) from the map
            .filter(|entity| matches!(entity, MeshEntity::Face(_))) // Filter for Face entities
            .collect()
    }

    /// Applies boundary conditions to system matrices and RHS vectors.
    ///
    /// # Parameters
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `rhs`: Mutable reference to the right-hand side vector.
    /// - `boundary_entities`: List of mesh entities to which boundary conditions are applied.
    /// - `entity_to_index`: Maps mesh entities to matrix indices.
    /// - `time`: Current simulation time for time-dependent conditions.
    ///
    /// # Computational Notes
    /// Modifies matrix coefficients and RHS values based on the type of boundary condition.
    /// Ensures consistency with finite volume and finite element methods as per Hydra's framework.
    ///
    /// # Returns
    /// Returns `Ok(())` if all boundary conditions are successfully applied,
    /// otherwise returns a `BoundaryError`.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError> {
        for entity in boundary_entities {
            let bc = self.get_bc(entity).ok_or_else(|| {
                let err = BoundaryError::EntityNotFound(format!(
                    "Entity {:?} not found in boundary conditions.",
                    entity
                ));
                error!("{}", err);
                err
            })?;

            let index = entity_to_index.get(entity).map(|idx| *idx).ok_or_else(|| {
                let err = BoundaryError::EntityNotFound(format!(
                    "Entity {:?} missing from entity_to_index mapping.",
                    entity
                ));
                error!("{}", err);
                err
            })?;

            match bc {
                BoundaryCondition::Dirichlet(value) => {
                    DirichletBC::new().apply_constant_dirichlet(matrix, rhs, index, value);
                    info!("Applied Dirichlet condition ({}) to entity {:?}", value, entity);
                }
                BoundaryCondition::Neumann(flux) => {
                    NeumannBC::new().apply_constant_neumann(rhs, index, flux);
                    info!("Applied Neumann condition ({}) to entity {:?}", flux, entity);
                }
                BoundaryCondition::Robin { alpha, beta } => {
                    RobinBC::new().apply_robin(matrix, rhs, index, alpha, beta);
                    info!(
                        "Applied Robin condition (alpha: {}, beta: {}) to entity {:?}",
                        alpha, beta, entity
                    );
                }
                BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                    SolidWallBC::new().apply_bc(matrix, rhs, entity_to_index)?;
                    info!("Applied SolidWall condition to entity {:?}", entity);
                }
                BoundaryCondition::DirichletFn(wrapper) => {
                    let coords = self.get_coordinates(entity)?;
                    let value = (wrapper.function)(time, &coords);
                    DirichletBC::new().apply_constant_dirichlet(matrix, rhs, index, value);
                    info!(
                        "Applied functional Dirichlet condition (computed value: {}) to entity {:?}",
                        value, entity
                    );
                }
                BoundaryCondition::NeumannFn(wrapper) => {
                    let coords = self.get_coordinates(entity)?;
                    let value = (wrapper.function)(time, &coords);
                    NeumannBC::new().apply_constant_neumann(rhs, index, value);
                    info!(
                        "Applied functional Neumann condition (computed value: {}) to entity {:?}",
                        value, entity
                    );
                }
                BoundaryCondition::Mixed { gamma, delta } => {
                    MixedBC::new().apply_mixed(matrix, rhs, index, gamma, delta);
                    info!(
                        "Applied Mixed condition (gamma: {}, delta: {}) to entity {:?}",
                        gamma, delta, entity
                    );
                }
                BoundaryCondition::Cauchy { lambda, mu } => {
                    CauchyBC::new().apply_cauchy(matrix, rhs, index, lambda, mu);
                    info!(
                        "Applied Cauchy condition (lambda: {}, mu: {}) to entity {:?}",
                        lambda, mu, entity
                    );
                }
                BoundaryCondition::FarField(value) => {
                    FarFieldBC::new().apply_far_field(matrix, rhs, index, value);
                    info!("Applied FarField condition ({}) to entity {:?}", value, entity);
                }
                BoundaryCondition::Injection(value) => {
                    InjectionBC::new().apply_injection(matrix, rhs, index, value);
                    info!("Applied Injection condition ({}) to entity {:?}", value, entity);
                }
                BoundaryCondition::InletOutlet => {
                    InletOutletBC::new().apply_bc(matrix, rhs, entity_to_index)?;
                    info!("Applied InletOutlet condition to entity {:?}", entity);
                }
                BoundaryCondition::Symmetry => {
                    SymmetryBC::new().apply_symmetry_plane(matrix, rhs, index);
                    info!("Applied Symmetry condition to entity {:?}", entity);
                }
                BoundaryCondition::Periodic { pairs } => {
                    let periodic_bc = PeriodicBC::new();
                    for (entity1, entity2) in pairs.clone() {
                        periodic_bc.set_pair(entity1.clone(), entity2.clone());
                    }
                    periodic_bc.apply_bc(matrix, rhs, entity_to_index)?;
                    info!(
                        "Applied Periodic condition to entity {:?} with pairs {:?}",
                        entity, pairs
                    );
                }
                _ => {
                    let err = BoundaryError::InvalidBoundaryType(format!(
                        "Unsupported boundary condition {:?} for entity {:?}",
                        bc, entity
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }
        }
        Ok(())
    }



}


/// The BoundaryConditionApply trait defines the `apply` method, which applies 
/// a boundary condition to a given mesh entity.
pub trait BoundaryConditionApply {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError>;
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError> {
        let index = match entity_to_index.get(entity) {
            Some(idx) => *idx,
            None => {
                let err = BoundaryError::EntityNotFound(format!(
                    "Entity {:?} not found in entity_to_index mapping.",
                    entity
                ));
                log_boundary_error(&err);
                return Err(err);
            }
        };

        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, *value);
                log_boundary_info(&format!(
                    "Applied Dirichlet condition ({}) to entity {:?}",
                    value, entity
                ));
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, *flux);
                log_boundary_info(&format!(
                    "Applied Neumann condition ({}) to entity {:?}",
                    flux, entity
                ));
            }
            BoundaryCondition::Robin { alpha, beta } => {
                let robin_bc = RobinBC::new();
                robin_bc.apply_robin(matrix, rhs, index, *alpha, *beta);
                log_boundary_info(&format!(
                    "Applied Robin condition (alpha: {}, beta: {}) to entity {:?}",
                    alpha, beta, entity
                ));
            }
            BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                let solid_wall_bc = SolidWallBC::new();
                solid_wall_bc.apply_bc(matrix, rhs, entity_to_index);
                log_boundary_info(&format!(
                    "Applied SolidWall condition to entity {:?}",
                    entity
                ));
            }
            BoundaryCondition::DirichletFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0]; // Replace with actual entity coordinates if available
                let value = (wrapper.function)(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                log_boundary_info(&format!(
                    "Applied functional Dirichlet condition (computed value: {}) to entity {:?}",
                    value, entity
                ));
            }
            BoundaryCondition::NeumannFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0];
                let value = (wrapper.function)(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, value);
                log_boundary_info(&format!(
                    "Applied functional Neumann condition (computed value: {}) to entity {:?}",
                    value, entity
                ));
            }
            BoundaryCondition::Mixed { gamma, delta } => {
                let mixed_bc = MixedBC::new();
                mixed_bc.apply_mixed(matrix, rhs, index, *gamma, *delta);
                log_boundary_info(&format!(
                    "Applied Mixed condition (gamma: {}, delta: {}) to entity {:?}",
                    gamma, delta, entity
                ));
            }
            BoundaryCondition::Cauchy { lambda, mu } => {
                let cauchy_bc = CauchyBC::new();
                cauchy_bc.apply_cauchy(matrix, rhs, index, *lambda, *mu);
                log_boundary_info(&format!(
                    "Applied Cauchy condition (lambda: {}, mu: {}) to entity {:?}",
                    lambda, mu, entity
                ));
            }
            BoundaryCondition::FarField(value) => {
                let far_field_bc = FarFieldBC::new();
                far_field_bc.apply_far_field(matrix, rhs, index, *value);
                log_boundary_info(&format!(
                    "Applied FarField condition ({}) to entity {:?}",
                    value, entity
                ));
            }
            BoundaryCondition::Injection(value) => {
                let injection_bc = InjectionBC::new();
                injection_bc.apply_injection(matrix, rhs, index, *value);
                log_boundary_info(&format!(
                    "Applied Injection condition ({}) to entity {:?}",
                    value, entity
                ));
            }
            BoundaryCondition::InletOutlet => {
                let inlet_outlet_bc = InletOutletBC::new();
                inlet_outlet_bc.apply_bc(matrix, rhs, entity_to_index);
                log_boundary_info(&format!(
                    "Applied InletOutlet condition to entity {:?}",
                    entity
                ));
            }
            BoundaryCondition::Symmetry => {
                let symmetry_bc = SymmetryBC::new();
                symmetry_bc.apply_symmetry_plane(matrix, rhs, index);
                log_boundary_info(&format!(
                    "Applied Symmetry condition to entity {:?}",
                    entity
                ));
            }
            BoundaryCondition::Periodic { pairs } => {
                let periodic_bc = PeriodicBC::new();
                for (entity1, entity2) in pairs {
                    periodic_bc.set_pair(entity1.clone(), entity2.clone());
                }
                periodic_bc.apply_bc(matrix, rhs, entity_to_index);
                log_boundary_info(&format!(
                    "Applied Periodic condition to entity {:?} with pairs {:?}",
                    entity, pairs
                ));
            }
            _ => {
                let err = BoundaryError::InvalidBoundaryType(format!(
                    "Unsupported boundary condition {:?} for entity {:?}",
                    self, entity
                ));
                log_boundary_error(&err);
                return Err(err);
            }
        }
        Ok(())
    }
}



#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;

    #[test]
    fn test_set_and_get_boundary_conditions() {
        let handler = BoundaryConditionHandler::new();
        let entity = MeshEntity::Face(1);
        let condition = BoundaryCondition::Dirichlet(10.0);

        handler.set_bc(entity.clone(), condition.clone());

        assert_eq!(handler.get_bc(&entity), Some(condition));
    }

    #[test]
    fn test_apply_dirichlet_condition() {
        // Create a matrix and RHS using Mat, then obtain mutable views
        let mut matrix = Mat::<f64>::zeros(3, 3);
        let mut rhs = Mat::<f64>::zeros(3, 1);
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Define the mesh entity and its mapping to matrix indices
        let entity = MeshEntity::Face(1);
        let index_map = DashMap::new();
        index_map.insert(entity.clone(), 1);

        // Define the Dirichlet boundary condition
        let dirichlet_bc = BoundaryCondition::Dirichlet(5.0);

        // Apply the Dirichlet boundary condition
        dirichlet_bc.apply(&entity, &mut rhs_mut, &mut matrix_mut, &index_map, 0.0);

        // Verify matrix and RHS updates
        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0); // Only the diagonal element should be 1.0
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0); // All other elements in the row should be 0.0
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 5.0); // Verify the RHS value
    }
}
