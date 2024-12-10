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
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use crate::boundary::solid_wall::SolidWallBC;
use faer::MatMut;

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
/// - `DirichletFn(FunctionWrapper)`: Functional Dirichlet condition with metadata.
/// - `NeumannFn(FunctionWrapper)`: Functional Neumann condition with metadata.
/// - `Periodic { pairs }`: Specifies a Periodic boundary condition between pairs.
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
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    pub fn global() -> Arc<RwLock<BoundaryConditionHandler>> {
        GLOBAL_BC_HANDLER.clone()
    }

    /// Sets a boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
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
                    BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                        let solid_wall_bc = SolidWallBC::new();
                        solid_wall_bc.apply_bc(matrix, rhs, entity_to_index);
                    }
                    BoundaryCondition::DirichletFn(wrapper) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = (wrapper.function)(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(wrapper) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = (wrapper.function)(time, &coords);
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
            BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                let solid_wall_bc = SolidWallBC::new();
                solid_wall_bc.apply_bc(matrix, rhs, entity_to_index);
            }
            BoundaryCondition::DirichletFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0];
                let value = (wrapper.function)(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0];
                let value = (wrapper.function)(time, &coords);
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
