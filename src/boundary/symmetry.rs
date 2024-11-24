//! Symmetry Plane Boundary Conditions
//! Implements symmetry plane constraints.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct SymmetryBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl SymmetryBC {
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
    ) {
        todo!("Implement symmetry plane logic");
    }
}

impl BoundaryConditionApply for SymmetryBC {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index);
    }
}
