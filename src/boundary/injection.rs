//! Injection Boundary Conditions
//! Handles injection boundary conditions for CFD simulations.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct InjectionBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl InjectionBC {
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
        todo!("Implement injection boundary conditions logic");
    }
}

impl BoundaryConditionApply for InjectionBC {
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
