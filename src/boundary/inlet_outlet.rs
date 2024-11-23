//! Inlet and Outlet Boundary Conditions
//! Includes handling of inflow and outflow conditions.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct InletOutletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl InletOutletBC {
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
        todo!("Implement inlet and outlet boundary conditions logic");
    }
}

impl BoundaryConditionApply for InletOutletBC {
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
