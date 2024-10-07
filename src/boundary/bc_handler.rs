// src/boundary/bc_handler.rs

use rustc_hash::FxHashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::section::Section;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;  // Import NeumannBC
use crate::boundary::robin::RobinBC;
use faer::MatMut;

pub type BoundaryConditionFn = Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    DirichletFn(BoundaryConditionFn), // Dirichlet with a function
    NeumannFn(BoundaryConditionFn),   // Neumann with a function
}

pub struct BoundaryConditionHandler {
    conditions: Section<BoundaryCondition>,
}

impl BoundaryConditionHandler {
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    pub fn set_bc(&mut self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    pub fn get_bc(&self, entity: &MeshEntity) -> Option<&BoundaryCondition> {
        self.conditions.restrict(entity)
    }

    pub fn apply_bc(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, boundary_entities: &[MeshEntity], entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        // Apply Dirichlet boundary condition
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), *value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        // Apply Neumann boundary condition
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), *flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(matrix, rhs, *entity_to_index.get(entity).unwrap(), *alpha, *beta);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];  // Placeholder: entity coordinates
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];  // Placeholder: entity coordinates
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), value);
                    }
                }
            }
        }
    }
}

pub trait BoundaryConditionApply {
    fn apply(&self, entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64);
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(&self, entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), *flux);
            }
            BoundaryCondition::Robin { alpha: _, beta: _ } => {
                // Robin-specific logic (to be implemented)
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];  // Placeholder: entity coordinates
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, *entity_to_index.get(entity).unwrap(), value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];  // Placeholder: entity coordinates
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, *entity_to_index.get(entity).unwrap(), value);
            }
        }
    }
}
