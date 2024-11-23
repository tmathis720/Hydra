//! Far-Field Boundary Conditions
//! Implements conditions for far-field boundaries, typically used to simulate the infinite extent of a domain.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

/// Represents far-field boundary conditions.
pub struct FarFieldBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl FarFieldBC {
    /// Creates a new instance of FarFieldBC.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Assigns a far-field boundary condition to a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the far-field boundary conditions to the system matrix and RHS vector.
    ///
    /// # Parameters
    /// - `matrix`: The system matrix to be modified.
    /// - `rhs`: The right-hand side vector to be adjusted.
    /// - `entity_to_index`: Mapping from mesh entities to matrix indices.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::FarField(value) => {
                        // Example: Implement logic for far-field conditions.
                        matrix.write(index, index, matrix.read(index, index) + value);
                        rhs.write(index, 0, rhs.read(index, 0) + value);
                    }
                    _ => todo!("Handle other far-field variations if needed."),
                }
            }
        }
    }
}

impl BoundaryConditionApply for FarFieldBC {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}
