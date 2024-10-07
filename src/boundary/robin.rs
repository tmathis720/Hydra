// src/boundary/robin.rs

use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use crate::domain::section::Section;
use faer::MatMut;

pub struct RobinBC {
    conditions: Section<BoundaryCondition>,  // Section to hold Robin conditions
}

impl RobinBC {
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    // Set a Robin boundary condition for a specific entity
    pub fn set_bc(&mut self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    // Apply the Robin boundary condition during the system matrix assembly
    pub fn apply_bc(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, _time: f64) {
        for (entity, &offset) in self.conditions.offsets.iter() {
            if let Some(&index) = entity_to_index.get(entity) {
                let condition = &self.conditions.data[offset];  // Access the condition using the offset
                match condition {
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn apply_robin(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, index: usize, alpha: f64, beta: f64) {
        // Robin condition modifies both the matrix and the RHS

        // Modify the diagonal of the matrix (alpha term)
        matrix.write(index, index, matrix.read(index, index) + alpha);

        // Add the beta * flux to the RHS (beta term)
        rhs.write(index, 0, rhs.read(index, 0) + beta);
    }
}

impl BoundaryConditionApply for RobinBC {
    fn apply(&self, _entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        // Robin-specific logic
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        // Create a 3x3 test matrix initialized to identity matrix
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        
        // Create a 3x1 test RHS vector initialized to zero
        let rhs = Mat::zeros(3, 1);
        
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let mut robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Robin boundary condition
        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        // Verify that the condition was set correctly
        let condition = robin_bc.conditions.restrict(&entity);
        assert!(matches!(condition, Some(BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 })));
    }

    #[test]
    fn test_apply_robin_bc() {
        let mut robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 1);

        // Set a Robin boundary condition
        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        // Create a test matrix and RHS vector
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the Robin condition
        robin_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Check that the matrix diagonal has been updated (alpha term)
        assert_eq!(matrix_mut[(1, 1)], 3.0);  // Initial value 1.0 + alpha 2.0

        // Check that the RHS has been updated (beta term)
        assert_eq!(rhs_mut[(1, 0)], 3.0);  // Beta term applied
    }
}
