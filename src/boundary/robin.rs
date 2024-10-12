// src/boundary/robin.rs

use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use crate::domain::section::Section;
use faer::MatMut;

/// The `RobinBC` struct represents a handler for applying Robin boundary conditions 
/// to a set of mesh entities. Robin boundary conditions involve a linear combination 
/// of Dirichlet and Neumann boundary conditions, and they modify both the system matrix 
/// and the right-hand side (RHS).
/// 
/// Example usage:
/// 
///    let robin_bc = RobinBC::new();  
///    let entity = MeshEntity::Vertex(1);  
///    robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });  
///    robin_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, 0.0);  
/// 
pub struct RobinBC {
    conditions: Section<BoundaryCondition>,
}

impl RobinBC {
    /// Creates a new instance of `RobinBC` with an empty section to store boundary conditions.
    /// 
    /// Example usage:
    /// 
    ///    let robin_bc = RobinBC::new();  
    /// 
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    /// Sets a Robin boundary condition for a specific mesh entity.
    ///
    /// # Arguments:
    /// * `entity` - The mesh entity to which the boundary condition will be applied.
    /// * `condition` - The boundary condition to set (Robin condition with alpha and beta).
    ///
    /// Example usage:
    /// 
    ///    let entity = MeshEntity::Vertex(1);  
    ///    robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });  
    /// 
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    /// Applies the stored Robin boundary conditions to both the system matrix and rhs. 
    /// It iterates over the stored conditions and applies the Robin boundary condition 
    /// to the corresponding entities.
    ///
    /// # Arguments:
    /// * `matrix` - The mutable system matrix.
    /// * `rhs` - The mutable right-hand side vector.
    /// * `entity_to_index` - A hash map that associates mesh entities with their indices in the system.
    /// * `_time` - The current time (unused in Robin BC).
    ///
    /// Example usage:
    /// 
    ///    let entity_to_index = FxHashMap::default();  
    ///    robin_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, 0.0);  
    /// 
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &FxHashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        let data = self.conditions.data.read().unwrap();
        for (entity, condition) in data.iter() {
            if let Some(&index) = entity_to_index.get(entity) {
                match condition {
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Applies a Robin boundary condition to the system matrix and rhs for a specific index.
    ///
    /// # Arguments:
    /// * `matrix` - The mutable system matrix.
    /// * `rhs` - The mutable right-hand side vector.
    /// * `index` - The index of the matrix row and rhs corresponding to the mesh entity.
    /// * `alpha` - The coefficient for the Robin condition in the matrix (modifying the diagonal).
    /// * `beta` - The constant value for the Robin condition (added to rhs).
    ///
    /// Example usage:
    /// 
    ///    robin_bc.apply_robin(&mut matrix, &mut rhs, 1, 2.0, 3.0);  
    /// 
    pub fn apply_robin(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        alpha: f64,
        beta: f64,
    ) {
        matrix.write(index, index, matrix.read(index, index) + alpha);
        rhs.write(index, 0, rhs.read(index, 0) + beta);
    }
}

impl BoundaryConditionApply for RobinBC {
    /// Applies the stored Robin boundary conditions for a specific mesh entity.
    ///
    /// This implementation utilizes the general `apply_bc` method to modify the matrix and rhs.
    ///
    /// # Arguments:
    /// * `_entity` - The mesh entity to which the boundary condition applies.
    /// * `rhs` - The mutable right-hand side vector.
    /// * `matrix` - The mutable system matrix.
    /// * `entity_to_index` - A hash map that associates mesh entities with their indices.
    /// * `time` - The current time (unused in Robin BC).
    ///
    /// Example usage:
    /// 
    ///    robin_bc.apply(&entity, &mut rhs, &mut matrix, &entity_to_index, 0.0);  
    /// 
    fn apply(&self, _entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
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
        let robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Robin boundary condition
        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        // Verify that the condition was set correctly
        let condition = robin_bc.conditions.restrict(&entity);
        assert!(matches!(condition, Some(BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 })));
    }

    #[test]
    fn test_apply_robin_bc() {
        let robin_bc = RobinBC::new();
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
