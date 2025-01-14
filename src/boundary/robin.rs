use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `RobinBC` struct represents a handler for applying Robin boundary conditions 
/// to a set of mesh entities. Robin boundary conditions involve a linear combination 
/// of Dirichlet and Neumann boundary conditions, and they modify both the system matrix 
/// and the right-hand side (RHS).
pub struct RobinBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl RobinBC {
    /// Creates a new instance of `RobinBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Robin boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Robin boundary conditions to both the system matrix and rhs. 
    /// It iterates over the stored conditions and applies the Robin boundary condition 
    /// to the corresponding entities.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    }
                    _ => {}
                }
            }
        });
    }

    /// Applies a Robin boundary condition to the system matrix and rhs for a specific index.
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
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        
        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        let condition = robin_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 })));
    }

    #[test]
    fn test_apply_robin_bc() {
        let robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        robin_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        assert_eq!(matrix_mut[(1, 1)], 3.0);  // Initial value 1.0 + alpha 2.0
        assert_eq!(rhs_mut[(1, 0)], 3.0);    // Beta term applied
    }
}
