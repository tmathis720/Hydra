use dashmap::DashMap;
use crate::boundary::BoundaryError;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use log::{info, warn, error};

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
        self.conditions.insert(entity.clone(), condition);
        info!("Robin BC set for entity: {:?}", entity);
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
    ) -> Result<(), BoundaryError> {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            match entity_to_index.get(entity) {
                Some(index) => {
                    let index = *index;
                    if index >= matrix.nrows() {
                        let err = BoundaryError::InvalidIndex(format!(
                            "Invalid matrix index {} for entity {:?}. Out of bounds.",
                            index, entity
                        ));
                        error!("{}", err);
                        return Err(err);
                    }

                    if let BoundaryCondition::Robin { alpha, beta } = condition {
                        info!(
                            "Applying Robin BC at index {}: alpha={}, beta={}",
                            index, alpha, beta
                        );
                        self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    } else {
                        warn!(
                            "Unexpected condition for entity {:?} in RobinBC: {:?}",
                            entity, condition
                        );
                    }
                }
                None => {
                    let err = BoundaryError::EntityNotFound(format!(
                        "Entity {:?} not found in index mapping for RobinBC",
                        entity
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }
        }
        Ok(())
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
        matrix[(index, index)] += alpha;
        rhs[(index, 0)] += beta;
    }
}

impl BoundaryConditionApply for RobinBC {
    /// Applies the stored Robin boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) -> Result<(), BoundaryError> {
        if let Some(index) = entity_to_index.get(entity) {
            let index = *index;
            if index >= matrix.nrows() {
                let err = BoundaryError::InvalidIndex(format!(
                    "Invalid matrix index {} for entity {:?}. Out of bounds.",
                    index, entity
                ));
                error!("{}", err);
                return Err(err);
            }

            if let Some(condition) = self.conditions.get(entity) {
                if let BoundaryCondition::Robin { alpha, beta } = condition.value() {
                    info!(
                        "Applying Robin BC at index {}: alpha={}, beta={}",
                        index, alpha, beta
                    );
                    self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    return Ok(());
                }
            }

            let err = BoundaryError::EntityNotFound(format!(
                "Robin boundary condition not found for entity {:?}",
                entity
            ));
            error!("{}", err);
            return Err(err);
        }

        let err = BoundaryError::EntityNotFound(format!(
            "Entity {:?} not found in index mapping for RobinBC",
            entity
        ));
        error!("{}", err);
        Err(err)
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

        let _ = robin_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        assert_eq!(matrix_mut[(1, 1)], 3.0);  // Initial value 1.0 + alpha 2.0
        assert_eq!(rhs_mut[(1, 0)], 3.0);    // Beta term applied
    }
}
