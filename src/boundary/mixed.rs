use dashmap::DashMap;
use crate::boundary::BoundaryError;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use log::{info, error};

/// The `MixedBC` struct manages Mixed boundary conditions.
pub struct MixedBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl MixedBC {
    /// Creates a new instance of `MixedBC`.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Mixed boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
        info!("Set mixed boundary condition for entity {:?}", entity);
    }

    /// Applies all Mixed boundary conditions within the handler to the system.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
    ) -> Result<(), BoundaryError> {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            match entity_to_index.get(entity) {
                Some(index) => match condition {
                    BoundaryCondition::Mixed { gamma, delta } => {
                        self.apply_mixed(matrix, rhs, *index, *gamma, *delta)?;
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Unsupported condition for MixedBC: {:?}",
                            condition
                        ));
                        error!("{}", err);
                        return Err(err);
                    }
                },
                None => {
                    let err = BoundaryError::EntityNotFound(format!(
                        "Entity {:?} not found in entity_to_index mapping.",
                        entity
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    /// Applies a Mixed boundary condition to the system matrix and RHS vector.
    pub fn apply_mixed(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        gamma: f64,
        delta: f64,
    ) -> Result<(), BoundaryError> {
        info!(
            "Applying Mixed boundary condition at index {} with gamma {}, delta {}",
            index, gamma, delta
        );

        if index >= matrix.nrows() || index >= rhs.nrows() {
            let err = BoundaryError::InvalidIndex(format!(
                "Index {} is out of bounds for matrix/rhs dimensions.",
                index
            ));
            error!("{}", err);
            return Err(err);
        }

        matrix[(index, index)] += gamma;
        rhs[(index, 0)] += delta;

        info!(
            "Mixed condition applied: matrix[{},{}] += {}, rhs[{},0] += {}",
            index, index, gamma, index, delta
        );

        Ok(())
    }
}

impl BoundaryConditionApply for MixedBC {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) -> Result<(), BoundaryError> {
        if let Some(index) = entity_to_index.get(entity) {
            if let Some(condition) = self.conditions.get(entity) {
                match *condition {
                    BoundaryCondition::Mixed { gamma, delta } => {
                        return self.apply_mixed(matrix, rhs, *index, gamma, delta);
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Invalid boundary condition type for entity {:?}",
                            entity
                        ));
                        error!("{}", err);
                        return Err(err);
                    }
                }
            }
            let err = BoundaryError::EntityNotFound(format!(
                "Boundary condition not found for entity {:?}",
                entity
            ));
            error!("{}", err);
            Err(err)
        } else {
            let err = BoundaryError::EntityNotFound(format!(
                "Entity {:?} not found in entity_to_index mapping.",
                entity
            ));
            error!("{}", err);
            Err(err)
        }
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
        let mixed_bc = MixedBC::new();
        let entity = MeshEntity::Vertex(1);

        mixed_bc.set_bc(entity, BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 });

        let condition = mixed_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 })));
    }

    #[test]
    fn test_apply_mixed_bc() {
        let mixed_bc = MixedBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        mixed_bc.set_bc(entity, BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 });

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        let _ = mixed_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Assert the matrix diagonal at index 1 has been incremented by gamma
        assert_eq!(matrix_mut[(1, 1)], 3.0);  // 1.0 initial + 2.0 gamma
        // Assert the RHS value at index 1 has been incremented by delta
        assert_eq!(rhs_mut[(1, 0)], 3.0);
    }
}
