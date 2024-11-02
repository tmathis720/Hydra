use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `MixedBC` struct represents a handler for applying Mixed boundary conditions
/// to a set of mesh entities. Mixed boundary conditions typically involve a combination
/// of Dirichlet and Neumann-type parameters that modify both the system matrix 
/// and the right-hand side (RHS) vector.
pub struct MixedBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl MixedBC {
    /// Creates a new instance of `MixedBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Mixed boundary condition for a specific mesh entity.
    /// 
    /// # Parameters
    /// - `entity`: The mesh entity to which the boundary condition applies.
    /// - `condition`: The boundary condition to apply, specified as a `BoundaryCondition`.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Mixed boundary conditions to the system matrix and RHS vector.
    /// This method iterates over all stored conditions, updating both the matrix and RHS
    /// based on the specified gamma and delta values for each entity.
    /// 
    /// # Parameters
    /// - `matrix`: The system matrix to be modified by the boundary condition.
    /// - `rhs`: The right-hand side vector to be modified by the boundary condition.
    /// - `index`: The index within the matrix and RHS corresponding to the mesh entity.
    /// - `gamma`: Coefficient applied to the system matrix for this boundary.
    /// - `delta`: Value added to the RHS vector for this boundary.
    pub fn apply_mixed(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        gamma: f64,
        delta: f64,
    ) {
        // Apply the gamma factor to the matrix at the diagonal index
        matrix.write(index, index, matrix.read(index, index) + gamma);
        // Modify the RHS with the delta value at the specific index
        rhs.write(index, 0, rhs.read(index, 0) + delta);
    }

    /// Applies all Mixed boundary conditions within the handler to the system.
    /// It fetches the index of each entity and applies the corresponding mixed boundary 
    /// condition values (gamma and delta) to the matrix and RHS.
    /// 
    /// # Parameters
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `entity_to_index`: Mapping from `MeshEntity` to matrix indices.
    /// - `time`: Current time, if time-dependent boundary values are desired.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                if let BoundaryCondition::Mixed { gamma, delta } = condition {
                    self.apply_mixed(matrix, rhs, index, *gamma, *delta);
                }
            }
        }
    }
}

impl BoundaryConditionApply for MixedBC {
    /// Applies the stored Mixed boundary conditions for a specific mesh entity.
    /// 
    /// # Parameters
    /// - `entity`: Reference to the mesh entity for which the boundary condition is applied.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `entity_to_index`: Reference to the mapping of entities to matrix indices.
    /// - `time`: Current time, allowing for time-dependent boundary conditions.
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

        mixed_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Assert the matrix diagonal at index 1 has been incremented by gamma
        assert_eq!(matrix_mut[(1, 1)], 3.0);  // 1.0 initial + 2.0 gamma
        // Assert the RHS value at index 1 has been incremented by delta
        assert_eq!(rhs_mut[(1, 0)], 3.0);
    }
}
