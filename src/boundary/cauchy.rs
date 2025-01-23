use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `CauchyBC` struct represents a handler for applying Cauchy boundary conditions
/// to a set of mesh entities. Cauchy boundary conditions typically involve both the
/// value of a state variable and its derivative, defined by parameters `lambda` and `mu`.
/// These conditions influence both the system matrix and the right-hand side vector.
pub struct CauchyBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl CauchyBC {
    /// Creates a new instance of `CauchyBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Cauchy boundary condition for a specific mesh entity.
    ///
    /// # Parameters
    /// - `entity`: The mesh entity to which the boundary condition applies.
    /// - `condition`: The boundary condition to apply, specified as a `BoundaryCondition`.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Cauchy boundary conditions to the system matrix and RHS vector.
    /// This method iterates over all stored conditions, updating both the matrix and RHS
    /// based on the specified lambda and mu values for each entity.
    ///
    /// # Parameters
    /// - `matrix`: The system matrix to be modified by the boundary condition.
    /// - `rhs`: The right-hand side vector to be modified by the boundary condition.
    /// - `index`: The index within the matrix and RHS corresponding to the mesh entity.
    /// - `lambda`: Coefficient for the matrix modification.
    /// - `mu`: Value to adjust the RHS vector.
    pub fn apply_cauchy(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        lambda: f64,
        mu: f64,
    ) {
        // Apply the lambda factor to the matrix at the diagonal index
        matrix[(index, index)] = matrix[(index, index)] + lambda;
        // Modify the RHS with the mu value at the specific index
        rhs[(index, 0)] = rhs[(index, 0)] + mu;
    }

    /// Applies all Cauchy boundary conditions within the handler to the system.
    /// It fetches the index of each entity and applies the corresponding Cauchy boundary 
    /// condition values (lambda and mu) to the matrix and RHS.
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
                if let BoundaryCondition::Cauchy { lambda, mu } = condition {
                    self.apply_cauchy(matrix, rhs, index, *lambda, *mu);
                }
            }
        }
    }
}

impl BoundaryConditionApply for CauchyBC {
    /// Applies the stored Cauchy boundary conditions for a specific mesh entity.
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
        let cauchy_bc = CauchyBC::new();
        let entity = MeshEntity::Vertex(1);

        cauchy_bc.set_bc(entity, BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 });

        let condition = cauchy_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 })));
    }

    #[test]
    fn test_apply_cauchy_bc() {
        let cauchy_bc = CauchyBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        cauchy_bc.set_bc(entity, BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 });

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        cauchy_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Assert the matrix diagonal at index 1 has been incremented by lambda
        assert_eq!(matrix_mut[(1, 1)], 2.5);  // Initial value 1.0 + lambda 1.5
        // Assert the RHS value at index 1 has been incremented by mu
        assert_eq!(rhs_mut[(1, 0)], 2.5);
    }
}
