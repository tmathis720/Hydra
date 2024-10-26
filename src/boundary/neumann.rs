use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `NeumannBC` struct represents a handler for applying Neumann boundary conditions 
/// to a set of mesh entities. Neumann boundary conditions involve specifying the flux across 
/// a boundary, and they modify only the right-hand side (RHS) of the system without modifying 
/// the system matrix.
pub struct NeumannBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl NeumannBC {
    /// Creates a new instance of `NeumannBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Neumann boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Neumann boundary conditions to the right-hand side (RHS) of the system. 
    /// It iterates over the stored conditions and applies either constant or function-based Neumann
    /// boundary conditions to the corresponding entities.
    pub fn apply_bc(
        &self,
        _matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::Neumann(value) => {
                        self.apply_constant_neumann(rhs, index, *value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = self.get_coordinates(entity);
                        let value = fn_bc(time, &coords);
                        self.apply_constant_neumann(rhs, index, value);
                    }
                    _ => {}
                }
            }
        });
    }

    /// Applies a constant Neumann boundary condition to the right-hand side (RHS) for a specific index.
    pub fn apply_constant_neumann(&self, rhs: &mut MatMut<f64>, index: usize, value: f64) {
        rhs.write(index, 0, rhs.read(index, 0) + value);
    }

    /// Retrieves the coordinates of the mesh entity (currently a placeholder).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for NeumannBC {
    /// Applies the stored Neumann boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        _matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(_matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;
    use std::sync::Arc;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        
        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(10.0));
        
        let condition = neumann_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Neumann(10.0))));
    }

    #[test]
    fn test_apply_constant_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(5.0));
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        neumann_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index, 0.0);

        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 2);

        neumann_bc.set_bc(
            entity,
            BoundaryCondition::NeumannFn(Arc::new(|_time: f64, _coords: &[f64]| 7.0)),
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        neumann_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index, 1.0);

        assert_eq!(rhs_mut[(2, 0)], 7.0);
    }
}
