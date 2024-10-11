// src/boundary/dirichlet.rs

use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use crate::domain::section::Section;
use faer::MatMut;

pub struct DirichletBC {
    conditions: Section<BoundaryCondition>,
}

impl DirichletBC {
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &FxHashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let data = self.conditions.data.read().unwrap();
        for (entity, condition) in data.iter() {
            if let Some(&index) = entity_to_index.get(entity) {
                match condition {
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_constant_dirichlet(matrix, rhs, index, *value);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = self.get_coordinates(entity);
                        let value = fn_bc(time, &coords);
                        self.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn apply_constant_dirichlet(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) {
        let ncols = matrix.ncols();
        for col in 0..ncols {
            matrix.write(index, col, 0.0);
        }
        matrix.write(index, index, 1.0);
        rhs.write(index, 0, value);
    }

    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}


impl BoundaryConditionApply for DirichletBC {
    fn apply(&self, _entity: &MeshEntity, rhs: &mut MatMut<f64>, matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        // Dirichlet-specific logic
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
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
        let mut dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Dirichlet boundary condition
        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(10.0));
        
        // Verify that the condition was set correctly
        let condition = dirichlet_bc.conditions.restrict(&entity);
        assert!(matches!(condition, Some(BoundaryCondition::Dirichlet(10.0))));
    }

    #[test]
    fn test_apply_constant_dirichlet() {
        let mut dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 1);

        // Set a Dirichlet boundary condition
        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(5.0));
        
        // Create a test matrix and RHS vector
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the Dirichlet condition
        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Check that the row in the matrix corresponding to the entity index is zeroed out
        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0); // Diagonal should be 1
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0); // Off-diagonal should be 0
            }
        }

        // Check that the RHS has the correct value for the Dirichlet condition
        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_dirichlet() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(2);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 2);

        dirichlet_bc.set_bc(
            entity,
            BoundaryCondition::DirichletFn(Arc::new(|_time: f64, _coords: &[f64]| 7.0)),
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 1.0);

        for col in 0..matrix_mut.ncols() {
            if col == 2 {
                assert_eq!(matrix_mut[(2, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(2, col)], 0.0);
            }
        }

        assert_eq!(rhs_mut[(2, 0)], 7.0);
    }
}
