// src/boundary/dirichlet.rs

use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use crate::domain::section::Section;
use faer::MatMut;

pub struct DirichletBC {
    conditions: Section<BoundaryCondition>,  // Section to hold Dirichlet conditions
}

impl DirichletBC {
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    // Set a Dirichlet boundary condition for a specific entity
    // Modify set_bc to accept BoundaryCondition instead of just f64
    pub fn set_bc(&mut self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    // Apply the Dirichlet boundary condition during the system matrix assembly
    pub fn apply_bc(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        for (entity, &offset) in self.conditions.offsets.iter() {
            if let Some(&index) = entity_to_index.get(entity) {
                let condition = &self.conditions.data[offset];  // Access the condition using the offset
                match condition {  // Dereference the condition
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_constant_dirichlet(matrix, rhs, index, *value);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = self.get_coordinates(entity);  // Placeholder for actual method
                        let value = fn_bc(time, &coords);
                        self.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn apply_constant_dirichlet(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, index: usize, value: f64) {
        let ncols = matrix.ncols();
        
        // Zero out the entire row for the Dirichlet condition
        for col in 0..ncols {
            matrix.write(index, col, 0.0);  // Set each element in the row to 0
        }
        
        // Set the diagonal element to 1 to maintain the boundary condition in the system
        matrix.write(index, index, 1.0);
        
        // Set the corresponding value in the RHS vector
        rhs.write(index, 0, value);
    }

    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        // Placeholder: Implement logic to retrieve the coordinates of the mesh entity.
        // This method would retrieve the spatial coordinates associated with a vertex or entity.
        [0.0, 0.0, 0.0]  // Example placeholder return value
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

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        // Create a 3x3 test matrix initialized to identity matrix
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        
        // Create a 3x1 test RHS vector initialized to zero
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
        let mut dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(2);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 2);

        // Set a function-based Dirichlet boundary condition
        dirichlet_bc.set_bc(entity, BoundaryCondition::DirichletFn(Box::new(|_time: f64, _coords: &[f64]| 7.0)));

        // Create a test matrix and RHS vector
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the function-based Dirichlet condition
        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 1.0);

        // Check that the row in the matrix corresponding to the entity index is zeroed out
        for col in 0..matrix_mut.ncols() {
            if col == 2 {
                assert_eq!(matrix_mut[(2, col)], 1.0); // Diagonal should be 1
            } else {
                assert_eq!(matrix_mut[(2, col)], 0.0); // Off-diagonal should be 0
            }
        }

        // Check that the RHS has the correct value based on the function
        assert_eq!(rhs_mut[(2, 0)], 7.0);
    }
}
