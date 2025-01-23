use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `DirichletBC` struct represents a handler for applying Dirichlet boundary conditions 
/// to a set of mesh entities. It stores the conditions in a `DashMap` and applies them to 
/// modify both the system matrix and the right-hand side (rhs).
pub struct DirichletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl DirichletBC {
    /// Creates a new instance of `DirichletBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Dirichlet boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Dirichlet boundary conditions to the system matrix and rhs. 
    /// It iterates over the stored conditions and applies either constant or function-based Dirichlet
    /// boundary conditions to the corresponding entities.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_constant_dirichlet(matrix, rhs, index, *value);
                    }
                    BoundaryCondition::DirichletFn(wrapper) => {
                        let coords = self.get_coordinates(entity);
                        let value = (wrapper.function)(time, &coords);
                        self.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    _ => {}
                }
            }
        });
    }

    /// Applies a constant Dirichlet boundary condition to the matrix and rhs for a specific index.
    pub fn apply_constant_dirichlet(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) {
        println!("Applying Dirichlet at index {} with value {}", index, value);
        println!("Matrix before update: {:?}", matrix);
        println!("RHS before update: {:?}", rhs);
        let ncols = matrix.ncols();
        for col in 0..ncols {
            matrix[(index,col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        rhs[(index, 0)] = value;
        println!("Matrix after update: {:?}", matrix);
        println!("RHS after update: {:?}", rhs);
    }

    /// Retrieves the coordinates of the mesh entity (placeholder for real coordinates).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for DirichletBC {
    /// Applies the stored Dirichlet boundary conditions for a specific mesh entity.
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
    use crate::{boundary::bc_handler::FunctionWrapper, domain::mesh_entity::MeshEntity};
    use std::sync::Arc;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Dirichlet boundary condition
        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(10.0));
        
        // Verify that the condition was set correctly
        let condition = dirichlet_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Dirichlet(10.0))));
    }

    #[test]
    fn test_apply_constant_dirichlet() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(5.0));
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0);
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_dirichlet() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 2);

        let wrapper = FunctionWrapper {
            description: "test_fn".to_string(),
            function: Arc::new(|_time, _coords| 7.0),
        };

        dirichlet_bc.set_bc(entity, BoundaryCondition::DirichletFn(wrapper));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        dirichlet_bc.apply_bc(&mut matrix.as_mut(), &mut rhs.as_mut(), &entity_to_index, 1.0);

        for col in 0..matrix.ncols() {
            if col == 2 {
                assert_eq!(matrix[(2, col)], 1.0);
            } else {
                assert_eq!(matrix[(2, col)], 0.0);
            }
        }
        assert_eq!(rhs[(2, 0)], 7.0);
    }
}
