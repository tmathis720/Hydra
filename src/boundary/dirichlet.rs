use dashmap::DashMap;
use crate::boundary::BoundaryError;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use log::{info, error};

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
        info!("Set Dirichlet boundary condition for entity {:?}", entity);
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
    ) -> Result<(), BoundaryError> {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            match entity_to_index.get(entity) {
                Some(index) => match condition {
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_constant_dirichlet(matrix, rhs, *index, *value)?;
                    }
                    BoundaryCondition::DirichletFn(wrapper) => {
                        let coords = self.get_coordinates(entity);
                        let value = (wrapper.function)(time, &coords);
                        self.apply_constant_dirichlet(matrix, rhs, *index, value)?;
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Invalid boundary condition type for entity {:?}",
                            entity
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

    /// Applies a constant Dirichlet boundary condition to the matrix and rhs for a specific index.
    pub fn apply_constant_dirichlet(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) -> Result<(), BoundaryError> {
        info!("Applying Dirichlet at index {} with value {}", index, value);

        let ncols = matrix.ncols();
        if index >= ncols {
            let err = BoundaryError::InvalidIndex(format!(
                "Index {} is out of bounds for matrix with {} columns.",
                index, ncols
            ));
            error!("{}", err);
            return Err(err);
        }

        for col in 0..ncols {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        rhs[(index, 0)] = value;

        info!(
            "Dirichlet condition applied: matrix[{},{}] = 1, rhs[{},0] = {}",
            index, index, index, value
        );

        Ok(())
    }

    /// Retrieves the coordinates of the mesh entity (placeholder for real coordinates).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0] // Placeholder; should integrate with a coordinate system.
    }
}

impl BoundaryConditionApply for DirichletBC {
    /// Applies the stored Dirichlet boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError> {
        if let Some(index) = entity_to_index.get(entity) {
            if let Some(condition) = self.conditions.get(entity) {
                match *condition {
                    BoundaryCondition::Dirichlet(value) => {
                        return self.apply_constant_dirichlet(matrix, rhs, *index, value);
                    }
                    BoundaryCondition::DirichletFn(ref wrapper) => {
                        let coords = self.get_coordinates(entity);
                        let value = (wrapper.function)(time, &coords);
                        return self.apply_constant_dirichlet(matrix, rhs, *index, value);
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
                "Dirichlet boundary condition not found for entity {:?}",
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

        let _ = dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

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
        let _ = dirichlet_bc.apply_bc(&mut matrix.as_mut(), &mut rhs.as_mut(), &entity_to_index, 1.0);

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
