use dashmap::DashMap;
use crate::boundary::BoundaryError;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use log::{info, error};

/// The `NeumannBC` struct manages Neumann boundary conditions.
pub struct NeumannBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl NeumannBC {
    /// Creates a new instance of `NeumannBC`.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Neumann boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
        info!("Set Neumann boundary condition for entity {:?}", entity);
    }

    /// Applies all Neumann boundary conditions to the right-hand side (RHS) of the system.
    pub fn apply_bc(
        &self,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError> {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            match entity_to_index.get(entity) {
                Some(index) => match condition {
                    BoundaryCondition::Neumann(value) => {
                        self.apply_constant_neumann(rhs, *index, *value)?;
                    }
                    BoundaryCondition::NeumannFn(wrapper) => {
                        let coords = self.get_coordinates(entity);
                        let computed_value = (wrapper.function)(time, &coords);
                        self.apply_constant_neumann(rhs, *index, computed_value)?;
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Unsupported condition for NeumannBC: {:?}",
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

    /// Applies a constant Neumann boundary condition to the right-hand side (RHS) for a specific index.
    pub fn apply_constant_neumann(&self, rhs: &mut MatMut<f64>, index: usize, value: f64) -> Result<(), BoundaryError> {
        info!(
            "Applying Neumann boundary condition at index {} with flux {}",
            index, value
        );

        if index >= rhs.nrows() {
            let err = BoundaryError::InvalidIndex(format!(
                "Index {} is out of bounds for RHS vector.",
                index
            ));
            error!("{}", err);
            return Err(err);
        }

        rhs[(index, 0)] += value;

        info!(
            "Neumann condition applied: rhs[{},0] += {}",
            index, value
        );

        Ok(())
    }

    /// Retrieves the coordinates of the mesh entity (placeholder for real coordinates).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for NeumannBC {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        _matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) -> Result<(), BoundaryError> {
        if let Some(index) = entity_to_index.get(entity) {
            if let Some(condition) = self.conditions.get(entity) {
                match *condition {
                    BoundaryCondition::Neumann(value) => {
                        return self.apply_constant_neumann(rhs, *index, value);
                    }
                    BoundaryCondition::NeumannFn(ref wrapper) => {
                        let coords = self.get_coordinates(entity);
                        let computed_value = (wrapper.function)(time, &coords);
                        return self.apply_constant_neumann(rhs, *index, computed_value);
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
    use std::sync::Arc;
    use crate::{boundary::bc_handler::FunctionWrapper, domain::mesh_entity::MeshEntity};

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

        // Set a constant Neumann boundary condition
        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(5.0));
        
        let mut rhs = create_test_matrix_and_rhs().1; // Only create the RHS vector
        let mut rhs_mut = rhs.as_mut();

        // Apply the Neumann boundary condition
        let _ = neumann_bc.apply_bc(&mut rhs_mut, &entity_to_index, 0.0);

        // Verify that the RHS was updated correctly
        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 2);

        let wrapper = FunctionWrapper {
            description: "neumann_fn".to_string(),
            function: Arc::new(|_time, _coords| 7.0),
        };

        neumann_bc.set_bc(entity, BoundaryCondition::NeumannFn(wrapper));

        let mut rhs = Mat::zeros(3, 1);
        let _ = neumann_bc.apply_bc(&mut rhs.as_mut(), &entity_to_index, 1.0);

        assert_eq!(rhs[(2, 0)], 7.0);
    }
}
