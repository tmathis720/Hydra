//! Far-Field Boundary Conditions
//! Implements conditions for far-field boundaries, typically used to simulate the infinite extent of a domain.

use crate::boundary::{bc_handler::{BoundaryCondition, BoundaryConditionApply}, BoundaryError};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use log::{info, error};

/// Represents far-field boundary conditions.
pub struct FarFieldBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl FarFieldBC {
    /// Creates a new instance of FarFieldBC.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Assigns a far-field boundary condition to a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
        info!("Set Far-Field boundary condition for entity {:?}", entity);
    }

    /// Applies the far-field boundary conditions to the system matrix and RHS vector.
    ///
    /// # Parameters
    /// - `matrix`: The system matrix to be modified.
    /// - `rhs`: The right-hand side vector to be adjusted.
    /// - `entity_to_index`: Mapping from mesh entities to matrix indices.
    /// - `time`: The current simulation time, for time-dependent boundary conditions.
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
                Some(index) => match condition {
                    BoundaryCondition::FarField(value) => {
                        self.apply_far_field(matrix, rhs, *index, *value)?;
                    }
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_dirichlet(matrix, rhs, *index, *value)?;
                    }
                    BoundaryCondition::Neumann(flux) => {
                        self.apply_neumann(rhs, *index, *flux)?;
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Unsupported condition for FarFieldBC: {:?}",
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

    /// Applies far-field conditions (e.g., constant value or known state).
    pub fn apply_far_field(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) -> Result<(), BoundaryError> {
        info!("Applying Far-Field condition at index {} with value {}", index, value);

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
        matrix[(index, index)] = 1.0; // Diagonal to enforce the condition
        rhs[(index, 0)] = value; // Set the far-field value

        info!(
            "Far-Field condition applied: matrix[{},{}] = 1, rhs[{},0] = {}",
            index, index, index, value
        );

        Ok(())
    }

    /// Applies a Dirichlet condition to enforce a fixed value.
    fn apply_dirichlet(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) -> Result<(), BoundaryError> {
        info!("Applying Dirichlet condition at index {} with value {}", index, value);

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

    /// Applies a Neumann condition to enforce a flux at the boundary.
    fn apply_neumann(
        &self,
        rhs: &mut MatMut<f64>,
        index: usize,
        flux: f64,
    ) -> Result<(), BoundaryError> {
        info!("Applying Neumann condition at index {} with flux {}", index, flux);

        if index >= rhs.nrows() {
            let err = BoundaryError::InvalidIndex(format!(
                "Index {} is out of bounds for rhs with {} rows.",
                index, rhs.nrows()
            ));
            error!("{}", err);
            return Err(err);
        }

        let current_value = rhs[(index, 0)];
        rhs[(index, 0)] = current_value + flux;

        info!("Neumann condition applied: rhs[{},0] = {}", index, rhs[(index, 0)]);

        Ok(())
    }
}

impl BoundaryConditionApply for FarFieldBC {
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
                    BoundaryCondition::FarField(value) => {
                        return self.apply_far_field(matrix, rhs, *index, value);
                    }
                    BoundaryCondition::Dirichlet(value) => {
                        return self.apply_dirichlet(matrix, rhs, *index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        return self.apply_neumann(rhs, *index, flux);
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
                "Far-Field boundary condition not found for entity {:?}",
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
    use dashmap::DashMap;
    use crate::boundary::bc_handler::BoundaryCondition;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_far_field() {
        let far_field_bc = FarFieldBC::new();
        let entity = MeshEntity::Face(0);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 0);

        // Set far-field condition
        far_field_bc.set_bc(entity.clone(), BoundaryCondition::FarField(100.0));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the boundary condition
        let _ = far_field_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Verify the matrix and RHS updates
        for col in 0..matrix_mut.ncols() {
            if col == 0 {
                assert_eq!(matrix_mut[(0, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(0, col)], 0.0);
            }
        }
        assert_eq!(rhs_mut[(0, 0)], 100.0);
    }
}
