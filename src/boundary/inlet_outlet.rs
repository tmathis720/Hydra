//! Inlet and Outlet Boundary Conditions
//! Includes handling of inflow and outflow conditions.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use log::{info, error};

use super::BoundaryError;

/// The `InletOutletBC` struct manages inflow and outflow boundary conditions.
pub struct InletOutletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl InletOutletBC {
    /// Creates a new instance of `InletOutletBC`.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a boundary condition for a given entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
        info!("Set inlet/outlet boundary condition for entity {:?}", entity);
    }

    /// Core method to apply boundary conditions to the matrix and RHS.
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
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_dirichlet(matrix, rhs, *index, *value)?;
                    }
                    BoundaryCondition::Neumann(flux) => {
                        self.apply_neumann(rhs, *index, *flux)?;
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, *index, *alpha, *beta)?;
                    }
                    _ => {
                        let err = BoundaryError::InvalidBoundaryType(format!(
                            "Unsupported condition for InletOutletBC: {:?}",
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

    /// Applies Dirichlet boundary condition for inlet.
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

        // Zero all entries in the row and set the diagonal to 1
        for col in 0..ncols {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        rhs[(index, 0)] = value;

        info!("Dirichlet condition applied: matrix[{},{}] = 1, rhs[{},0] = {}", index, index, index, value);

        Ok(())
    }

    /// Applies Neumann boundary condition for outlet.
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

        rhs[(index, 0)] += flux;

        info!("Neumann condition applied: rhs[{},0] = {}", index, rhs[(index, 0)]);

        Ok(())
    }

    /// Applies Robin boundary condition (generalized inflow/outflow).
    fn apply_robin(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<(), BoundaryError> {
        info!(
            "Applying Robin condition at index {} with alpha {}, beta {}",
            index, alpha, beta
        );

        if index >= matrix.nrows() || index >= rhs.nrows() {
            let err = BoundaryError::InvalidIndex(format!(
                "Index {} is out of bounds for matrix/rhs dimensions.",
                index
            ));
            error!("{}", err);
            return Err(err);
        }

        matrix[(index, index)] += alpha;
        rhs[(index, 0)] += beta;

        info!(
            "Robin condition applied: matrix[{},{}] += {}, rhs[{},0] += {}",
            index, index, alpha, index, beta
        );

        Ok(())
    }
}

impl BoundaryConditionApply for InletOutletBC {
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
                    BoundaryCondition::Dirichlet(value) => {
                        return self.apply_dirichlet(matrix, rhs, *index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        return self.apply_neumann(rhs, *index, flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        return self.apply_robin(matrix, rhs, *index, alpha, beta);
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
    use dashmap::DashMap;
    use crate::boundary::bc_handler::BoundaryCondition;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_and_apply_dirichlet_bc() {
        let inlet_outlet = InletOutletBC::new();
        let entity = MeshEntity::Face(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 1);

        // Set a Dirichlet boundary condition
        inlet_outlet.set_bc(entity.clone(), BoundaryCondition::Dirichlet(5.0));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the boundary condition
        let _ = inlet_outlet.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify the matrix and RHS updates
        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0); // Diagonal element should be 1.0
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0); // Other elements should be 0.0
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 5.0); // RHS value should match the Dirichlet condition
    }

    #[test]
    fn test_set_and_apply_neumann_bc() {
        let inlet_outlet = InletOutletBC::new();
        let entity = MeshEntity::Face(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 2);

        // Set a Neumann boundary condition
        inlet_outlet.set_bc(entity.clone(), BoundaryCondition::Neumann(3.0));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        // Apply the boundary condition
        let _ = inlet_outlet.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index);

        // Verify that RHS was updated correctly
        assert_eq!(rhs_mut[(2, 0)], 3.0);
    }

    #[test]
    fn test_set_and_apply_robin_bc() {
        let inlet_outlet = InletOutletBC::new();
        let entity = MeshEntity::Face(0);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 0);

        // Set a Robin boundary condition
        inlet_outlet.set_bc(
            entity.clone(),
            BoundaryCondition::Robin { alpha: 2.0, beta: 4.0 },
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the boundary condition
        let _ = inlet_outlet.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify the updates to matrix and RHS
        assert_eq!(matrix_mut[(0, 0)], 3.0); // Original diagonal value (1.0) + alpha (2.0)
        assert_eq!(rhs_mut[(0, 0)], 4.0);    // Original RHS value (0.0) + beta (4.0)
    }
}
