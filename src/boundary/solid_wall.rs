//! Solid Wall Boundary Conditions
//! Handles Inviscid and Viscous solid wall boundary conditions.

use crate::boundary::{bc_handler::{BoundaryCondition, BoundaryConditionApply}, BoundaryError};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use log::{info, warn, error};

/// Struct to manage Solid Wall Boundary Conditions.
pub struct SolidWallBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl SolidWallBC {
    /// Creates a new instance of SolidWallBC.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a solid wall boundary condition (Inviscid or Viscous) for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity.clone(), condition);
        info!("Solid Wall BC set for entity: {:?}", entity);
    }

    /// Applies solid wall boundary conditions (inviscid or viscous) to the system matrix and RHS vector.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
    ) -> Result<(), BoundaryError> {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            match entity_to_index.get(entity) {
                Some(index) => {
                    let index = *index;
                    if index >= matrix.nrows() {
                        let err = BoundaryError::InvalidIndex(format!(
                            "Invalid matrix index {} for entity {:?}. Out of bounds.",
                            index, entity
                        ));
                        error!("{}", err);
                        return Err(err);
                    }

                    match condition {
                        BoundaryCondition::SolidWallInviscid => {
                            info!("Applying Inviscid Solid Wall BC at index {}", index);
                            self.apply_inviscid_wall(matrix, rhs, index);
                        }
                        BoundaryCondition::SolidWallViscous { normal_velocity } => {
                            info!(
                                "Applying Viscous Solid Wall BC at index {} with normal velocity {}",
                                index, normal_velocity
                            );
                            self.apply_viscous_wall(matrix, rhs, index, *normal_velocity);
                        }
                        _ => {
                            warn!(
                                "Unexpected condition for entity {:?} in SolidWallBC: {:?}",
                                entity, condition
                            );
                        }
                    }
                }
                None => {
                    let err = BoundaryError::EntityNotFound(format!(
                        "Entity {:?} not found in index mapping for SolidWallBC",
                        entity
                    ));
                    error!("{}", err);
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    /// Applies inviscid solid wall boundary conditions.
    ///
    /// Inviscid walls enforce the condition that there is no flow normal to the wall.
    fn apply_inviscid_wall(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
    ) {
        let ncols = matrix.ncols();
        for col in 0..ncols {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        rhs[(index, 0)] = 0.0; // Enforce no flux
    }

    /// Applies viscous solid wall boundary conditions.
    ///
    /// Viscous walls enforce no-slip (zero velocity at the wall).
    fn apply_viscous_wall(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        normal_velocity: f64,
    ) {
        let ncols = matrix.ncols();
        for col in 0..ncols {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        rhs[(index, 0)] = normal_velocity; // Enforce no-slip
    }
}

impl BoundaryConditionApply for SolidWallBC {
    /// Applies solid wall boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) -> Result<(), BoundaryError> {
        self.apply_bc(matrix, rhs, entity_to_index)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::bc_handler::BoundaryCondition;
    use dashmap::DashMap;
    use faer::Mat;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_apply_inviscid_wall() {
        let solid_wall_bc = SolidWallBC::new();
        let entity = MeshEntity::Face(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 1);

        solid_wall_bc.set_bc(entity.clone(), BoundaryCondition::SolidWallInviscid);

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let _ = solid_wall_bc.apply_bc(&mut matrix.as_mut(), &mut rhs.as_mut(), &entity_to_index);

        // Verify matrix diagonal for inviscid wall
        for col in 0..matrix.ncols() {
            if col == 1 {
                assert_eq!(matrix[(1, col)], 1.0);
            } else {
                assert_eq!(matrix[(1, col)], 0.0);
            }
        }

        // Verify RHS remains unchanged
        assert_eq!(rhs[(1, 0)], 0.0);
    }

    #[test]
    fn test_apply_viscous_wall() {
        let solid_wall_bc = SolidWallBC::new();
        let entity = MeshEntity::Face(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 2);

        let normal_velocity = -0.5;
        solid_wall_bc.set_bc(
            entity.clone(),
            BoundaryCondition::SolidWallViscous {
                normal_velocity,
            },
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let _ = solid_wall_bc.apply_bc(&mut matrix.as_mut(), &mut rhs.as_mut(), &entity_to_index);

        // Verify matrix diagonal for viscous wall
        for col in 0..matrix.ncols() {
            if col == 2 {
                assert_eq!(matrix[(2, col)], 1.0);
            } else {
                assert_eq!(matrix[(2, col)], 0.0);
            }
        }

        // Verify RHS enforces no-slip condition
        assert_eq!(rhs[(2, 0)], normal_velocity);
    }
}
