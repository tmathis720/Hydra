//! Symmetry Plane Boundary Conditions
//! Implements symmetry plane constraints.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct SymmetryBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl SymmetryBC {
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a symmetry boundary condition for a specific entity
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies symmetry boundary conditions to the matrix and RHS
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
    ) {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity) {
                match condition {
                    BoundaryCondition::SolidWallInviscid => {
                        self.apply_symmetry_plane(matrix, rhs, *index);
                    }
                    _ => {
                        panic!("Unsupported condition for Symmetry Plane");
                    }
                }
            }
        }
    }

    /// Applies symmetry condition by zeroing normal velocity components and flux
    pub fn apply_symmetry_plane(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
    ) {
        // Zero all entries in the matrix row related to normal velocity
        for col in 0..matrix.ncols() {
            matrix.write(index, col, 0.0);
        }

        // Set the diagonal to 1 to enforce the symmetry constraint
        matrix.write(index, index, 1.0);

        // Ensure RHS value is zero for symmetry
        rhs.write(index, 0, 0.0);
    }
}

impl BoundaryConditionApply for SymmetryBC {
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index);
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
    fn test_apply_symmetry_plane() {
        let symmetry_bc = SymmetryBC::new();
        let entity = MeshEntity::Face(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 1);

        // Set a symmetry boundary condition
        symmetry_bc.set_bc(entity.clone(), BoundaryCondition::SolidWallInviscid);

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply the boundary condition
        symmetry_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify the matrix row and RHS
        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0); // Diagonal element should be 1.0
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0); // Other elements should be 0.0
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 0.0); // RHS value should be zero
    }
}
