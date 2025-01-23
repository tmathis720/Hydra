//! Inlet and Outlet Boundary Conditions
//! Includes handling of inflow and outflow conditions.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct InletOutletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl InletOutletBC {
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a boundary condition for a given entity
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Core method to apply boundary conditions to matrix and RHS
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
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_dirichlet(matrix, rhs, *index, *value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        self.apply_neumann(rhs, *index, *flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, *index, *alpha, *beta);
                    }
                    _ => {
                        panic!("Unsupported condition for Inlet/Outlet");
                    }
                }
            }
        }
    }

    /// Applies Dirichlet boundary condition for inlet
    pub fn apply_dirichlet(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, index: usize, value: f64) {
        // Set all entries in the row to zero and set diagonal to 1
        for col in 0..matrix.ncols() {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;
        // Set RHS to the prescribed value
        rhs[(index, 0)] = value;
    }

    /// Applies Neumann boundary condition for outlet
    pub fn apply_neumann(&self, rhs: &mut MatMut<f64>, index: usize, flux: f64) {
        rhs[(index, 0)] += flux;
    }

    /// Applies Robin boundary condition (generalized inflow/outflow)
    pub fn apply_robin(&self, matrix: &mut MatMut<f64>, rhs: &mut MatMut<f64>, index: usize, alpha: f64, beta: f64) {
        matrix[(index, index)] += alpha;
        rhs[(index, 0)] += beta;
    }
}

impl BoundaryConditionApply for InletOutletBC {
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
        inlet_outlet.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

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
        inlet_outlet.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index);

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
        inlet_outlet.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify the updates to matrix and RHS
        assert_eq!(matrix_mut[(0, 0)], 3.0); // Original diagonal value (1.0) + alpha (2.0)
        assert_eq!(rhs_mut[(0, 0)], 4.0);    // Original RHS value (0.0) + beta (4.0)
    }
}
