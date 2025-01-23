//! Injection Boundary Conditions
//! Handles injection boundary conditions for CFD simulations.

use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

pub struct InjectionBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl InjectionBC {
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets an injection boundary condition for a specific entity
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies injection boundary conditions to the matrix and RHS
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
                        self.apply_injection(matrix, rhs, *index, *value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        self.apply_neumann_injection(rhs, *index, *flux);
                    }
                    _ => {
                        panic!("Unsupported condition for Injection Boundary");
                    }
                }
            }
        }
    }

    /// Applies Dirichlet-type injection by enforcing a fixed value at the boundary
    pub fn apply_injection(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) {
        // Zero all entries in the row and set the diagonal to 1
        for col in 0..matrix.ncols() {
            matrix[(index, col)] = 0.0;
        }
        matrix[(index, index)] = 1.0;

        // Set the RHS to the specified injection value
        rhs[(index, 0)] = value;
    }

    /// Applies Neumann-type injection by adding flux to the RHS
    pub fn apply_neumann_injection(&self, rhs: &mut MatMut<f64>, index: usize, flux: f64) {
        rhs[(index, 0)] += flux;
    }
}

impl BoundaryConditionApply for InjectionBC {
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
    fn test_dirichlet_injection() {
        let injection_bc = InjectionBC::new();
        let entity = MeshEntity::Face(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 1);

        // Set Dirichlet injection
        injection_bc.set_bc(entity.clone(), BoundaryCondition::Dirichlet(10.0));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        // Apply injection
        injection_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify matrix and RHS updates
        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0);
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 10.0);
    }

    #[test]
    fn test_neumann_injection() {
        let injection_bc = InjectionBC::new();
        let entity = MeshEntity::Face(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity.clone(), 2);

        // Set Neumann injection
        injection_bc.set_bc(entity.clone(), BoundaryCondition::Neumann(5.0));

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        // Apply injection
        injection_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index);

        // Verify RHS update
        assert_eq!(rhs_mut[(2, 0)], 5.0);
    }
}
