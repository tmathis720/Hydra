//! Periodic Boundary Conditions
//! Implements periodic boundary conditions for mesh entities.

use crate::boundary::bc_handler::BoundaryConditionApply;
use faer::MatMut;
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;

/// Struct for managing periodic boundary conditions.
/// Maps entities to their periodic counterparts.
pub struct PeriodicBC {
    pairs: DashMap<MeshEntity, MeshEntity>, // Map each entity to its periodic counterpart.
}

impl PeriodicBC {
    /// Creates a new instance of PeriodicBC.
    pub fn new() -> Self {
        Self {
            pairs: DashMap::new(),
        }
    }

    /// Sets a pair of periodic entities.
    /// 
    /// # Parameters
    /// - `entity`: The mesh entity.
    /// - `counterpart`: The corresponding periodic counterpart.
    pub fn set_pair(&self, entity: MeshEntity, counterpart: MeshEntity) {
        self.pairs.insert(entity, counterpart);
        self.pairs.insert(counterpart, entity); // Ensure bidirectional mapping.
    }

    /// Applies periodic boundary conditions to the system matrix and RHS vector.
    /// 
    /// This ensures that the degrees of freedom at periodic pairs are identical.
    /// 
    /// # Parameters
    /// - `matrix`: The system matrix to modify.
    /// - `rhs`: The right-hand side vector to adjust.
    /// - `entity_to_index`: Map from mesh entities to matrix indices.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
    ) {
        for entry in self.pairs.iter() {
            let (entity, counterpart) = entry.pair();
            if let (Some(idx1), Some(idx2)) = (
                entity_to_index.get(entity),
                entity_to_index.get(counterpart),
            ) {
                // Enforce periodic constraints by modifying the system matrix.
                for col in 0..matrix.ncols() {
                    let avg = (matrix.read(*idx1, col) + matrix.read(*idx2, col)) / 2.0;
                    matrix.write(*idx1, col, avg);
                    matrix.write(*idx2, col, avg);
                }

                // Adjust the RHS vector to match periodic conditions.
                let rhs_avg = (rhs.read(*idx1, 0) + rhs.read(*idx2, 0)) / 2.0;
                rhs.write(*idx1, 0, rhs_avg);
                rhs.write(*idx2, 0, rhs_avg);
            }
        }
    }
}

impl BoundaryConditionApply for PeriodicBC {
    /// Applies periodic boundary conditions for a specific mesh entity.
    /// 
    /// # Parameters
    /// - `entity`: The mesh entity for which the boundary condition is applied.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `entity_to_index`: Mapping of mesh entities to matrix indices.
    /// - `time`: Current time (not used for periodic conditions but included for consistency).
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        if let Some(counterpart) = self.pairs.get(entity) {
            if let (Some(idx1), Some(idx2)) = (
                entity_to_index.get(entity),
                entity_to_index.get(counterpart.key()),
            ) {
                // Enforce periodic constraint for this specific pair.
                for col in 0..matrix.ncols() {
                    let avg = (matrix.read(*idx1, col) + matrix.read(*idx2, col)) / 2.0;
                    matrix.write(*idx1, col, avg);
                    matrix.write(*idx2, col, avg);
                }

                let rhs_avg = (rhs.read(*idx1, 0) + rhs.read(*idx2, 0)) / 2.0;
                rhs.write(*idx1, 0, rhs_avg);
                rhs.write(*idx2, 0, rhs_avg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(4, 4, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::from_fn(4, 1, |i, _| i as f64 + 1.0);
        (matrix, rhs)
    }

    #[test]
    fn test_set_pair() {
        let periodic_bc = PeriodicBC::new();
        let entity1 = MeshEntity::Face(1);
        let entity2 = MeshEntity::Face(2);

        periodic_bc.set_pair(entity1.clone(), entity2.clone());

        assert_eq!(periodic_bc.pairs.get(&entity1).map(|v| v.clone()), Some(entity2));
        assert_eq!(periodic_bc.pairs.get(&entity2).map(|v| v.clone()), Some(entity1));
    }

    #[test]
    fn test_apply_periodic_bc() {
        let periodic_bc = PeriodicBC::new();
        let entity1 = MeshEntity::Face(1);
        let entity2 = MeshEntity::Face(2);

        periodic_bc.set_pair(entity1.clone(), entity2.clone());

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity1.clone(), 1);
        entity_to_index.insert(entity2.clone(), 2);

        periodic_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

        // Verify that matrix and RHS values at indices 1 and 2 are averaged.
        for col in 0..matrix_mut.ncols() {
            let avg = (matrix[(1, col)] + matrix[(2, col)]) / 2.0;
            assert_eq!(matrix[(1, col)], avg);
            assert_eq!(matrix[(2, col)], avg);
        }

        let rhs_avg = (rhs[(1, 0)] + rhs[(2, 0)]) / 2.0;
        assert_eq!(rhs[(1, 0)], rhs_avg);
        assert_eq!(rhs[(2, 0)], rhs_avg);
    }
}
