use crate::domain::{Section, MeshEntity};
use rustc_hash::FxHashMap;
use faer_core::{inner::DenseColMajor, Matrix};

pub struct DirichletBC {
    pub values: FxHashMap<MeshEntity, f64>,  // Map boundary entities to their prescribed values
}

impl DirichletBC {
    /// Creates a new DirichletBC structure
    pub fn new() -> Self {
        DirichletBC {
            values: FxHashMap::default(),
        }
    }

    /// Set a Dirichlet boundary condition for a given mesh entity
    pub fn set_bc(&mut self, entity: MeshEntity, value: f64) {
        self.values.insert(entity, value);
    }

    /// Check if a mesh entity has a Dirichlet boundary condition
    pub fn is_bc(&self, entity: &MeshEntity) -> bool {
        self.values.contains_key(entity)
    }

    /// Get the Dirichlet value for a mesh entity
    pub fn get_value(&self, entity: &MeshEntity) -> f64 {
        *self.values.get(entity).expect(&format!(
            "Dirichlet BC not set for entity {:?}",
            entity
        ))
    }

    /// Apply the Dirichlet boundary condition to the system matrix and RHS vector
    ///
    /// `matrix`: The system matrix (`faer::Matrix`)
    /// `rhs`: The RHS vector (`faer::Matrix` for a column vector)
    /// `entity_to_index`: Mapping from MeshEntity to matrix index
    pub fn apply_bc(
        &self,
        matrix: &mut Matrix<DenseColMajor<f64>>,  // Replacing DMatrix<f64> with faer::Matrix
        rhs: &mut Matrix<DenseColMajor<f64>>,    // Replacing DVector<f64> with faer::Matrix (1D column matrix)
        entity_to_index: &FxHashMap<MeshEntity, usize>,
    ) {
        for (entity, &value) in &self.values {
            if let Some(&index) = entity_to_index.get(entity) {
                // Set the corresponding row in the matrix to zero
                for j in 0..matrix.ncols() {
                    matrix.write(index, j, 0.0);
                }
                // Set the diagonal to 1
                matrix.write(index, index, 1.0);
                // Set the RHS value
                rhs.write(index, 0, value);
            } else {
                panic!("Entity {:?} not found in entity_to_index mapping", entity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use rustc_hash::FxHashMap;  // Updated to use FxHashMap instead of std::collections::HashMap
    use faer_core::{inner::DenseColMajor, Matrix};

    #[test]
    fn test_dirichlet_bc() {
        // Create a mock solution vector and system matrix
        let mut matrix = Matrix::<DenseColMajor<f64>>::zeros(5, 5);  // 5x5 system matrix
        let mut rhs = Matrix::<DenseColMajor<f64>>::zeros(5, 1);     // 5x1 RHS vector (column matrix)

        // Create the DirichletBC structure
        let mut dirichlet_bc = DirichletBC::new();

        // Define a couple of boundary entities
        let boundary_entity_1 = MeshEntity::Cell(1);
        let boundary_entity_2 = MeshEntity::Cell(3);

        // Set boundary conditions for those entities
        dirichlet_bc.set_bc(boundary_entity_1, 100.0);
        dirichlet_bc.set_bc(boundary_entity_2, 50.0);

        // Create entity to index mapping
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(MeshEntity::Cell(0), 0);
        entity_to_index.insert(MeshEntity::Cell(1), 1);
        entity_to_index.insert(MeshEntity::Cell(2), 2);
        entity_to_index.insert(MeshEntity::Cell(3), 3);
        entity_to_index.insert(MeshEntity::Cell(4), 4);

        // Apply the boundary conditions to the matrix and RHS
        dirichlet_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index);

        // Check that the RHS vector was modified correctly
        assert_eq!(rhs.read(1, 0), 100.0);
        assert_eq!(rhs.read(3, 0), 50.0);

        // Check that the system matrix was modified (rows corresponding to boundary conditions)
        for j in 0..5 {
            assert_eq!(matrix.read(1, j), if j == 1 { 1.0 } else { 0.0 });
            assert_eq!(matrix.read(3, j), if j == 3 { 1.0 } else { 0.0 });
        }
    }
}
