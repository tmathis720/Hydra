use crate::domain::MeshEntity;
use rustc_hash::FxHashMap;
use faer::MatMut;

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
    /// `matrix`: The system matrix (`faer::Mat`)
    /// `rhs`: The RHS vector (`faer::Mat` for a column vector)
    /// `entity_to_index`: Mapping from MeshEntity to matrix index
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,  // Mutable system matrix
        rhs: &mut MatMut<f64>,     // Mutable RHS vector (1D column matrix)
        entity_to_index: &FxHashMap<MeshEntity, usize>,
    ) {
        for (entity, &value) in &self.values {
            if let Some(&index) = entity_to_index.get(entity) {
                // Set the corresponding row in the matrix to zero (all entries in that row)
                for j in 0..matrix.ncols() {
                    matrix.write(index, j, 0.0);  // Write 0 to all elements in the row
                }
                // Set the diagonal to 1
                matrix.write(index, index, 1.0);  // Set the diagonal element to 1
                // Set the RHS value
                rhs.write(index, 0, value);  // Write the boundary value in the RHS
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
    use rustc_hash::FxHashMap;
    use faer::Mat;

    #[test]
    fn test_dirichlet_bc() {
        // Create a 5x5 system matrix and a 5x1 RHS vector
        let mut matrix = Mat::<f64>::zeros(5, 5);
        let mut rhs = Mat::<f64>::zeros(5, 1);

        // Create the DirichletBC structure
        let mut dirichlet_bc = DirichletBC::new();

        // Define a couple of boundary entities
        let boundary_entity_1 = MeshEntity::Cell(1);
        let boundary_entity_2 = MeshEntity::Cell(3);

        // Set boundary conditions for those entities
        dirichlet_bc.set_bc(boundary_entity_1, 100.0);
        dirichlet_bc.set_bc(boundary_entity_2, 50.0);

        // Create entity-to-index mapping
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(MeshEntity::Cell(0), 0);
        entity_to_index.insert(MeshEntity::Cell(1), 1);
        entity_to_index.insert(MeshEntity::Cell(2), 2);
        entity_to_index.insert(MeshEntity::Cell(3), 3);
        entity_to_index.insert(MeshEntity::Cell(4), 4);

        // Apply the boundary conditions to the matrix and RHS
        let mut matrix_mut = matrix.as_mut();  // Mutable access
        let mut rhs_mut = rhs.as_mut();        // Mutable access
        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index);

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
