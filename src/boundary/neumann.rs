// src/boundary/neumann.rs

use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use crate::domain::section::Section;
use faer::MatMut;

/// The `NeumannBC` struct represents a handler for applying Neumann boundary conditions 
/// to a set of mesh entities. Neumann boundary conditions involve specifying the flux across 
/// a boundary, and they modify only the right-hand side (RHS) of the system without modifying 
/// the system matrix.
/// 
/// Example usage:
/// 
///    let neumann_bc = NeumannBC::new();  
///    let entity = MeshEntity::Vertex(1);  
///    neumann_bc.set_bc(entity, BoundaryCondition::Neumann(10.0));  
///    neumann_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, 0.0);  
/// 
pub struct NeumannBC {
    conditions: Section<BoundaryCondition>,
}

impl NeumannBC {
    /// Creates a new instance of `NeumannBC` with an empty section to store boundary conditions.
    /// 
    /// Example usage:
    /// 
    ///    let neumann_bc = NeumannBC::new();  
    /// 
    pub fn new() -> Self {
        Self {
            conditions: Section::new(),
        }
    }

    /// Sets a Neumann boundary condition for a specific mesh entity.
    ///
    /// # Arguments:
    /// * `entity` - The mesh entity to which the boundary condition will be applied.
    /// * `condition` - The boundary condition to set (either a constant flux or a functional form).
    ///
    /// Example usage:
    /// 
    ///    let entity = MeshEntity::Vertex(1);  
    ///    neumann_bc.set_bc(entity, BoundaryCondition::Neumann(5.0));  
    /// 
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.set_data(entity, condition);
    }

    /// Applies the stored Neumann boundary conditions to the right-hand side (RHS) of the system. 
    /// It iterates over the stored conditions and applies either constant or function-based Neumann
    /// boundary conditions to the corresponding entities.
    ///
    /// # Arguments:
    /// * `_matrix` - The mutable system matrix (unused in Neumann BC).
    /// * `rhs` - The mutable right-hand side vector.
    /// * `entity_to_index` - A hash map that associates mesh entities with their indices in the system.
    /// * `time` - The current time, used for time-dependent boundary conditions.
    ///
    /// Example usage:
    /// 
    ///    let entity_to_index = FxHashMap::default();  
    ///    neumann_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, 0.0);  
    /// 
    pub fn apply_bc(
        &self,
        _matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &FxHashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let data = self.conditions.data.read().unwrap();
        for (entity, condition) in data.iter() {
            if let Some(&index) = entity_to_index.get(entity) {
                match condition {
                    BoundaryCondition::Neumann(value) => {
                        self.apply_constant_neumann(rhs, index, *value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = self.get_coordinates(entity);
                        let value = fn_bc(time, &coords);
                        self.apply_constant_neumann(rhs, index, value);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Applies a constant Neumann boundary condition to the right-hand side (RHS) for a specific index.
    ///
    /// # Arguments:
    /// * `rhs` - The mutable right-hand side vector.
    /// * `index` - The index of the rhs corresponding to the mesh entity.
    /// * `value` - The Neumann flux to be applied.
    ///
    /// Example usage:
    /// 
    ///    neumann_bc.apply_constant_neumann(&mut rhs, 1, 5.0);  
    /// 
    pub fn apply_constant_neumann(&self, rhs: &mut MatMut<f64>, index: usize, value: f64) {
        rhs.write(index, 0, rhs.read(index, 0) + value);
    }

    /// Retrieves the coordinates of the mesh entity.
    ///
    /// This method currently returns a default placeholder value, but it can be expanded 
    /// to extract real entity coordinates if needed.
    ///
    /// # Arguments:
    /// * `_entity` - The mesh entity for which coordinates are being requested.
    ///
    /// Returns an array of default coordinates `[0.0, 0.0, 0.0]`.
    ///
    /// Example usage:
    /// 
    ///    let coords = neumann_bc.get_coordinates(&entity);  
    /// 
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for NeumannBC {
    /// Applies the stored Neumann boundary conditions for a specific mesh entity.
    ///
    /// This implementation utilizes the general `apply_bc` method to modify the rhs.
    ///
    /// # Arguments:
    /// * `_entity` - The mesh entity to which the boundary condition applies.
    /// * `rhs` - The mutable right-hand side vector.
    /// * `_matrix` - The mutable system matrix (unused in Neumann BC).
    /// * `entity_to_index` - A hash map that associates mesh entities with their indices.
    /// * `time` - The current time, used for time-dependent boundary conditions.
    ///
    /// Example usage:
    /// 
    ///    neumann_bc.apply(&entity, &mut rhs, &mut matrix, &entity_to_index, 0.0);  
    /// 
    fn apply(&self, _entity: &MeshEntity, rhs: &mut MatMut<f64>, _matrix: &mut MatMut<f64>, entity_to_index: &FxHashMap<MeshEntity, usize>, time: f64) {
        self.apply_bc(_matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;
    use std::sync::Arc;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        // Create a 3x3 test matrix initialized to identity matrix (though unused for NeumannBC)
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        
        // Create a 3x1 test RHS vector initialized to zero
        let rhs = Mat::zeros(3, 1);
        
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Neumann boundary condition
        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(10.0));
        
        // Verify that the condition was set correctly
        let condition = neumann_bc.conditions.restrict(&entity);
        assert!(matches!(condition, Some(BoundaryCondition::Neumann(10.0))));
    }

    #[test]
    fn test_apply_constant_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 1);

        // Set a Neumann boundary condition
        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(5.0));
        
        // Create a test matrix and RHS vector
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        // Apply the Neumann condition
        neumann_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index, 0.0);

        // Check that the RHS has been updated with the Neumann flux
        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(2);
        let mut entity_to_index = FxHashMap::default();
        entity_to_index.insert(entity, 2);

        neumann_bc.set_bc(
            entity,
            BoundaryCondition::NeumannFn(Arc::new(|_time: f64, _coords: &[f64]| 7.0)),
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        neumann_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index, 1.0);

        assert_eq!(rhs_mut[(2, 0)], 7.0);
    }
}
