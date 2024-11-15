Generate a detailed outline for a users guide for the `Boundary` module for Hydra. The outline should provide high level details which can later be fleshed out. I am going to provide the code for all of the parts of the `Boundary` module below, and you can analyze and build the detailed outline based on this version of the source code. Retain the details of your analysis for later as we will be going through the outline in detail throughout this conversation and refining it.

---

`src/boundary/bc_handler.rs`

```rust
use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// BoundaryCondition represents various types of boundary conditions
/// that can be applied to mesh entities.
#[derive(Clone)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}

/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl BoundaryConditionHandler {
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    /// Applies the boundary conditions to the system matrices and right-hand side vectors.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                let index = *entity_to_index.get(entity).unwrap();
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(matrix, rhs, index, alpha, beta);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, value);
                    }
                    BoundaryCondition::Mixed { gamma, delta } => {
                        let mixed_bc = MixedBC::new();
                        mixed_bc.apply_mixed(matrix, rhs, index, gamma, delta);
                    }
                    BoundaryCondition::Cauchy { lambda, mu } => {
                        let cauchy_bc = CauchyBC::new();
                        cauchy_bc.apply_cauchy(matrix, rhs, index, lambda, mu);
                    }
                }
            }
        }
    }
}

/// The BoundaryConditionApply trait defines the `apply` method, which is used to apply 
/// a boundary condition to a given mesh entity.
pub trait BoundaryConditionApply {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    );
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let index = *entity_to_index.get(entity).unwrap();
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, *flux);
            }
            BoundaryCondition::Robin { alpha, beta } => {
                let robin_bc = RobinBC::new();
                robin_bc.apply_robin(matrix, rhs, index, *alpha, *beta);
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, value);
            }
            BoundaryCondition::Mixed { gamma, delta } => {
                let mixed_bc = MixedBC::new();
                mixed_bc.apply_mixed(matrix, rhs, index, *gamma, *delta);
            }
            BoundaryCondition::Cauchy { lambda, mu } => {
                let cauchy_bc = CauchyBC::new();
                cauchy_bc.apply_cauchy(matrix, rhs, index, *lambda, *mu);
            }
        }
    }
}
```

---

`src/boundary/cauchy.rs`

```rust
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `CauchyBC` struct represents a handler for applying Cauchy boundary conditions
/// to a set of mesh entities. Cauchy boundary conditions typically involve both the
/// value of a state variable and its derivative, defined by parameters `lambda` and `mu`.
/// These conditions influence both the system matrix and the right-hand side vector.
pub struct CauchyBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl CauchyBC {
    /// Creates a new instance of `CauchyBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Cauchy boundary condition for a specific mesh entity.
    ///
    /// # Parameters
    /// - `entity`: The mesh entity to which the boundary condition applies.
    /// - `condition`: The boundary condition to apply, specified as a `BoundaryCondition`.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Cauchy boundary conditions to the system matrix and RHS vector.
    /// This method iterates over all stored conditions, updating both the matrix and RHS
    /// based on the specified lambda and mu values for each entity.
    ///
    /// # Parameters
    /// - `matrix`: The system matrix to be modified by the boundary condition.
    /// - `rhs`: The right-hand side vector to be modified by the boundary condition.
    /// - `index`: The index within the matrix and RHS corresponding to the mesh entity.
    /// - `lambda`: Coefficient for the matrix modification.
    /// - `mu`: Value to adjust the RHS vector.
    pub fn apply_cauchy(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        lambda: f64,
        mu: f64,
    ) {
        // Apply the lambda factor to the matrix at the diagonal index
        matrix.write(index, index, matrix.read(index, index) + lambda);
        // Modify the RHS with the mu value at the specific index
        rhs.write(index, 0, rhs.read(index, 0) + mu);
    }

    /// Applies all Cauchy boundary conditions within the handler to the system.
    /// It fetches the index of each entity and applies the corresponding Cauchy boundary 
    /// condition values (lambda and mu) to the matrix and RHS.
    ///
    /// # Parameters
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `entity_to_index`: Mapping from `MeshEntity` to matrix indices.
    /// - `time`: Current time, if time-dependent boundary values are desired.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                if let BoundaryCondition::Cauchy { lambda, mu } = condition {
                    self.apply_cauchy(matrix, rhs, index, *lambda, *mu);
                }
            }
        }
    }
}

impl BoundaryConditionApply for CauchyBC {
    /// Applies the stored Cauchy boundary conditions for a specific mesh entity.
    ///
    /// # Parameters
    /// - `entity`: Reference to the mesh entity for which the boundary condition is applied.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `entity_to_index`: Reference to the mapping of entities to matrix indices.
    /// - `time`: Current time, allowing for time-dependent boundary conditions.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let cauchy_bc = CauchyBC::new();
        let entity = MeshEntity::Vertex(1);

        cauchy_bc.set_bc(entity, BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 });

        let condition = cauchy_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 })));
    }

    #[test]
    fn test_apply_cauchy_bc() {
        let cauchy_bc = CauchyBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        cauchy_bc.set_bc(entity, BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 });

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        cauchy_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Assert the matrix diagonal at index 1 has been incremented by lambda
        assert_eq!(matrix_mut[(1, 1)], 2.5);  // Initial value 1.0 + lambda 1.5
        // Assert the RHS value at index 1 has been incremented by mu
        assert_eq!(rhs_mut[(1, 0)], 2.5);
    }
}
```

---

`src/boundary/dirichlet.rs`

```rust
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `DirichletBC` struct represents a handler for applying Dirichlet boundary conditions 
/// to a set of mesh entities. It stores the conditions in a `DashMap` and applies them to 
/// modify both the system matrix and the right-hand side (rhs).
pub struct DirichletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl DirichletBC {
    /// Creates a new instance of `DirichletBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Dirichlet boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Dirichlet boundary conditions to the system matrix and rhs. 
    /// It iterates over the stored conditions and applies either constant or function-based Dirichlet
    /// boundary conditions to the corresponding entities.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        // Iterate through the conditions and apply each condition accordingly.
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::Dirichlet(value) => {
                        self.apply_constant_dirichlet(matrix, rhs, index, *value);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = self.get_coordinates(entity);
                        let value = fn_bc(time, &coords);
                        self.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    _ => {}
                }
            }
        });
    }

    /// Applies a constant Dirichlet boundary condition to the matrix and rhs for a specific index.
    pub fn apply_constant_dirichlet(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        value: f64,
    ) {
        let ncols = matrix.ncols();
        for col in 0..ncols {
            matrix.write(index, col, 0.0);
        }
        matrix.write(index, index, 1.0);
        rhs.write(index, 0, value);
    }

    /// Retrieves the coordinates of the mesh entity (placeholder for real coordinates).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for DirichletBC {
    /// Applies the stored Dirichlet boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;
    use std::sync::Arc;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        
        // Set a Dirichlet boundary condition
        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(10.0));
        
        // Verify that the condition was set correctly
        let condition = dirichlet_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Dirichlet(10.0))));
    }

    #[test]
    fn test_apply_constant_dirichlet() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(5.0));
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        for col in 0..matrix_mut.ncols() {
            if col == 1 {
                assert_eq!(matrix_mut[(1, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(1, col)], 0.0);
            }
        }
        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_dirichlet() {
        let dirichlet_bc = DirichletBC::new();
        let entity = MeshEntity::Vertex(2);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 2);

        dirichlet_bc.set_bc(
            entity,
            BoundaryCondition::DirichletFn(Arc::new(|_time: f64, _coords: &[f64]| 7.0)),
        );

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        dirichlet_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 1.0);

        for col in 0..matrix_mut.ncols() {
            if col == 2 {
                assert_eq!(matrix_mut[(2, col)], 1.0);
            } else {
                assert_eq!(matrix_mut[(2, col)], 0.0);
            }
        }
        assert_eq!(rhs_mut[(2, 0)], 7.0);
    }
}
```

---

`src/boundary/mixed.rs`

```rust
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `MixedBC` struct represents a handler for applying Mixed boundary conditions
/// to a set of mesh entities. Mixed boundary conditions typically involve a combination
/// of Dirichlet and Neumann-type parameters that modify both the system matrix 
/// and the right-hand side (RHS) vector.
pub struct MixedBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl MixedBC {
    /// Creates a new instance of `MixedBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Mixed boundary condition for a specific mesh entity.
    /// 
    /// # Parameters
    /// - `entity`: The mesh entity to which the boundary condition applies.
    /// - `condition`: The boundary condition to apply, specified as a `BoundaryCondition`.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Mixed boundary conditions to the system matrix and RHS vector.
    /// This method iterates over all stored conditions, updating both the matrix and RHS
    /// based on the specified gamma and delta values for each entity.
    /// 
    /// # Parameters
    /// - `matrix`: The system matrix to be modified by the boundary condition.
    /// - `rhs`: The right-hand side vector to be modified by the boundary condition.
    /// - `index`: The index within the matrix and RHS corresponding to the mesh entity.
    /// - `gamma`: Coefficient applied to the system matrix for this boundary.
    /// - `delta`: Value added to the RHS vector for this boundary.
    pub fn apply_mixed(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        gamma: f64,
        delta: f64,
    ) {
        // Apply the gamma factor to the matrix at the diagonal index
        matrix.write(index, index, matrix.read(index, index) + gamma);
        // Modify the RHS with the delta value at the specific index
        rhs.write(index, 0, rhs.read(index, 0) + delta);
    }

    /// Applies all Mixed boundary conditions within the handler to the system.
    /// It fetches the index of each entity and applies the corresponding mixed boundary 
    /// condition values (gamma and delta) to the matrix and RHS.
    /// 
    /// # Parameters
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `entity_to_index`: Mapping from `MeshEntity` to matrix indices.
    /// - `time`: Current time, if time-dependent boundary values are desired.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        for entry in self.conditions.iter() {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                if let BoundaryCondition::Mixed { gamma, delta } = condition {
                    self.apply_mixed(matrix, rhs, index, *gamma, *delta);
                }
            }
        }
    }
}

impl BoundaryConditionApply for MixedBC {
    /// Applies the stored Mixed boundary conditions for a specific mesh entity.
    /// 
    /// # Parameters
    /// - `entity`: Reference to the mesh entity for which the boundary condition is applied.
    /// - `rhs`: Mutable reference to the RHS vector.
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `entity_to_index`: Reference to the mapping of entities to matrix indices.
    /// - `time`: Current time, allowing for time-dependent boundary conditions.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let mixed_bc = MixedBC::new();
        let entity = MeshEntity::Vertex(1);

        mixed_bc.set_bc(entity, BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 });

        let condition = mixed_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 })));
    }

    #[test]
    fn test_apply_mixed_bc() {
        let mixed_bc = MixedBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        mixed_bc.set_bc(entity, BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 });

        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        mixed_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        // Assert the matrix diagonal at index 1 has been incremented by gamma
        assert_eq!(matrix_mut[(1, 1)], 3.0);  // 1.0 initial + 2.0 gamma
        // Assert the RHS value at index 1 has been incremented by delta
        assert_eq!(rhs_mut[(1, 0)], 3.0);
    }
}
```

---

`src/boundary/mod.rs`

```rust
// src/boundary/mod.rs
pub mod bc_handler;
pub mod dirichlet;
pub mod neumann;
pub mod robin;
pub mod cauchy;
pub mod mixed;
```

---

`src/boundary/neumann.rs`

```rust
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `NeumannBC` struct represents a handler for applying Neumann boundary conditions 
/// to a set of mesh entities. Neumann boundary conditions involve specifying the flux across 
/// a boundary, and they modify only the right-hand side (RHS) of the system without modifying 
/// the system matrix.
pub struct NeumannBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl NeumannBC {
    /// Creates a new instance of `NeumannBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Neumann boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Neumann boundary conditions to the right-hand side (RHS) of the system. 
    /// It iterates over the stored conditions and applies either constant or function-based Neumann
    /// boundary conditions to the corresponding entities.
    pub fn apply_bc(
        &self,
        _matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
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
        });
    }

    /// Applies a constant Neumann boundary condition to the right-hand side (RHS) for a specific index.
    pub fn apply_constant_neumann(&self, rhs: &mut MatMut<f64>, index: usize, value: f64) {
        rhs.write(index, 0, rhs.read(index, 0) + value);
    }

    /// Retrieves the coordinates of the mesh entity (currently a placeholder).
    fn get_coordinates(&self, _entity: &MeshEntity) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }
}

impl BoundaryConditionApply for NeumannBC {
    /// Applies the stored Neumann boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        _matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(_matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;
    use std::sync::Arc;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        
        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(10.0));
        
        let condition = neumann_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Neumann(10.0))));
    }

    #[test]
    fn test_apply_constant_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        neumann_bc.set_bc(entity, BoundaryCondition::Neumann(5.0));
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut rhs_mut = rhs.as_mut();

        neumann_bc.apply_bc(&mut matrix.as_mut(), &mut rhs_mut, &entity_to_index, 0.0);

        assert_eq!(rhs_mut[(1, 0)], 5.0);
    }

    #[test]
    fn test_apply_function_based_neumann() {
        let neumann_bc = NeumannBC::new();
        let entity = MeshEntity::Vertex(2);
        let entity_to_index = DashMap::new();
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
```

---

`src/boundary/robin.rs`

```rust
use dashmap::DashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionApply};
use faer::MatMut;

/// The `RobinBC` struct represents a handler for applying Robin boundary conditions 
/// to a set of mesh entities. Robin boundary conditions involve a linear combination 
/// of Dirichlet and Neumann boundary conditions, and they modify both the system matrix 
/// and the right-hand side (RHS).
pub struct RobinBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl RobinBC {
    /// Creates a new instance of `RobinBC` with an empty `DashMap` to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a Robin boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Applies the stored Robin boundary conditions to both the system matrix and rhs. 
    /// It iterates over the stored conditions and applies the Robin boundary condition 
    /// to the corresponding entities.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        self.conditions.iter().for_each(|entry| {
            let (entity, condition) = entry.pair();
            if let Some(index) = entity_to_index.get(entity).map(|i| *i) {
                match condition {
                    BoundaryCondition::Robin { alpha, beta } => {
                        self.apply_robin(matrix, rhs, index, *alpha, *beta);
                    }
                    _ => {}
                }
            }
        });
    }

    /// Applies a Robin boundary condition to the system matrix and rhs for a specific index.
    pub fn apply_robin(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        index: usize,
        alpha: f64,
        beta: f64,
    ) {
        matrix.write(index, index, matrix.read(index, index) + alpha);
        rhs.write(index, 0, rhs.read(index, 0) + beta);
    }
}

impl BoundaryConditionApply for RobinBC {
    /// Applies the stored Robin boundary conditions for a specific mesh entity.
    fn apply(
        &self,
        _entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        self.apply_bc(matrix, rhs, entity_to_index, time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use crate::domain::mesh_entity::MeshEntity;

    fn create_test_matrix_and_rhs() -> (Mat<f64>, Mat<f64>) {
        let matrix = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let rhs = Mat::zeros(3, 1);
        (matrix, rhs)
    }

    #[test]
    fn test_set_bc() {
        let robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        
        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        let condition = robin_bc.conditions.get(&entity).map(|entry| entry.clone());
        assert!(matches!(condition, Some(BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 })));
    }

    #[test]
    fn test_apply_robin_bc() {
        let robin_bc = RobinBC::new();
        let entity = MeshEntity::Vertex(1);
        let entity_to_index = DashMap::new();
        entity_to_index.insert(entity, 1);

        robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
        
        let (mut matrix, mut rhs) = create_test_matrix_and_rhs();
        let mut matrix_mut = matrix.as_mut();
        let mut rhs_mut = rhs.as_mut();

        robin_bc.apply_bc(&mut matrix_mut, &mut rhs_mut, &entity_to_index, 0.0);

        assert_eq!(matrix_mut[(1, 1)], 3.0);  // Initial value 1.0 + alpha 2.0
        assert_eq!(rhs_mut[(1, 0)], 3.0);    // Beta term applied
    }
}
```