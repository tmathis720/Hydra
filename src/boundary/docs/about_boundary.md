The `Boundary` module provides functionality for applying boundary conditions to mesh entities within a computational domain mesh. 

This capability is essential for solving partial differential equations (PDEs) in computational fluid dynamics (CFD), where boundary conditions at domain boundaries influence simulation accuracy. The module’s core components include:
   - Boundary condition types (`Dirichlet`, `Neumann`, `Robin`, `Mixed`, `Cauchy`) and their functional variants.
   - `BoundaryConditionHandler` for centralized boundary condition management and application across mesh entities.
   - `BoundaryConditionApply` trait, which defines a standardized interface for applying conditions.

This modular design provides scalability and flexibility, enabling new boundary conditions to be added and customized without requiring major refactoring.

---

### 2. **Core Components**

#### 2.1 Boundary Condition Types (Enum Variants)
The primary boundary condition types, represented in the `BoundaryCondition` enum, are:
   - **Dirichlet**: Specifies a fixed value at the boundary.
   - **Neumann**: Specifies a flux or gradient at the boundary.
   - **Robin**: Combines Dirichlet and Neumann conditions with parameters `alpha` and `beta`.
   - **Mixed**: Specifies a combination of field value and flux, using parameters `gamma` and `delta`.
   - **Cauchy**: Combines boundary value and gradient, using parameters `lambda` and `mu`.
   - **Functional Variants**: Function-based forms of Dirichlet and Neumann conditions allow for time- or position-dependent values, broadening the applicability of boundary conditions to time-evolving or spatially varying scenarios.

Each boundary condition type is encapsulated in its own file (e.g., `dirichlet.rs`, `neumann.rs`), where specific logic is implemented for modifying the system matrices and RHS vectors as per the mathematical representation of each boundary type.

#### 2.2 BoundaryConditionHandler
`BoundaryConditionHandler` serves as a central manager for boundary conditions. It employs a `DashMap` for concurrent access, mapping each mesh entity to its associated boundary condition. The main methods include:
   - `new`: Initializes the handler.
   - `set_bc`: Assigns a boundary condition to a mesh entity.
   - `get_bc`: Retrieves the assigned condition for a given entity.
   - `apply_bc`: Modifies the system matrix and RHS to reflect the boundary effects, with boundary conditions applied uniformly through the `BoundaryConditionApply` trait.

`BoundaryConditionHandler` is linked with `MeshEntity` objects from the `Domain` module, and each entity’s matrix indices are identified via `entity_to_index` mapping for efficient matrix updates.

#### 2.3 BoundaryConditionApply Trait
This trait defines a generic `apply` method, implemented by each boundary condition type. It allows all boundary conditions to be applied uniformly across mesh entities, improving modularity and extensibility. By implementing `BoundaryConditionApply`, each boundary type specifies how it interacts with the system matrix and RHS vector, ensuring boundary conditions are applied consistently across the domain.

---

### 3. **Boundary Condition Types**

#### 3.1 Dirichlet Boundary Condition (`dirichlet.rs`)
Dirichlet conditions specify fixed values on boundaries, modifying both the matrix and RHS. The `apply_constant_dirichlet` method zeros out non-diagonal matrix entries and sets the RHS to the Dirichlet value. Functional variants, `DirichletFn`, allow for time-dependent values, adapting boundary conditions to changing simulation states.

#### 3.2 Neumann Boundary Condition (`neumann.rs`)
Neumann conditions specify fluxes across boundaries, impacting only the RHS. The `apply_constant_neumann` method directly modifies the RHS based on the specified flux. The `NeumannFn` variant allows for time- or position-dependent fluxes, useful for modeling dynamic outflows.

#### 3.3 Robin Boundary Condition (`robin.rs`)
Robin conditions combine Dirichlet and Neumann conditions, adjusting both the matrix and RHS based on `alpha` and `beta` parameters. They are well-suited for convective boundaries, where values and fluxes interact.

#### 3.4 Mixed Boundary Condition (`mixed.rs`)
The Mixed boundary condition is represented by parameters `gamma` and `delta`, combining field values and fluxes. This type adds flexibility for boundary treatments that involve both fixed values and gradients, impacting both the matrix and RHS as per the specified gamma-delta relationship.

#### 3.5 Cauchy Boundary Condition (`cauchy.rs`)
Cauchy conditions, governed by `lambda` and `mu`, blend boundary values with gradients, impacting both the matrix and RHS. This type is essential in stress-boundary conditions where both value and flux influence boundary behavior.

---

### 4. **BoundaryConditionHandler**

#### 4.1 Structure and Initialization
The handler uses a `DashMap<MeshEntity, BoundaryCondition>`, initialized through `new`. This concurrent map allows for safe, parallel access to boundary conditions, setting up efficient modification for multi-threaded applications.

```rust
use dashmap::DashMap;

/// Mock `MeshEntity` to represent an entity in the computational mesh.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct MeshEntity {
    id: u32,
}

/// Enumeration of possible boundary conditions.
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
}

/// Handler for managing boundary conditions on mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl BoundaryConditionHandler {
    /// Creates a new `BoundaryConditionHandler`.
    ///
    /// # Examples
    ///
    /// ```
    /// let handler = BoundaryConditionHandler::new();
    /// assert!(handler.conditions.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }
}

```

#### 4.2 Setting and Retrieving Boundary Conditions
   - **`set_bc`** associates a boundary condition with a specific `MeshEntity`, preparing each boundary entity for simulation.
   - **`get_bc`** retrieves the boundary condition for a given entity, returning `None` if none is set.

#### 4.3 Applying Boundary Conditions
The `apply_bc` method applies boundary conditions by iterating through boundary entities and using `apply` methods from `BoundaryConditionApply`. This method modifies the system matrix and RHS based on the specific boundary type, calling helper functions to handle each condition type:

```rust
use dashmap::DashMap;

/// Mock `MeshEntity` to represent an entity in the computational mesh.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct MeshEntity {
    id: u32,
}

/// Enumeration of possible boundary conditions.
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
}

/// Mock `MatMut` to represent a mutable matrix structure.
pub struct MatMut<T> {
    data: Vec<Vec<T>>,
}

impl<T> MatMut<T> {
    pub fn new(rows: usize, cols: usize, initial_value: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![vec![initial_value; cols]; rows],
        }
    }

    pub fn write(&mut self, row: usize, col: usize, value: T) {
        self.data[row][col] = value;
    }

    pub fn read(&self, row: usize, col: usize) -> &T {
        &self.data[row][col]
    }
}

impl BoundaryCondition {
    /// Mock `apply` method to simulate the boundary condition application.
    pub fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        let index = *entity_to_index.get(entity).unwrap();
        match self {
            BoundaryCondition::Dirichlet(value) => {
                matrix.write(index, index, *value);
                rhs.write(index, 0, *value);
            }
            BoundaryCondition::Neumann(flux) => {
                rhs.write(index, 0, rhs.read(index, 0) + flux);
            }
        }
    }
}

/// Handler for managing boundary conditions on mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl BoundaryConditionHandler {
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    /// Applies the boundary conditions to the system matrix and RHS vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut handler = BoundaryConditionHandler::new();
    /// let entity = MeshEntity { id: 1 };
    /// handler.set_bc(entity.clone(), BoundaryCondition::Dirichlet(10.0));
    ///
    /// let mut matrix = MatMut::new(3, 3, 0.0);
    /// let mut rhs = MatMut::new(3, 1, 0.0);
    /// let mut entity_to_index = DashMap::new();
    /// entity_to_index.insert(entity.clone(), 1);
    ///
    /// handler.apply_bc(&mut matrix, &mut rhs, &[entity.clone()], &entity_to_index, 0.0);
    ///
    /// assert_eq!(*matrix.read(1, 1), 10.0);
    /// assert_eq!(*rhs.read(1, 0), 10.0);
    /// ```
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
                bc.apply(entity, rhs, matrix, entity_to_index, time);
            }
        }
    }
}
```

#### 4.4 Integration with Domain and Solver Modules
By utilizing the `Domain` module’s `MeshEntity` objects and maintaining an `entity_to_index` map, `BoundaryConditionHandler` ensures conditions are applied at the correct matrix indices. During solver execution, this setup translates boundary effects into matrix modifications, ensuring that boundary interactions are accurately represented.

---

### 5. **BoundaryConditionApply Trait**

#### 5.1 Trait Definition
The `BoundaryConditionApply` trait provides a unified interface, ensuring each boundary type can modify system matrices and RHS vectors consistently. This modular approach improves maintainability and makes future extensions straightforward.

```rust

use dashmap::DashMap;

/// Mock `MeshEntity` struct to represent an entity in the computational mesh.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct MeshEntity {
    id: u32,
}

/// Mock `MatMut` struct to represent a mutable matrix.
pub struct MatMut<T> {
    data: Vec<Vec<T>>,
}

impl<T> MatMut<T> {
    /// Creates a new matrix with given dimensions and initial value.
    pub fn new(rows: usize, cols: usize, initial_value: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![vec![initial_value; cols]; rows],
        }
    }

    /// Writes a value to the matrix at the specified row and column.
    pub fn write(&mut self, row: usize, col: usize, value: T) {
        self.data[row][col] = value;
    }

    /// Reads a value from the matrix at the specified row and column.
    pub fn read(&self, row: usize, col: usize) -> &T {
        &self.data[row][col]
    }
}

/// Trait to apply boundary conditions to a mesh entity within the matrix and RHS vector.
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

/// Dirichlet boundary condition implementation.
pub struct DirichletCondition {
    value: f64,
}

impl BoundaryConditionApply for DirichletCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        if let Some(index) = entity_to_index.get(entity) {
            matrix.write(*index, *index, 1.0);
            rhs.write(*index, 0, self.value);
        }
    }
}

/// Neumann boundary condition implementation.
pub struct NeumannCondition {
    flux: f64,
}

impl BoundaryConditionApply for NeumannCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        _matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        _time: f64,
    ) {
        if let Some(index) = entity_to_index.get(entity) {
            rhs.write(*index, 0, rhs.read(*index, 0) + self.flux);
        }
    }
}
```

#### 5.2 Implementation for Each Boundary Condition Type
Each boundary type implements `BoundaryConditionApply`:
   - **Dirichlet** and **DirichletFn** set matrix rows for fixed values.
   - **Neumann** and **NeumannFn** update only the RHS.
   - **Robin** conditions adjust both matrix and RHS.
   - **Mixed** and **Cauchy** conditions influence both matrix and RHS with their respective parameters, supporting stress or field-value relationships at boundaries.

---

### 6. **Summary and Extensibility**

The `Boundary` module’s extensible and modular structure supports diverse boundary condition requirements. Through `BoundaryConditionApply` and `BoundaryConditionHandler`, the module integrates effectively with `Domain` and `Solver`, enabling consistent boundary applications and adaptability to future computational demands.