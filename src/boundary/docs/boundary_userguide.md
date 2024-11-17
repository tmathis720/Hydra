# Hydra `Boundary` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Boundary Module](#2-overview-of-the-boundary-module)
3. [Core Structures](#3-core-structures)
   - [BoundaryCondition Enum](#boundarycondition-enum)
   - [BoundaryConditionFn Type](#boundaryconditionfn-type)
4. [Boundary Condition Handlers](#4-boundary-condition-handlers)
   - [BoundaryConditionHandler Struct](#boundaryconditionhandler-struct)
5. [Managing Boundary Conditions](#5-managing-boundary-conditions)
   - [Adding Boundary Conditions to Entities](#adding-boundary-conditions-to-entities)
   - [Retrieving Boundary Conditions](#retrieving-boundary-conditions)
6. [Applying Boundary Conditions](#6-applying-boundary-conditions)
   - [Matrix and RHS Modifications](#matrix-and-rhs-modifications)
7. [BoundaryConditionApply Trait](#7-boundaryconditionapply-trait)
8. [Specific Boundary Condition Implementations](#8-specific-boundary-condition-implementations)
   - [DirichletBC](#dirichletbc)
   - [NeumannBC](#neumannbc)
   - [RobinBC](#robinbc)
   - [MixedBC](#mixedbc)
   - [CauchyBC](#cauchybc)
9. [Working with Function-Based Boundary Conditions](#9-working-with-function-based-boundary-conditions)
10. [Testing and Validation](#10-testing-and-validation)
    - [Unit Testing](#unit-testing)
    - [Integration Testing](#integration-testing)
11. [Best Practices](#11-best-practices)
    - [Efficient Boundary Condition Management](#efficient-boundary-condition-management)
    - [Performance Optimization](#performance-optimization)
    - [Handling Complex Boundary Conditions](#handling-complex-boundary-conditions)
12. [Conclusion](#12-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Boundary` module of the Hydra computational framework. This module is essential for managing boundary conditions in finite volume method (FVM) simulations. Boundary conditions define how the simulation interacts with the environment outside the computational domain, and they are crucial for accurately modeling physical systems.

---

## **2. Overview of the Boundary Module**

The `Boundary` module allows users to define, manage, and apply various types of boundary conditions to mesh entities within Hydra. It supports multiple types of boundary conditions commonly used in computational fluid dynamics (CFD) and other simulation domains, including:

- **Dirichlet Conditions**: Specify fixed values for variables at the boundary.
- **Neumann Conditions**: Specify fixed fluxes (derivatives) across the boundary.
- **Robin Conditions**: Combine Dirichlet and Neumann conditions in a linear fashion.
- **Mixed Conditions**: Offer flexibility by combining characteristics of different condition types.
- **Cauchy Conditions**: Involve both the variable and its derivative, often used in elastodynamics.

The module provides a unified interface for applying these conditions to the system matrices and RHS vectors, ensuring that simulations accurately reflect the intended physical behaviors.

---

## **3. Core Structures**

### BoundaryCondition Enum

The `BoundaryCondition` enum is the core structure for specifying boundary conditions. It defines the different types of conditions that can be applied:

```rust
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}
```

- **`Dirichlet(f64)`**: Sets a fixed value at the boundary.
- **`Neumann(f64)`**: Sets a fixed flux across the boundary.
- **`Robin { alpha, beta }`**: Applies a linear combination of value and flux.
- **`Mixed { gamma, delta }`**: Custom combination of parameters.
- **`Cauchy { lambda, mu }`**: Involves both value and derivative with separate coefficients.
- **`DirichletFn` and `NeumannFn`**: Allow function-based boundary conditions.

### BoundaryConditionFn Type

For time-dependent or spatially varying boundary conditions, the module uses function pointers encapsulated in `Arc` for thread safety:

```rust
pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;
```

---

## **4. Boundary Condition Handlers**

### BoundaryConditionHandler Struct

The `BoundaryConditionHandler` manages boundary conditions across mesh entities. It maintains a `DashMap` for efficient concurrent access:

```rust
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

- **`conditions`**: Stores the mapping from mesh entities to their boundary conditions.

---

## **5. Managing Boundary Conditions**

### Adding Boundary Conditions to Entities

To assign a boundary condition to a mesh entity:

```rust
let boundary_handler = BoundaryConditionHandler::new();
boundary_handler.set_bc(entity, BoundaryCondition::Dirichlet(1.0));
```

- **`set_bc(entity, condition)`**: Assigns a boundary condition to the specified entity.

### Retrieving Boundary Conditions

To retrieve the boundary condition for a specific entity:

```rust
if let Some(condition) = boundary_handler.get_bc(&entity) {
    // Use the condition
}
```

- **`get_bc(entity)`**: Returns an `Option<BoundaryCondition>` for the entity.

---

## **6. Applying Boundary Conditions**

Boundary conditions are applied to the system's matrices and RHS vectors to enforce the specified conditions during the simulation.

```rust
boundary_handler.apply_bc(
    &mut matrix,
    &mut rhs,
    &boundary_entities,
    &entity_to_index,
    current_time,
);
```

Parameters:

- **`matrix`**: The system matrix to be modified.
- **`rhs`**: The RHS vector to be modified.
- **`boundary_entities`**: A list of entities where boundary conditions are applied.
- **`entity_to_index`**: A mapping from `MeshEntity` to indices in the matrix and RHS vector.
- **`current_time`**: The current simulation time for time-dependent conditions.

### Matrix and RHS Modifications

Each boundary condition type modifies the matrix and RHS differently:

- **Dirichlet**:
  - **Matrix**: Row corresponding to the boundary entity is zeroed out, diagonal set to 1.
  - **RHS**: Set to the Dirichlet value.
- **Neumann**:
  - **Matrix**: Unchanged.
  - **RHS**: Adjusted by the flux value.
- **Robin**, **Mixed**, **Cauchy**:
  - **Matrix**: Diagonal element adjusted according to the condition's parameters.
  - **RHS**: Adjusted by the specified values.

---

## **7. BoundaryConditionApply Trait**

The `BoundaryConditionApply` trait defines a common interface for boundary condition handlers:

```rust
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
```

- Each boundary condition handler implements this trait to define how it applies conditions to the system.

---

## **8. Specific Boundary Condition Implementations**

### DirichletBC

Handles Dirichlet boundary conditions, enforcing fixed values at boundaries.

#### Structure

```rust
pub struct DirichletBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

#### Methods

- **`new()`**: Creates a new instance.
- **`set_bc(entity, condition)`**: Assigns a Dirichlet condition to an entity.
- **`apply_bc(matrix, rhs, entity_to_index, time)`**: Applies the conditions to the system.

#### Example

```rust
let dirichlet_bc = DirichletBC::new();
dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(5.0));
dirichlet_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, current_time);
```

### NeumannBC

Handles Neumann boundary conditions, specifying fluxes across boundaries.

#### Structure

```rust
pub struct NeumannBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

#### Methods

- **`new()`**: Creates a new instance.
- **`set_bc(entity, condition)`**: Assigns a Neumann condition to an entity.
- **`apply_bc(matrix, rhs, entity_to_index, time)`**: Applies the conditions to the system.

#### Example

```rust
let neumann_bc = NeumannBC::new();
neumann_bc.set_bc(entity, BoundaryCondition::Neumann(10.0));
neumann_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, current_time);
```

### RobinBC

Handles Robin boundary conditions, combining value and flux at the boundary.

#### Structure

```rust
pub struct RobinBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

#### Methods

- **`new()`**: Creates a new instance.
- **`set_bc(entity, condition)`**: Assigns a Robin condition to an entity.
- **`apply_bc(matrix, rhs, entity_to_index, time)`**: Applies the conditions to the system.

#### Example

```rust
let robin_bc = RobinBC::new();
robin_bc.set_bc(entity, BoundaryCondition::Robin { alpha: 2.0, beta: 3.0 });
robin_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, current_time);
```

### MixedBC

Handles Mixed boundary conditions, allowing customized combinations of parameters.

#### Structure

```rust
pub struct MixedBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

#### Methods

- **`new()`**: Creates a new instance.
- **`set_bc(entity, condition)`**: Assigns a Mixed condition to an entity.
- **`apply_bc(matrix, rhs, entity_to_index, time)`**: Applies the conditions to the system.

#### Example

```rust
let mixed_bc = MixedBC::new();
mixed_bc.set_bc(entity, BoundaryCondition::Mixed { gamma: 2.0, delta: 3.0 });
mixed_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, current_time);
```

### CauchyBC

Handles Cauchy boundary conditions, involving both the variable and its derivative.

#### Structure

```rust
pub struct CauchyBC {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

#### Methods

- **`new()`**: Creates a new instance.
- **`set_bc(entity, condition)`**: Assigns a Cauchy condition to an entity.
- **`apply_bc(matrix, rhs, entity_to_index, time)`**: Applies the conditions to the system.

#### Example

```rust
let cauchy_bc = CauchyBC::new();
cauchy_bc.set_bc(entity, BoundaryCondition::Cauchy { lambda: 1.5, mu: 2.5 });
cauchy_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, current_time);
```

---

## **9. Working with Function-Based Boundary Conditions**

Function-based boundary conditions allow for time-dependent or spatially varying conditions.

### Dirichlet Function-Based Conditions

```rust
dirichlet_bc.set_bc(
    entity,
    BoundaryCondition::DirichletFn(Arc::new(|time, coords| {
        // Define the function based on time and coordinates
        100.0 * time + coords[0]
    })),
);
```

### Neumann Function-Based Conditions

```rust
neumann_bc.set_bc(
    entity,
    BoundaryCondition::NeumannFn(Arc::new(|time, coords| {
        // Define the flux function
        50.0 * coords[1] - 10.0 * time
    })),
);
```

- **Usage**: Enables modeling of dynamic systems where boundary conditions change over time or space.

---

## **10. Testing and Validation**

### Unit Testing

Ensure that each boundary condition handler correctly stores and applies conditions.

- **Example Test for DirichletBC**:

  ```rust
  #[test]
  fn test_set_bc() {
      let dirichlet_bc = DirichletBC::new();
      let entity = MeshEntity::Vertex(1);
      dirichlet_bc.set_bc(entity, BoundaryCondition::Dirichlet(5.0));
      let condition = dirichlet_bc.conditions.get(&entity).map(|entry| entry.clone());
      assert!(matches!(condition, Some(BoundaryCondition::Dirichlet(5.0))));
  }
  ```

### Integration Testing

Test the interaction of multiple boundary conditions and their cumulative effects on the system.

- **Validate**: Stability and accuracy of the overall simulation when multiple conditions are applied.

### Debugging Tips

- **Check Assignments**: Verify that boundary conditions are assigned to the correct entities.
- **Inspect Modifications**: Examine the matrix and RHS after applying conditions to ensure they have been modified appropriately.
- **Mapping Verification**: Ensure that `entity_to_index` correctly maps entities to matrix indices.

---

## **11. Best Practices**

### Efficient Boundary Condition Management

- **Concurrent Access**: Use `DashMap` for thread-safe operations when managing boundary conditions in parallel computations.
- **Centralized Handling**: Utilize `BoundaryConditionHandler` for centralized management.

### Performance Optimization

- **Cache Function Outputs**: When using function-based conditions, cache outputs if the same values are needed multiple times.
- **Parallel Processing**: Apply conditions in parallel when dealing with large meshes to improve performance.

### Handling Complex Boundary Conditions

- **Layering Conditions**: For complex simulations, layer multiple boundary conditions strategically.
- **Custom Conditions**: Implement custom boundary conditions by extending the `BoundaryConditionApply` trait.

---

## **12. Conclusion**

The `Boundary` module in Hydra provides a flexible and robust framework for managing boundary conditions in simulations. By supporting various types of conditions and allowing for time-dependent and spatially varying specifications, it enables accurate modeling of physical systems. Proper utilization of this module is essential for ensuring that simulations produce reliable and realistic results.