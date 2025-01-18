# Hydra `Boundary` Module User Guide

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Boundary Module](#2-overview-of-the-boundary-module)  
3. [Core Structures](#3-core-structures)  
   - [BoundaryCondition Enum](#boundarycondition-enum)  
   - [BoundaryConditionFn Type & FunctionWrapper](#boundaryconditionfn-type--functionwrapper)  
4. [Boundary Condition Handler](#4-boundary-condition-handler)  
   - [BoundaryConditionHandler Struct](#boundaryconditionhandler-struct)  
   - [Global Handler Access](#global-handler-access)  
   - [Applying Boundary Conditions](#applying-boundary-conditions)  
5. [Managing Boundary Conditions](#5-managing-boundary-conditions)  
   - [Adding Boundary Conditions to Entities](#adding-boundary-conditions-to-entities)  
   - [Retrieving Boundary Conditions](#retrieving-boundary-conditions)  
6. [BoundaryConditionApply Trait](#6-boundaryconditionapply-trait)  
7. [Specific Boundary Condition Implementations](#7-specific-boundary-condition-implementations)  
   - [DirichletBC](#dirichletbc)  
   - [NeumannBC](#neumannbc)  
   - [RobinBC](#robinbc)  
   - [MixedBC](#mixedbc)  
   - [CauchyBC](#cauchybc)  
   - [SolidWallBC](#solidwallbc)  
   - [FarFieldBC](#farfieldbc)  
   - [InjectionBC](#injectionbc)  
   - [InletOutletBC](#inletoutletbc)  
   - [PeriodicBC](#periodicbc)  
   - [SymmetryBC](#symmetrybc)  
8. [Working with Function-Based Boundary Conditions](#8-working-with-function-based-boundary-conditions)  
9. [Testing and Validation](#9-testing-and-validation)  
   - [Unit Testing](#unit-testing)  
   - [Integration Testing](#integration-testing)  
10. [Best Practices](#10-best-practices)  
    - [Efficient Boundary Condition Management](#efficient-boundary-condition-management)  
    - [Performance Optimization](#performance-optimization)  
    - [Handling Complex or Multiple Conditions](#handling-complex-or-multiple-conditions)  
11. [Conclusion](#11-conclusion)

---

## **1. Introduction**

Welcome to the user guide for the **`Boundary`** module in Hydra. This module manages **boundary conditions** for numerical simulations—especially in the context of **CFD (Computational Fluid Dynamics)** or **FEM/FVM**-based solvers.  

Boundary conditions specify how the domain interacts with the “outside” environment and are critical for physical accuracy. The `Boundary` module in Hydra supports a range of condition types—from classical Dirichlet/Neumann/Robin to more specialized conditions like **solid walls**, **far-field**, **periodic**, **injection**, and more.

---

## **2. Overview of the Boundary Module**

- **Single Interface** for many boundary condition types (e.g., Dirichlet, Neumann, Robin, Mixed, Cauchy, etc.).  
- **Function-based** (time-dependent/spatial) boundary conditions using closures.  
- Integration with Hydra’s **`domain::mesh_entity::MeshEntity`** for applying conditions to specific **faces** or other entities.  
- **Global** or **local** boundary handlers to store and apply boundary conditions.  
- Uniform **application** to system matrices and RHS vectors (via `faer::MatMut`).

---

## **3. Core Structures**

### BoundaryCondition Enum

```rust
#[derive(Clone, PartialEq, Debug)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    SolidWallInviscid,
    SolidWallViscous { normal_velocity: f64 },
    DirichletFn(FunctionWrapper),
    NeumannFn(FunctionWrapper),
    Periodic { pairs: Vec<(MeshEntity, MeshEntity)> },
    FarField(f64),
    Injection(f64),
    InletOutlet,
    Symmetry,
}
```

**Key Variants**:

- **Dirichlet/DirichletFn**: Specifies a fixed value at the boundary; the `_Fn` variant uses a runtime function.  
- **Neumann/NeumannFn**: Specifies a flux at the boundary; the `_Fn` variant uses a runtime function.  
- **Robin**, **Mixed**, **Cauchy**: Linear combinations or specialized forms using parameters like `alpha, beta`, etc.  
- **SolidWallInviscid / SolidWallViscous**: For no-penetration or no-slip walls.  
- **FarField**: Emulates infinite domain boundaries.  
- **Injection**: Inject mass/momentum/energy at the boundary.  
- **InletOutlet**: Basic inflow/outflow combination.  
- **Periodic**: Pairs of entities that share the same solution DOF.  
- **Symmetry**: Zero normal velocity or flux across a plane.

### BoundaryConditionFn Type & FunctionWrapper

```rust
pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

#[derive(Clone)]
pub struct FunctionWrapper {
    pub description: String, 
    pub function: BoundaryConditionFn,
}
```

- **`BoundaryConditionFn`**: A thread-safe closure type accepting `(time, coordinates) -> boundary_value`.  
- **`FunctionWrapper`**: Stores additional metadata like a `description` to help with logging or debugging.

---

## **4. Boundary Condition Handler**

### BoundaryConditionHandler Struct

```rust
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}
```

- **Purpose**: Central storage for boundary conditions, keyed by `MeshEntity`.  
- Internally uses `DashMap` for thread-safe read/write.

### Global Handler Access

```rust
lazy_static! {
    static ref GLOBAL_BC_HANDLER: Arc<RwLock<BoundaryConditionHandler>> =
        Arc::new(RwLock::new(BoundaryConditionHandler::new()));
}

pub fn global() -> Arc<RwLock<BoundaryConditionHandler>> {
    GLOBAL_BC_HANDLER.clone()
}
```

- **`BoundaryConditionHandler::global()`**: Provides a **global** singleton for boundary conditions if desired.

### Applying Boundary Conditions

```rust
pub fn apply_bc(
    &self,
    matrix: &mut MatMut<f64>,
    rhs: &mut MatMut<f64>,
    boundary_entities: &[MeshEntity],
    entity_to_index: &DashMap<MeshEntity, usize>,
    time: f64,
)
```

- Iterates over the specified `boundary_entities`.  
- For each entity with a known condition, it delegates to the appropriate boundary condition logic (e.g., `DirichletBC`, `NeumannBC`, etc.).  
- Modifies the **system matrix** (`matrix`) and **RHS vector** (`rhs`) accordingly.  
- Example usage:

  ```rust
  let bc_handler = BoundaryConditionHandler::new();
  let boundary_entities = vec![MeshEntity::Face(10)];
  bc_handler.apply_bc(&mut matrix, &mut rhs, &boundary_entities, &entity_to_index, current_time);
  ```

---

## **5. Managing Boundary Conditions**

### Adding Boundary Conditions to Entities

```rust
let bc_handler = BoundaryConditionHandler::new();
let face = MeshEntity::Face(10);

bc_handler.set_bc(face, BoundaryCondition::Dirichlet(1.0));
```

- **`set_bc(entity, condition)`**: Assigns or overwrites the boundary condition on `entity`.

### Retrieving Boundary Conditions

```rust
if let Some(cond) = bc_handler.get_bc(&face) {
    println!("Boundary condition is: {:?}", cond);
}
```

- **`get_bc(entity)`**: Returns the assigned boundary condition, if any.

---

## **6. BoundaryConditionApply Trait**

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

- Implemented by each boundary condition struct.  
- Allows a uniform **`apply(...)`** call that modifies the system matrix and RHS.

**Note**: The enum `BoundaryCondition` itself also implements this trait, delegating to the specialized boundary condition logic.

---

## **7. Specific Boundary Condition Implementations**

Below are the main boundary condition structs. Each maintains an internal `DashMap` that can be populated with per-entity boundary conditions. Alternatively, you can rely on the `BoundaryConditionHandler` which dispatches to them automatically.

### DirichletBC

**File**: `dirichlet.rs`  
- **Enforces** a fixed value at the boundary (constant or function-based).  
- **Key Methods**:
  - `apply_constant_dirichlet(matrix, rhs, index, value)`
  - `apply_bc(...)` for all stored Dirichlet conditions.
- **System Effect**: Zeros out row in matrix, sets diagonal to 1, and sets RHS to the Dirichlet value.

Example:
```rust
let dirichlet_bc = DirichletBC::new();
dirichlet_bc.set_bc(face, BoundaryCondition::Dirichlet(5.0));
// Then apply
dirichlet_bc.apply_bc(&mut matrix, &mut rhs, &entity_to_index, 0.0);
```

### NeumannBC

**File**: `neumann.rs`  
- **Specifies** flux at the boundary.  
- **Key Methods**:
  - `apply_constant_neumann(rhs, index, flux)`
  - `apply_bc(...)` for all stored conditions.
- **System Effect**: Adds flux to the RHS; the matrix row typically remains unchanged.

Example:
```rust
let neumann_bc = NeumannBC::new();
neumann_bc.set_bc(face, BoundaryCondition::Neumann(2.0));
neumann_bc.apply_bc(&mut rhs, &entity_to_index, time);
```

### RobinBC

**File**: `robin.rs`  
- **Combines** Dirichlet and Neumann in the form `alpha*u + beta*(du/dn) = something`.  
- **System Effect**: Modifies the diagonal by `alpha`; adds `beta` to RHS.  
- **Usage**:
  ```rust
  robin_bc.apply_robin(matrix, rhs, index, alpha, beta);
  ```

### MixedBC

**File**: `mixed.rs`  
- A **hybrid** or generalized BC with parameters `gamma, delta`.  
- **System Effect**: Adds `gamma` to the matrix diagonal and `delta` to the RHS at the boundary row.

### CauchyBC

**File**: `cauchy.rs`  
- Typically used in **fluid-structure** or other PDE contexts. Involves both value and derivative via `lambda` and `mu`.  
- **System Effect**: Increases diagonal by `lambda`, adds `mu` to RHS.

### SolidWallBC

**File**: `solid_wall.rs`  
- Encompasses **inviscid** (`SolidWallInviscid`) and **viscous** (`SolidWallViscous`) conditions.  
- **Inviscid**: Enforces no flow normal to the wall; sets the boundary row diagonal to 1 with zero RHS.  
- **Viscous**: Also sets velocity normal to zero but may impose a user-specified `normal_velocity` in RHS.

### FarFieldBC

**File**: `far_field.rs`  
- Ideal for **far-field** boundaries that emulate “infinite domain.”  
- Typically sets the boundary row to a known state or vacuum.  
- May also handle Dirichlet or Neumann sub-conditions.

### InjectionBC

**File**: `injection.rs`  
- Models injecting fluid or property at a boundary.  
- If Dirichlet, enforces a fixed state; if Neumann, adds flux.  
- **System Effect**: Zeros row if Dirichlet, modifies RHS if flux.

### InletOutletBC

**File**: `inlet_outlet.rs`  
- Combines various conditions for a “mixed” inlet/outlet scenario.  
- **Key Methods**: 
  - `apply_dirichlet(matrix, rhs, index, value)`
  - `apply_neumann(rhs, index, flux)`
  - `apply_robin(matrix, rhs, index, alpha, beta)`

### PeriodicBC

**File**: `periodic.rs`  
- Maintains a **mapping** of pairs of entities that share the same DOF.  
- **System Effect**: Averages matrix row/column entries (and RHS values) across paired indices, forcing them to be equal.

### SymmetryBC

**File**: `symmetry.rs`  
- Zeroes out normal velocity/flux across a plane of symmetry.  
- Implementation is very similar to an inviscid wall but contextually for symmetrical planes.

---

## **8. Working with Function-Based Boundary Conditions**

- **FunctionWrapper** allows storing function closures with a descriptive label.  
- **DirichletFn** or **NeumannFn** accept a `(time, coords) -> f64` function:
  ```rust
  use std::sync::Arc;
  use crate::boundary::bc_handler::{FunctionWrapper, BoundaryConditionFn};

  let func = Arc::new(|time: f64, coords: &[f64]| -> f64 {
      // e.g., a wave-like boundary
      time.sin() * coords[0]
  });

  let wrapper = FunctionWrapper {
      description: String::from("Wave BC"),
      function: func,
  };

  bc_handler.set_bc(
      face,
      BoundaryCondition::DirichletFn(wrapper)
  );
  ```
- During `apply_bc(...)`, the code calls your function with `time` and (placeholder) `coords = [0.0,0.0,0.0]`.

---

## **9. Testing and Validation**

### Unit Testing

- Validate **individual** boundary conditions. For instance, `DirichletBC`:
  ```rust
  #[test]
  fn test_dirichlet_bc() {
      let dirichlet_bc = DirichletBC::new();
      let face = MeshEntity::Face(1);
      dirichlet_bc.set_bc(face, BoundaryCondition::Dirichlet(10.0));
      // Prepare a small test matrix & vector with known dimensions
      // ...
      // then check if the row/column is updated as expected
  }
  ```

- Ensure each BC modifies the matrix and RHS in the intended manner.

### Integration Testing

- Combine multiple boundary types on different faces.  
- Verify that a solver or time-step loop reads these conditions accurately.  
- **Check** if conditions remain consistent during partitioning or reordering in the Hydra `domain`.

---

## **10. Best Practices**

### Efficient Boundary Condition Management

1. **Use `DashMap`** for concurrency: The `BoundaryConditionHandler` is thread-safe.  
2. **Maintain a Single Source**: Keep boundary conditions in either a single `BoundaryConditionHandler` or distributed in specialized BC structs, but be consistent.

### Performance Optimization

1. **Apply in Parallel**: If you have thousands of faces, consider parallel iteration over boundary entities.  
2. **Function Caching**: If the same function-based BC is called frequently with the same `(time, coords)`, consider caching.

### Handling Complex or Multiple Conditions

1. **Composite BC**: If a face has multiple constraints, prefer combining them into a custom boundary type or stage the matrix modifications carefully.  
2. **Periodic**: Double-check index mapping for paired faces/cells.  
3. **Domain Overlap**: In multi-partition scenarios, align your boundary conditions with ghost entities if needed.

---

## **11. Conclusion**

The Hydra `Boundary` module enables a **rich** set of boundary conditions—both **traditional** (Dirichlet, Neumann, Robin) and more **specialized** (solid walls, far field, injection, etc.). Its architecture is flexible enough to handle function-based, time-dependent BCs, as well as advanced setups like periodic or mixed boundaries.

By combining the `BoundaryConditionHandler` with Hydra’s **`domain`** module (for entity and matrix indexing), you can accurately impose the physical constraints of your simulation domain. Remember to test and validate each boundary type in isolation and in integration to ensure physical fidelity.

Use these components in conjunction with Hydra’s solver pipeline for a robust, efficient, and feature-complete boundary condition workflow.