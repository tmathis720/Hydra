# Hydra `Domain` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Domain Module](#2-overview-of-the-domain-module)
3. [Core Structures](#3-core-structures)
   - [MeshEntity Enum](#meshentity-enum)
   - [Arrow Struct](#arrow-struct)
4. [Mesh Connectivity with Sieve](#4-mesh-connectivity-with-sieve)
   - [Sieve Structure](#sieve-structure)
   - [Core Methods](#core-methods)
5. [Stratification of Mesh Entities](#5-stratification-of-mesh-entities)
6. [Filling Missing Entities](#6-filling-missing-entities)
7. [Data Association with Section](#7-data-association-with-section)
   - [Section Struct](#section-struct)
   - [Parallel Data Updates](#parallel-data-updates)
8. [Domain Overlap Management](#8-domain-overlap-management)
   - [Overlap Struct](#overlap-struct)
   - [Delta Struct](#delta-struct)
9. [Mesh Management](#9-mesh-management)
   - [Mesh Struct](#mesh-struct)
   - [Entities Management](#entities-management)
   - [Boundary Handling](#boundary-handling)
   - [Geometry Calculations](#geometry-calculations)
   - [Hierarchical Mesh](#hierarchical-mesh)
   - [Topology Validation](#topology-validation)
   - [Reordering Algorithms](#reordering-algorithms)
10. [Testing and Validation](#10-testing-and-validation)
    - [Unit Testing](#unit-testing)
    - [Integration Testing](#integration-testing)
11. [Best Practices](#11-best-practices)
    - [Efficient Mesh Management](#efficient-mesh-management)
    - [Performance Optimization](#performance-optimization)
    - [Handling Complex Mesh Structures](#handling-complex-mesh-structures)
12. [Advanced Configurations and Extensibility](#12-advanced-configurations-and-extensibility)
13. [Conclusion](#13-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Domain` module of the Hydra computational framework. This module is fundamental for managing mesh-related data and operations essential for finite volume method (FVM) simulations. It provides robust structures for representing mesh entities—such as vertices, edges, faces, and cells—and their relationships.

---

## **2. Overview of the Domain Module**

The `Domain` module is designed to handle complex, unstructured 2D and 3D meshes commonly used in computational fluid dynamics (CFD) and geophysical simulations. It offers:

- **Mesh Entity Representation**: Defines core elements like vertices, edges, faces, and cells.
- **Mesh Connectivity Management**: Establishes and manages relationships between mesh entities using the `Sieve` data structure.
- **Data Association**: Associates data with mesh entities through the `Section` struct.
- **Domain Overlap and Boundary Data Communication**: Manages domain overlap and facilitates boundary data communication in partitioned meshes using `Overlap` and `Delta`.
- **Hierarchical Mesh Support**: Enables adaptive mesh refinement via hierarchical mesh nodes.
- **Topology Validation and Reordering**: Validates mesh topology and implements reordering algorithms to optimize memory access patterns.

These capabilities ensure efficient and consistent domain-specific operations across different levels of mesh granularity within Hydra.

---

## **3. Core Structures**

### MeshEntity Enum

The `MeshEntity` enum represents different types of elements within a mesh:

```rust
pub enum MeshEntity {
    Vertex(usize),
    Edge(usize),
    Face(usize),
    Cell(usize),
}
```

- **Methods**:
  - `get_id()`: Retrieves the unique identifier of the entity.
  - `get_entity_type()`: Returns the type of the entity as a string.
  - `with_id(new_id)`: Creates a new `MeshEntity` with a specified identifier.

**Example Usage**:

```rust
let vertex = MeshEntity::Vertex(1);
println!("Vertex ID: {}", vertex.get_id()); // Output: Vertex ID: 1
```

### Arrow Struct

The `Arrow` struct represents directed relationships between two `MeshEntity` instances:

```rust
pub struct Arrow {
    pub from: MeshEntity,
    pub to: MeshEntity,
}
```

- **Methods**:
  - `new(from, to)`: Constructs a new `Arrow`.
  - `get_relation()`: Retrieves the `from` and `to` entities.

**Example Usage**:

```rust
let from = MeshEntity::Vertex(1);
let to = MeshEntity::Edge(2);
let arrow = Arrow::new(from, to);
let (start, end) = arrow.get_relation();
```

---

## **4. Mesh Connectivity with Sieve**

### Sieve Structure

The `Sieve` structure manages relationships among mesh entities using a directed graph format:

```rust
pub struct Sieve {
    pub adjacency: DashMap<MeshEntity, FxHashSet<MeshEntity>>,
}
```

- **Adjacency Map**: A thread-safe map where each key is a `MeshEntity`, and the value is a set of related entities.

### Core Methods

- **Creating a Sieve**:

  ```rust
  let sieve = Sieve::new();
  ```

- **Adding Relationships**:

  ```rust
  sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));
  ```

- **Querying Relationships**:

  - **Cone**: Retrieves entities directly related to a given entity.

    ```rust
    let related = sieve.cone(&MeshEntity::Vertex(1)).unwrap();
    ```

  - **Closure**: Retrieves all entities connected to a given entity, recursively.

    ```rust
    let closure = sieve.closure(&MeshEntity::Vertex(1));
    ```

  - **Star**: Retrieves the entity and all entities that directly cover it.

    ```rust
    let star = sieve.star(&MeshEntity::Vertex(1));
    ```

- **Parallel Processing**:

  ```rust
  sieve.par_for_each_adjacent(|entity, adjacents| {
      // Process adjacency in parallel
  });
  ```

---

## **5. Stratification of Mesh Entities**

The `stratify` method organizes mesh entities into strata based on their dimensions:

- **Stratum 0**: Vertices
- **Stratum 1**: Edges
- **Stratum 2**: Faces
- **Stratum 3**: Cells

**Usage**:

```rust
let strata = sieve.stratify();
for (dimension, entities) in strata {
    println!("Dimension {}: {} entities", dimension, entities.len());
}
```

---

## **6. Filling Missing Entities**

The `fill_missing_entities` method infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.

**Process**:

- For each cell, the method generates all combinations of its vertices to create edges or faces that may be missing.

**Usage**:

```rust
sieve.fill_missing_entities();
```

- **Note**: This operation is essential for ensuring that the mesh is fully connected and suitable for simulation.

---

## **7. Data Association with Section**

### Section Struct

The `Section<T>` struct associates values of a specified type `T` with individual `MeshEntity` elements.

```rust
pub struct Section<T> {
    data: DashMap<MeshEntity, T>,
}
```

- **Methods**:
  - `new()`: Creates a new `Section`.
  - `set_data(entity, value)`: Associates data with an entity.
  - `restrict(entity)`: Retrieves data associated with an entity.
  - `update_data(entity, new_value)`: Updates data for an entity.
  - `clear()`: Clears all data from the section.
  - `entities()`: Retrieves all entities with associated data.

**Example Usage**:

```rust
let mut section = Section::new();
section.set_data(MeshEntity::Vertex(1), 100.0);
if let Some(value) = section.restrict(&MeshEntity::Vertex(1)) {
    println!("Data at Vertex 1: {}", value);
}
```

### Parallel Data Updates

The `parallel_update` method applies a function to all data entries in parallel:

```rust
section.parallel_update(|value| {
    *value *= 2.0;
});
```

---

## **8. Domain Overlap Management**

### Overlap Struct

Manages local and ghost entities in partitioned domains:

```rust
pub struct Overlap {
    local_entities: DashSet<MeshEntity>,
    ghost_entities: DashSet<MeshEntity>,
}
```

- **Methods**:
  - `add_local_entity(entity)`: Adds an entity to the local set.
  - `add_ghost_entity(entity)`: Adds an entity to the ghost set.
  - `is_local(entity)`: Checks if an entity is local.
  - `is_ghost(entity)`: Checks if an entity is a ghost.

**Example Usage**:

```rust
let mut overlap = Overlap::new();
overlap.add_local_entity(MeshEntity::Cell(1));
overlap.add_ghost_entity(MeshEntity::Cell(2));
```

### Delta Struct

Manages transformation data for entities in overlapping regions:

```rust
pub struct Delta<T> {
    data: DashMap<MeshEntity, T>,
}
```

- **Methods**:
  - `set_data(entity, value)`: Associates transformation data with an entity.
  - `get_data(entity)`: Retrieves transformation data for an entity.

**Example Usage**:

```rust
let mut delta = Delta::new();
delta.set_data(MeshEntity::Vertex(1), [1.0, 0.0, 0.0]);
```

---

## **9. Mesh Management**

### Mesh Struct

The `Mesh` struct is the central data structure for managing the domain:

```rust
pub struct Mesh {
    pub sieve: Arc<Sieve>,
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,
    pub boundary_data_sender: Option<Sender<BoundaryData>>,
    pub boundary_data_receiver: Option<Receiver<BoundaryData>>,
}
```

### Entities Management

- **Adding Entities**:

  ```rust
  mesh.add_entity(MeshEntity::Vertex(1));
  ```

- **Establishing Relationships**:

  ```rust
  mesh.add_relationship(MeshEntity::Vertex(1), MeshEntity::Edge(1));
  ```

- **Setting Vertex Coordinates**:

  ```rust
  mesh.set_vertex_coordinates(1, [0.0, 1.0, 2.0]);
  ```

- **Retrieving Entities**:

  ```rust
  let cells = mesh.get_cells();
  ```

### Boundary Handling

- **Setting Boundary Channels**:

  ```rust
  mesh.set_boundary_channels(sender, receiver);
  ```

- **Synchronizing Boundary Data**:

  ```rust
  mesh.sync_boundary_data();
  ```

### Geometry Calculations

- **Getting Face Area**:

  ```rust
  let area = mesh.get_face_area(&face);
  ```

- **Computing Cell Centroid**:

  ```rust
  let centroid = mesh.get_cell_centroid(&cell);
  ```

- **Distance Between Cells**:

  ```rust
  let distance = mesh.get_distance_between_cells(&cell1, &cell2);
  ```

### Hierarchical Mesh

Supports adaptive mesh refinement:

- **MeshNode Enum**:

  ```rust
  pub enum MeshNode<T> {
      Leaf(T),
      Branch {
          data: T,
          children: Box<[MeshNode<T>; 4]>,
      },
  }
  ```

- **Methods**:
  - `refine(init_child_data)`: Refines a leaf node into a branch.
  - `coarsen()`: Coarsens a branch back into a leaf node.
  - `leaf_iter()`: Iterates over all leaf nodes.

**Example Usage**:

```rust
let mut node = MeshNode::Leaf(10);
node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
```

### Topology Validation

Validates the structural integrity and connectivity of the mesh:

```rust
pub struct TopologyValidation<'a> {
    mesh: &'a Mesh,
}
```

- **Methods**:
  - `validate_connectivity()`: Validates cell-to-face and face-to-vertex connections.
  - `validate_unique_relationships()`: Ensures unique edges within cells.

**Example Usage**:

```rust
let topology_validation = TopologyValidation::new(&mesh);
assert!(topology_validation.validate_connectivity());
```

### Reordering Algorithms

Reordering improves computational efficiency:

- **Cuthill-McKee Algorithm**: Reduces matrix bandwidth.

  ```rust
  let ordering = mesh.rcm_ordering(start_node);
  mesh.apply_reordering(&ordering);
  ```

- **Morton Order (Z-order Curve)**: Preserves spatial locality.

  ```rust
  mesh.reorder_by_morton_order(&mut elements);
  ```

---

## **10. Testing and Validation**

### Unit Testing

- **Mesh Entity Tests**: Verify creation and manipulation of `MeshEntity` instances.
- **Sieve Tests**: Validate relationship management and query methods.
- **Section Tests**: Ensure data association and parallel updates function correctly.

**Example Test**:

```rust
#[test]
fn test_mesh_entity() {
    let vertex = MeshEntity::Vertex(1);
    assert_eq!(vertex.get_id(), 1);
}
```

### Integration Testing

- **Mesh Construction Tests**: Build a mesh and verify its integrity.
- **Boundary Data Synchronization Tests**: Ensure boundary data is correctly communicated between partitions.
- **Reordering Tests**: Confirm that reordering algorithms improve performance.

---

## **11. Best Practices**

### Efficient Mesh Management

- **Use Thread-Safe Structures**: Utilize `DashMap` and `DashSet` for concurrent operations.
- **Avoid Redundant Entities**: Ensure that entities are not duplicated within the mesh.
- **Validate Mesh Regularly**: Use `TopologyValidation` methods to check mesh integrity after modifications.

### Performance Optimization

- **Parallel Processing**: Leverage parallel methods in `Sieve` and `Section` for performance gains.
- **Reordering**: Apply reordering algorithms to improve memory locality and solver efficiency.
- **Adaptive Refinement**: Use hierarchical meshes to focus computational resources where needed.

### Handling Complex Mesh Structures

- **Hierarchical Meshes**: Implement adaptive mesh refinement for simulations requiring varying resolutions.
- **Boundary Conditions**: Properly manage boundary entities and data synchronization in partitioned meshes.
- **Data Association**: Use `Section` to efficiently associate and update data with mesh entities.

---

## **12. Advanced Configurations and Extensibility**

- **Custom Mesh Entities**: Extend `MeshEntity` to include new types if needed.
- **Custom Topology Validation**: Implement additional validation rules in `TopologyValidation`.
- **Integration with External Data Sources**: Extend boundary handling to incorporate real-time data inputs.

---

## **13. Conclusion**

The `Domain` module in Hydra provides a comprehensive framework for managing mesh entities and their relationships, essential for accurate and efficient FVM simulations. By utilizing its robust structures and methods, users can construct, manipulate, and validate complex meshes, ensuring that simulations are both accurate and performant.

---

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

---

# Hydra `Linear Algebra` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Linear Algebra Module](#2-overview-of-the-linear-algebra-module)
3. [Core Components](#3-core-components)
   - [Vectors](#vectors)
   - [Matrices](#matrices)
4. [Vector Module](#4-vector-module)
   - [Vector Traits](#vector-traits)
   - [Vector Implementations](#vector-implementations)
     - [Implementation for `Vec<f64>`](#implementation-for-vecf64)
     - [Implementation for `Mat<f64>`](#implementation-for-matf64)
   - [Vector Builder](#vector-builder)
   - [Vector Testing](#vector-testing)
5. [Matrix Module](#5-matrix-module)
   - [Matrix Traits](#matrix-traits)
   - [Matrix Implementations](#matrix-implementations)
     - [Implementation for `Mat<f64>`](#implementation-for-matf64-1)
   - [Matrix Builder](#matrix-builder)
   - [Matrix Testing](#matrix-testing)
6. [Using the Linear Algebra Module](#6-using-the-linear-algebra-module)
   - [Creating Vectors](#creating-vectors)
   - [Performing Vector Operations](#performing-vector-operations)
   - [Creating Matrices](#creating-matrices)
   - [Performing Matrix Operations](#performing-matrix-operations)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Linear Algebra` module of the Hydra computational framework. This module provides essential linear algebra functionalities, including vector and matrix operations, which are fundamental for numerical simulations and computational methods used in finite volume methods (FVM) and computational fluid dynamics (CFD).

---

## **2. Overview of the Linear Algebra Module**

The `Linear Algebra` module in Hydra is designed to offer:

- **Abstract Traits**: Define common interfaces for vectors and matrices, allowing for flexible implementations.
- **Implementations**: Provide concrete implementations for standard data structures such as `Vec<f64>` and `Mat<f64>`.
- **Builders**: Facilitate the construction and manipulation of vectors and matrices.
- **Operations**: Support essential linear algebra operations like dot products, norms, matrix-vector multiplication, etc.
- **Testing**: Ensure reliability through comprehensive unit tests.

This modular design allows users to integrate various underlying data structures and optimize for performance and memory usage.

---

## **3. Core Components**

### Vectors

- **Traits**: Define the `Vector` trait, which includes methods for vector operations.
- **Implementations**: Provide implementations for common vector types, such as Rust's `Vec<f64>` and the `Mat<f64>` type from the `faer` library.
- **Operations**: Include methods for dot product, scaling, addition, element-wise operations, cross product, and statistical functions.

### Matrices

- **Traits**: Define the `Matrix` trait, which includes methods for matrix operations.
- **Implementations**: Provide implementations for matrix types, particularly `Mat<f64>` from the `faer` library.
- **Operations**: Include methods for matrix-vector multiplication, trace, Frobenius norm, and access to elements.

---

## **4. Vector Module**

The vector module is organized into several components:

- **Traits** (`traits.rs`)
- **Implementations** (`vec_impl.rs` and `mat_impl.rs`)
- **Vector Builder** (`vector_builder.rs`)
- **Testing** (`tests.rs`)

### Vector Traits

The `Vector` trait defines a set of common operations for vectors:

```rust
pub trait Vector: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Self::Scalar;
    fn set(&mut self, i: usize, value: Self::Scalar);
    fn as_slice(&self) -> &[Self::Scalar];
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;
    fn norm(&self) -> Self::Scalar;
    fn scale(&mut self, scalar: Self::Scalar);
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn cross(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str>;
    fn sum(&self) -> Self::Scalar;
    fn max(&self) -> Self::Scalar;
    fn min(&self) -> Self::Scalar;
    fn mean(&self) -> Self::Scalar;
    fn variance(&self) -> Self::Scalar;
}
```

- **Thread Safety**: All implementations must be `Send` and `Sync`.
- **Scalar Type**: The `Scalar` associated type allows for flexibility in the numeric type used.

### Vector Implementations

#### Implementation for `Vec<f64>`

The standard Rust `Vec<f64>` type implements the `Vector` trait, providing methods for vector operations.

- **Key Methods**:
  - `len()`: Returns the length of the vector.
  - `get(i)`: Retrieves the element at index `i`.
  - `set(i, value)`: Sets the element at index `i` to `value`.
  - `dot(other)`: Computes the dot product with another vector.
  - `norm()`: Calculates the Euclidean norm.
  - `scale(scalar)`: Scales the vector by a scalar.
  - `axpy(a, x)`: Performs the operation `self = a * x + self`.
  - `cross(other)`: Computes the cross product (only for 3D vectors).

**Example Usage**:

```rust
let mut vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];

// Dot product
let dot = vec1.dot(&vec2);

// Scaling
vec1.scale(2.0);

// Element-wise addition
vec1.element_wise_add(&vec2);
```

#### Implementation for `Mat<f64>`

The `Mat<f64>` type from the `faer` library is used to represent column vectors and implements the `Vector` trait.

- **Key Methods**:
  - `len()`: Returns the number of rows (since it's a column vector).
  - `get(i)`: Retrieves the element at row `i`.
  - `set(i, value)`: Sets the element at row `i` to `value`.
  - Supports all other methods defined in the `Vector` trait.

**Example Usage**:

```rust
use faer::Mat;

// Creating a column vector with 3 elements
let mut mat_vec = Mat::<f64>::zeros(3, 1);
mat_vec.set(0, 1.0);
mat_vec.set(1, 2.0);
mat_vec.set(2, 3.0);

// Computing the norm
let norm = mat_vec.norm();
```

### Vector Builder

The `VectorBuilder` struct provides methods to build and manipulate vectors generically.

- **Methods**:
  - `build_vector(size)`: Constructs a vector of a specified type and size.
  - `build_dense_vector(size)`: Constructs a dense vector using `Mat<f64>`.
  - `resize_vector(vector, new_size)`: Resizes a vector while maintaining memory safety.

**Example Usage**:

```rust
let size = 5;
let vector = VectorBuilder::build_vector::<Vec<f64>>(size);

// Resizing the vector
VectorBuilder::resize_vector(&mut vector, 10);
```

### Vector Testing

Comprehensive tests are provided to ensure the correctness of vector operations.

- **Test Cases**:
  - Length retrieval
  - Element access and modification
  - Dot product calculation
  - Norm computation
  - Scaling and axpy operations
  - Element-wise addition, multiplication, and division
  - Cross product
  - Statistical functions: sum, max, min, mean, variance

---

## **5. Matrix Module**

The matrix module includes:

- **Traits** (`traits.rs`)
- **Implementations** (`mat_impl.rs`)
- **Matrix Builder** (`matrix_builder.rs`)
- **Testing** (`tests.rs`)

### Matrix Traits

The `Matrix` trait defines essential matrix operations:

```rust
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>);
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    fn trace(&self) -> Self::Scalar;
    fn frobenius_norm(&self) -> Self::Scalar;
    fn as_slice(&self) -> Box<[Self::Scalar]>;
    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}
```

Additional traits for matrix construction and manipulation:

- **`MatrixOperations`**: For constructing and accessing matrix elements.
- **`ExtendedMatrixOperations`**: For resizing matrices.

### Matrix Implementations

#### Implementation for `Mat<f64>`

The `Mat<f64>` type from the `faer` library implements the `Matrix` trait.

- **Key Methods**:
  - `nrows()`: Returns the number of rows.
  - `ncols()`: Returns the number of columns.
  - `mat_vec(x, y)`: Performs matrix-vector multiplication.
  - `get(i, j)`: Retrieves the element at row `i`, column `j`.
  - `set(i, j, value)`: Sets the element at row `i`, column `j` to `value`.
  - `trace()`: Calculates the trace of the matrix.
  - `frobenius_norm()`: Computes the Frobenius norm.

**Example Usage**:

```rust
use faer::Mat;

// Creating a 3x3 zero matrix
let mut matrix = Mat::<f64>::zeros(3, 3);

// Setting elements
matrix.set(0, 0, 1.0);
matrix.set(1, 1, 1.0);
matrix.set(2, 2, 1.0);

// Matrix-vector multiplication
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y);
```

### Matrix Builder

The `MatrixBuilder` struct provides methods to build and manipulate matrices generically.

- **Methods**:
  - `build_matrix(rows, cols)`: Constructs a matrix of a specified type and dimensions.
  - `build_dense_matrix(rows, cols)`: Constructs a dense matrix using `Mat<f64>`.
  - `resize_matrix(matrix, new_rows, new_cols)`: Resizes a matrix while maintaining memory safety.
  - `apply_preconditioner(preconditioner, matrix)`: Demonstrates compatibility with preconditioners.

**Example Usage**:

```rust
let rows = 4;
let cols = 4;
let matrix = MatrixBuilder::build_matrix::<Mat<f64>>(rows, cols);

// Resizing the matrix
MatrixBuilder::resize_matrix(&mut matrix, 5, 5);
```

### Matrix Testing

Comprehensive tests are provided to ensure the correctness of matrix operations.

- **Test Cases**:
  - Dimension retrieval
  - Element access and modification
  - Matrix-vector multiplication with different vector types
  - Trace and Frobenius norm calculations
  - Thread safety
  - Handling of edge cases (e.g., out-of-bounds access)

---

## **6. Using the Linear Algebra Module**

This section provides practical examples of how to use the `Linear Algebra` module in Hydra.

### Creating Vectors

**Using `Vec<f64>`**:

```rust
let mut vector = vec![0.0; 5]; // Creates a vector of length 5 initialized with zeros.
vector.set(0, 1.0); // Sets the first element to 1.0.
```

**Using `Mat<f64>` from `faer`**:

```rust
use faer::Mat;

let mut mat_vector = Mat::<f64>::zeros(5, 1); // Creates a column vector with 5 rows.
mat_vector.set(0, 1.0); // Sets the first element to 1.0.
```

### Performing Vector Operations

**Dot Product**:

```rust
let vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];
let dot = vec1.dot(&vec2); // Computes the dot product.
```

**Norm Calculation**:

```rust
let norm = vec1.norm(); // Calculates the Euclidean norm of vec1.
```

**Scaling and AXPY Operation**:

```rust
vec1.scale(2.0); // Scales vec1 by 2.0.
vec1.axpy(1.5, &vec2); // Performs vec1 = 1.5 * vec2 + vec1.
```

**Element-wise Operations**:

```rust
vec1.element_wise_add(&vec2); // Adds vec2 to vec1 element-wise.
vec1.element_wise_mul(&vec2); // Multiplies vec1 by vec2 element-wise.
```

### Creating Matrices

**Using `Mat<f64>`**:

```rust
use faer::Mat;

// Creating a 3x3 zero matrix
let mut matrix = Mat::<f64>::zeros(3, 3);

// Setting elements
matrix.set(0, 0, 1.0);
matrix.set(1, 1, 2.0);
matrix.set(2, 2, 3.0);
```

### Performing Matrix Operations

**Matrix-Vector Multiplication**:

```rust
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y); // Computes y = matrix * x.
```

**Trace and Norm Calculations**:

```rust
let trace = matrix.trace(); // Calculates the trace of the matrix.
let fro_norm = matrix.frobenius_norm(); // Calculates the Frobenius norm.
```

---

## **7. Best Practices**

- **Thread Safety**: Ensure that vectors and matrices used across threads implement `Send` and `Sync`.
- **Consistent Dimensions**: Always verify that vector and matrix dimensions are compatible for operations like multiplication and addition.
- **Error Handling**: Handle potential errors, such as out-of-bounds access or invalid dimensions for operations (e.g., cross product requires 3D vectors).
- **Performance Optimization**: Utilize efficient data structures and avoid unnecessary copies by using slices and references where appropriate.
- **Testing**: Incorporate unit tests to verify the correctness of custom implementations or extensions.

---

## **8. Conclusion**

The `Linear Algebra` module in Hydra provides a flexible and robust framework for vector and matrix operations essential in computational simulations. By defining abstract traits and providing concrete implementations, it allows for extensibility and optimization based on specific needs. Proper utilization of this module ensures that numerical computations are accurate, efficient, and maintainable.

---

# Hydra `Geometry` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Geometry Module](#2-overview-of-the-geometry-module)
3. [Core Structures](#3-core-structures)
   - [Geometry Struct](#geometry-struct)
   - [GeometryCache Struct](#geometrycache-struct)
   - [CellShape Enum](#cellshape-enum)
   - [FaceShape Enum](#faceshape-enum)
4. [Mesh Entities and Connectivity](#4-mesh-entities-and-connectivity)
5. [Working with Vertices](#5-working-with-vertices)
   - [Adding and Updating Vertices](#adding-and-updating-vertices)
6. [Computing Geometric Properties](#6-computing-geometric-properties)
   - [Cell Centroids](#cell-centroids)
   - [Cell Volumes](#cell-volumes)
   - [Face Areas](#face-areas)
   - [Face Centroids](#face-centroids)
   - [Face Normals](#face-normals)
   - [Distance Calculations](#distance-calculations)
7. [Shape-Specific Computations](#7-shape-specific-computations)
   - [Triangles](#triangles)
   - [Quadrilaterals](#quadrilaterals)
   - [Tetrahedrons](#tetrahedrons)
   - [Hexahedrons](#hexahedrons)
   - [Prisms](#prisms)
   - [Pyramids](#pyramids)
8. [Caching and Performance Optimization](#8-caching-and-performance-optimization)
9. [Advanced Usage](#9-advanced-usage)
   - [Updating All Cell Volumes](#updating-all-cell-volumes)
   - [Computing Total Volume and Centroid](#computing-total-volume-and-centroid)
10. [Best Practices](#10-best-practices)
11. [Testing and Validation](#11-testing-and-validation)
12. [Conclusion](#12-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Geometry` module of the Hydra computational framework. This module is integral for managing geometric data and performing computations related to mesh entities within Hydra. It provides tools for calculating geometric properties such as centroids, volumes, areas, and normals for various 2D and 3D shapes.

---

## **2. Overview of the Geometry Module**

The `Geometry` module is designed to handle geometric computations for finite element and finite volume meshes. It supports both 2D and 3D shapes, including:

- **2D Shapes**: Triangles and Quadrilaterals.
- **3D Shapes**: Tetrahedrons, Hexahedrons, Prisms, and Pyramids.

The module provides functionalities to:

- Store and manage vertex coordinates.
- Compute centroids, volumes, areas, and normals of mesh entities.
- Cache computed properties for performance optimization.

---

## **3. Core Structures**

### Geometry Struct

The `Geometry` struct is the central component of the module. It holds geometric data and provides methods for computations.

```rust
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>,  // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,         // Volumes of each cell
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>, // Cache for computed properties
}
```

### GeometryCache Struct

The `GeometryCache` struct stores computed properties of geometric entities to avoid redundant calculations.

```rust
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,  // Precomputed normal vector for a face
}
```

### CellShape Enum

Defines the different cell shapes supported in the mesh.

```rust
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}
```

### FaceShape Enum

Defines the different face shapes supported in the mesh.

```rust
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}
```

---

## **4. Mesh Entities and Connectivity**

Mesh entities are fundamental components such as vertices, edges, faces, and cells. The `Geometry` module interacts closely with the `Mesh` structure, using mesh entities to perform geometric computations.

- **Vertices**: Points in 3D space.
- **Faces**: Surfaces bounded by edges (e.g., triangles, quadrilaterals).
- **Cells**: Volumetric elements (e.g., tetrahedrons, hexahedrons).

The module relies on the mesh's connectivity information to access entities and their associated vertices.

---

## **5. Working with Vertices**

### Adding and Updating Vertices

To set or update a vertex's coordinates:

```rust
pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]);
```

**Example:**

```rust
let mut geometry = Geometry::new();
geometry.set_vertex(0, [1.0, 2.0, 3.0]); // Adds or updates vertex at index 0
```

- If the `vertex_index` exceeds the current size of the `vertices` vector, it automatically resizes.
- Updating a vertex invalidates the cache to ensure consistency.

---

## **6. Computing Geometric Properties**

### Cell Centroids

Computes and caches the centroid of a cell based on its shape and vertices.

```rust
pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3];
```

- **Usage**: Provides the geometric center of a cell, which is essential in various numerical methods.
- **Caching**: The computed centroid is stored in the cache for future use.

### Cell Volumes

Computes and caches the volume of a cell.

```rust
pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64;
```

- **Usage**: Calculates the volume using shape-specific methods.
- **Caching**: Stored in the cache to avoid redundant computations.

### Face Areas

Computes the area of a face based on its shape.

```rust
pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Supports**: Triangles and quadrilaterals.
- **Caching**: Face areas are cached using `face_id`.

### Face Centroids

Computes the centroid of a face.

```rust
pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Usage**: Essential for flux calculations across faces in finite volume methods.

### Face Normals

Computes and caches the normal vector of a face.

```rust
pub fn compute_face_normal(
    &mut self,
    mesh: &Mesh,
    face: &MeshEntity,
    cell: &MeshEntity,
) -> Option<[f64; 3]>;
```

- **Usage**: Normal vectors are crucial for calculating fluxes and enforcing boundary conditions.
- **Caching**: Normals are cached for efficiency.

### Distance Calculations

Computes the Euclidean distance between two points.

```rust
pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64;
```

- **Usage**: Used internally for height calculations, distances between centroids, etc.

---

## **7. Shape-Specific Computations**

### Triangles

#### Compute Centroid

```rust
pub fn compute_triangle_centroid(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the three vertex coordinates.

#### Compute Area

```rust
pub fn compute_triangle_area(&self, triangle_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: Using the cross product of two edge vectors.

#### Compute Normal

```rust
pub fn compute_triangle_normal(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Normal vector via cross product of two edges.

### Quadrilaterals

#### Compute Area

```rust
pub fn compute_quadrilateral_area(&self, quad_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: By splitting the quadrilateral into two triangles.

#### Compute Centroid

```rust
pub fn compute_quadrilateral_centroid(&self, quad_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the four vertex coordinates.

#### Compute Normal

```rust
pub fn compute_quadrilateral_normal(&self, quad_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the normals of the two triangles formed.

### Tetrahedrons

#### Compute Centroid

```rust
pub fn compute_tetrahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the four vertex coordinates.

#### Compute Volume

```rust
pub fn compute_tetrahedron_volume(&self, tet_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: Using the determinant of a matrix formed by edges from one vertex.

### Hexahedrons

#### Compute Centroid

```rust
pub fn compute_hexahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the eight vertex coordinates.

#### Compute Volume

```rust
pub fn compute_hexahedron_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: By decomposing the hexahedron into tetrahedrons and summing their volumes.

### Prisms

#### Compute Centroid

```rust
pub fn compute_prism_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Average of the centroids of the top and bottom triangles.

#### Compute Volume

```rust
pub fn compute_prism_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: Base area multiplied by height.

### Pyramids

#### Compute Centroid

```rust
pub fn compute_pyramid_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- **Calculates**: Weighted average of the base centroid and apex.

#### Compute Volume

```rust
pub fn compute_pyramid_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64;
```

- **Calculates**: For triangular base (tetrahedron) or square base (sum of two tetrahedrons).

---

## **8. Caching and Performance Optimization**

The `Geometry` module uses caching extensively to improve performance:

- **Cache Structure**: `FxHashMap<usize, GeometryCache>` stores computed properties keyed by entity IDs.
- **Thread Safety**: The cache is wrapped in a `Mutex` to ensure thread-safe access in parallel computations.
- **Invalidation**: When geometry changes (e.g., vertex updates), the cache is invalidated using:

  ```rust
  fn invalidate_cache(&mut self);
  ```

- **Usage**: Before performing computations, the module checks if the result is already cached.

---

## **9. Advanced Usage**

### Updating All Cell Volumes

Recomputes and updates the volumes of all cells in parallel.

```rust
pub fn update_all_cell_volumes(&mut self, mesh: &Mesh);
```

- **Parallel Computation**: Utilizes Rayon for concurrent processing.
- **Usage**: Essential after significant mesh modifications.

### Computing Total Volume and Centroid

#### Total Volume

```rust
pub fn compute_total_volume(&self) -> f64;
```

- **Calculates**: Sum of all cell volumes.

#### Total Centroid

```rust
pub fn compute_total_centroid(&self) -> [f64; 3];
```

- **Calculates**: Average of all cell centroids.

---

## **10. Best Practices**

- **Cache Management**: Always invalidate the cache when making changes to geometry to ensure consistency.
- **Thread Safety**: When accessing or modifying shared data, ensure thread safety to prevent data races.
- **Shape Verification**: Before performing computations, verify that the number of vertices matches the expected shape.

  ```rust
  assert!(cell_vertices.len() == expected_count, "Incorrect number of vertices");
  ```

- **Error Handling**: Use assertions to catch invalid inputs early.

---

## **11. Testing and Validation**

The module includes comprehensive tests for each shape and computation:

- **Unit Tests**: Validate individual functions with known inputs and outputs.
- **Degenerate Cases**: Tests include degenerate shapes (e.g., zero area or volume) to ensure robustness.
- **Accuracy**: Floating-point comparisons account for precision errors using tolerances (e.g., `1e-10`).

**Example Test for Tetrahedron Volume:**

```rust
#[test]
fn test_tetrahedron_volume() {
    let geometry = Geometry::new();
    let tetrahedron_vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let volume = geometry.compute_tetrahedron_volume(&tetrahedron_vertices);
    assert!((volume - 1.0 / 6.0).abs() < 1e-10);
}
```

---

## **12. Conclusion**

The `Geometry` module is a powerful tool within the Hydra framework, providing essential functionalities for geometric computations on mesh entities. Its support for various shapes and caching mechanisms makes it efficient and versatile for computational simulations.

By understanding and utilizing the methods and structures provided, users can effectively perform and optimize geometric calculations necessary for advanced numerical methods in computational fluid dynamics, structural analysis, and other fields.

---

