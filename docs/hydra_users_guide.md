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

The `use_cases` module in Hydra implements specific operations that fulfill the needs of various higher-level application workflows. In a clean architecture, `use_cases` act as intermediaries between the core application logic and the lower-level modules like data handling, UI, or external interfaces. This structure promotes modularity, testability, and flexibility by separating domain logic from implementation details.

In Hydra, the `use_cases` module is designed to handle specialized tasks, such as creating and initializing matrices and right-hand side (RHS) vectors, which are essential in solving computational fluid dynamics (CFD) equations. By abstracting these tasks into separate use cases, Hydra can:
1. **Ensure Clear and Testable Operations**: Each use case is responsible for a well-defined operation, making it easy to test and validate.
2. **Encapsulate Domain-Specific Logic**: Use cases encapsulate the logic of matrix and RHS construction, initialization, and manipulation, keeping the core logic isolated from the specifics of data handling or external dependencies.
3. **Enable Flexibility and Reusability**: By separating use cases, other parts of Hydra (or even different projects) can reuse these operations without modification.

## Background: Clean Architecture Principles in Use Cases

In clean architecture:
- **Use Cases**: Represent application-specific operations (e.g., creating and setting up a matrix for CFD).
- **Entities**: Define core business objects (e.g., `Matrix`, `Vector`).
- **Interface Adapters**: Provide interfaces that handle implementation-specific details, like adapting external libraries or custom data structures to the Hydra environment.
  
By following this structure, `use_cases` operate independently of changes in interface adapters or low-level data handling, as they rely only on high-level interfaces. They bridge the gap between domain-specific tasks and the Hydra project's data structures.

### Key Concepts in `use_cases`

- **Single Responsibility**: Each use case file focuses on one operation, such as constructing a matrix (`matrix_construction.rs`) or creating an RHS vector (`rhs_construction.rs`).
- **Explicit Interfaces**: Use cases interact with data through interfaces, not concrete implementations, making it easy to replace dependencies or adjust workflows without altering core logic.
- **Dependency Inversion**: The `use_cases` module depends on high-level abstractions, like `MatrixOperations` and `Vector`, rather than specific implementations, promoting loose coupling.

## Overview of Use Cases in Hydra

### 1. `matrix_construction.rs`

This use case is responsible for creating and initializing matrices used in various computational tasks in Hydra. 

- **Purpose**: To build and set up matrices with specified dimensions and values, which can then be used in simulation workflows.
- **Functions**:
  - **`build_zero_matrix`**: Creates a new dense matrix with the specified number of rows and columns, initialized to zero. This function uses `MatrixAdapter` to ensure consistency in matrix creation across Hydra.
  - **`initialize_matrix_with_value`**: Fills an existing matrix with a specific value. This can be helpful for setting initial conditions in simulations.
  - **`resize_matrix`**: Changes the dimensions of a matrix, maintaining data where possible. It relies on the `ExtendedMatrixOperations` trait, ensuring the operation works consistently for matrices of different types.
  
  **Usage Example**:
  ```rust,ignore
  let mut matrix = MatrixConstruction::build_zero_matrix(4, 4);
  MatrixConstruction::initialize_matrix_with_value(&mut matrix, 1.0);
  MatrixConstruction::resize_matrix(&mut matrix, 6, 6);
  ```

  This setup constructs a 4x4 matrix filled with 1.0 and resizes it to 6x6.

### 2. `rhs_construction.rs`

This use case constructs and manages the right-hand side (RHS) vector used in solving linear systems, essential for CFD and other mathematical modeling tasks.

- **Purpose**: To build, initialize, and resize RHS vectors as needed in various computations within Hydra.
- **Functions**:
  - **`build_zero_rhs`**: Creates an RHS vector of a given size, initialized to zero, facilitating consistent initialization.
  - **`initialize_rhs_with_value`**: Sets each element in the RHS vector to a specified value. This is useful for setting boundary conditions or initial states in simulation workflows.
  - **`resize_rhs`**: Resizes the RHS vector, preserving existing data and initializing new entries to zero.

  **Usage Example**:
  ```rust,ignore
  let mut rhs_vector = RHSConstruction::build_zero_rhs(5);
  RHSConstruction::initialize_rhs_with_value(&mut rhs_vector, 3.5);
  RHSConstruction::resize_rhs(&mut rhs_vector, 8);
  ```

  This creates a 5-element vector filled with 3.5, then resizes it to 8 elements, initializing new elements to zero.

## Testing and Validation

Following Test-Driven Development (TDD), each function in `matrix_construction.rs` and `rhs_construction.rs` has a corresponding test to verify correct behavior.

### Testing Guidelines

1. **Isolation**: Each test targets a single function, ensuring each function behaves as expected in isolation.
2. **Consistency**: Tests ensure that matrix and vector creation, resizing, and initialization are consistent across Hydra, regardless of the underlying data structure or adapter.
3. **Boundary Cases**: Tests check edge cases, such as resizing matrices and vectors to larger or smaller dimensions, to validate data preservation and initialization.

### Summary

The `use_cases` module in Hydra:
- Adheres to clean architecture principles by encapsulating specific tasks, interfacing with high-level abstractions, and ensuring modular and flexible code.
- Provides reusable, testable, and consistent operations for initializing and managing matrices and RHS vectors, critical components in the Hydra project.

This modular structure enables developers to work confidently with Hydra’s data structures, knowing the `use_cases` provide a stable, consistent foundation for their operations.

---

The `src/interface_adapters` module in Hydra is designed to standardize and manage interactions between the core data structures (`Vector` and `Matrix`) and external components like `faer`. This setup allows seamless manipulation of mathematical objects, ensuring compatibility with various mathematical operations, resizing, and solver preconditioning while maintaining a consistent interface.

### Overview of `src/interface_adapters/`

#### 1. Module Structure

- **`mod.rs`**: This serves as the main entry point for the `interface_adapters` module, exposing two submodules:
  - `vector_adapter`
  - `matrix_adapter`

Each adapter in this module encapsulates functionality for its corresponding mathematical structure (`Vector` or `Matrix`), supporting operations like creation, resizing, element access, and preconditioning. This abstraction helps ensure that changes to internal vector or matrix handling are isolated within these adapters, simplifying integration with other Hydra components.

#### 2. `vector_adapter.rs`

The `VectorAdapter` struct offers functions to create, resize, and access elements within a `Vector`. It leverages `faer::Mat` to support dense vector structures, which are widely applicable in numerical operations.

- **Key Functions**:
  - **`new_dense_vector`**: Creates a dense vector with specified dimensions, initializing all elements to zero.
  - **`resize_vector`**: Allows resizing of a vector by altering its length, using Rust’s `Vec` resizing to maintain safety.
  - **`set_element` / `get_element`**: Provides setter and getter functions for individual elements within the vector, enforcing safe and consistent element access.

- **Tests**:
  - Each function is tested to ensure correct vector creation, element setting and getting, and resizing behavior. For instance, `test_resize_vector` checks if the vector adjusts its length correctly when resized, maintaining initial values where applicable.

#### 3. `matrix_adapter.rs`

The `MatrixAdapter` struct is the interface for working with `Matrix` structures in Hydra. It is implemented with support for dense matrix handling using `faer::Mat`, and it accommodates resizing and preconditioning, which are essential in matrix operations for numerical solvers.

- **Key Functions**:
  - **`new_dense_matrix`**: Creates a dense matrix with specified dimensions, initialized to zero.
  - **`resize_matrix`**: Handles resizing operations, leveraging the `ExtendedMatrixOperations` trait to safely adjust matrix dimensions.
  - **`set_element` / `get_element`**: Provides access to specific elements within the matrix for both read and write operations, enforcing controlled access to matrix data.
  - **`apply_preconditioner`**: Integrates with the `Preconditioner` trait, demonstrating compatibility with solver preconditioning by applying transformations to the matrix.

- **Tests**:
  - This module includes tests for matrix creation, element access, and validation of matrix data after operations. Tests like `test_set_and_get_element` ensure that element updates are correctly reflected in the matrix.

### Summary

The `interface_adapters` module abstracts operations for `Vector` and `Matrix` data types, supporting:
- **Standardized API**: Uniform methods for vector and matrix handling.
- **Compatibility**: Encapsulation ensures compatibility across different modules, even if underlying libraries or implementations change.
- **Safety**: Controlled resizing and element access to prevent data inconsistency or memory issues.
- **Extensibility**: Easily accommodates additional features or optimizations, as adapters separate data handling logic from core algorithms.

This setup ensures that Hydra’s core can flexibly interact with various mathematical and solver components, maintaining clean and efficient interfaces for advanced fluid dynamics simulations.

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

# Hydra `Time Stepping` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Time Stepping Module](#2-overview-of-the-time-stepping-module)
3. [Core Components](#3-core-components)
   - [TimeDependentProblem Trait](#timedependentproblem-trait)
   - [TimeStepper Trait](#timestepper-trait)
4. [Implemented Time Stepping Methods](#4-implemented-time-stepping-methods)
   - [Forward Euler Method](#forward-euler-method)
   - [Backward Euler Method](#backward-euler-method)
5. [Using the Time Stepping Module](#5-using-the-time-stepping-module)
   - [Defining a Time-Dependent Problem](#defining-a-time-dependent-problem)
   - [Selecting a Time Stepping Method](#selecting-a-time-stepping-method)
   - [Performing Time Steps](#performing-time-steps)
6. [Planned Features and Not Yet Implemented Components](#6-planned-features-and-not-yet-implemented-components)
   - [Adaptive Time Stepping](#adaptive-time-stepping)
   - [Higher-Order Methods](#higher-order-methods)
   - [Step Size Control and Error Estimation](#step-size-control-and-error-estimation)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Time Stepping` module of the Hydra computational framework. This module provides tools and interfaces for numerically solving time-dependent problems, such as ordinary differential equations (ODEs) and partial differential equations (PDEs). Time stepping is a critical component in simulations that evolve over time, and Hydra's module is designed to offer flexibility and extensibility for various time integration methods.

**Note**: As of the current version, some parts of the module are not yet implemented. This guide will point out the implemented features and those planned for future development.

---

## **2. Overview of the Time Stepping Module**

The `Time Stepping` module is structured to facilitate the integration of different time-stepping methods and to provide a unified interface for time-dependent problems. The key components include:

- **Traits**: Define interfaces for time-dependent problems and time-stepping methods.
- **Time Stepping Methods**: Implementations of specific algorithms like Forward Euler and Backward Euler.
- **Adaptivity Components** (Planned): Modules for error estimation and adaptive step size control.

**Module Structure**:

```bash
time_stepping/
├── adaptivity/
│   ├── error_estimate.rs      # Not yet implemented
│   ├── step_size_control.rs   # Not yet implemented
│   └── mod.rs                 # Not yet implemented
├── methods/
│   ├── backward_euler.rs      # Implemented
│   ├── euler.rs               # Implemented
│   ├── crank_nicolson.rs      # Not yet implemented
│   ├── runge_kutta.rs         # Not yet implemented
│   └── mod.rs
├── ts.rs                      # Core traits and structures
└── mod.rs                     # Module exports
```

---

## **3. Core Components**

### TimeDependentProblem Trait

The `TimeDependentProblem` trait defines the interface for any time-dependent problem that can be solved using the time-stepping methods provided. Implementing this trait requires specifying:

- **State Type**: The type representing the system's state, which must implement the `Vector` trait.
- **Time Type**: The type representing time, typically `f64`.
- **Methods**:
  - `compute_rhs`: Computes the right-hand side (RHS) of the system.
  - `initial_state`: Provides the initial state of the system.
  - `time_to_scalar`: Converts time values to the scalar type used in vectors.
  - `get_matrix`: Returns a matrix representation if applicable (used in implicit methods).
  - `solve_linear_system`: Solves linear systems for implicit methods.

**Trait Definition**:

```rust
pub trait TimeDependentProblem {
    type State: Vector;
    type Time;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar;

    fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}
```

### TimeStepper Trait

The `TimeStepper` trait defines the interface for time-stepping methods. It requires the implementation of:

- `step`: Advances the solution by one time step.
- `adaptive_step`: Performs an adaptive time step (not fully implemented yet).
- `set_time_interval`: Sets the start and end times for the simulation.
- `set_time_step`: Sets the fixed time step size.

**Trait Definition**:

```rust
pub trait TimeStepper<P: TimeDependentProblem> {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    fn set_time_step(&mut self, dt: P::Time);
}
```

---

## **4. Implemented Time Stepping Methods**

As of the current version, the following time-stepping methods are implemented:

### Forward Euler Method

The Forward Euler method is an explicit first-order method for numerically integrating ordinary differential equations.

**Implementation Highlights**:

- **Module**: `euler.rs`
- **Struct**: `ForwardEuler`
- **Key Characteristics**:
  - Simple and easy to implement.
  - Suitable for problems where accuracy and stability are not critical.

**Usage**:

Implementing the `TimeStepper` trait for `ForwardEuler`:

```rust
pub struct ForwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for ForwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut rhs = problem.initial_state();
        problem.compute_rhs(time, state, &mut rhs)?;
        let scalar_dt = problem.time_to_scalar(dt);
        state.axpy(scalar_dt, &rhs);
        Ok(())
    }
    
    // Other methods...
}
```

**Key Methods**:

- **`step`**: Performs the explicit update `state = state + dt * rhs`.

### Backward Euler Method

The Backward Euler method is an implicit first-order method, offering better stability properties compared to the Forward Euler method.

**Implementation Highlights**:

- **Module**: `backward_euler.rs`
- **Struct**: `BackwardEuler`
- **Key Characteristics**:
  - Implicit method requiring the solution of a linear system at each time step.
  - More stable for stiff problems.

**Usage**:

Implementing the `TimeStepper` trait for `BackwardEuler`:

```rust
pub struct BackwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut matrix = problem.get_matrix();
        let mut rhs = problem.initial_state();
        problem.compute_rhs(time, state, &mut rhs)?;
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;
        Ok(())
    }
    
    // Other methods...
}
```

**Key Methods**:

- **`step`**: Involves computing the RHS and solving the linear system `A * state = rhs`.

---

## **5. Using the Time Stepping Module**

### Defining a Time-Dependent Problem

To use the time-stepping methods, you need to define a struct that implements the `TimeDependentProblem` trait.

**Example**:

```rust
struct MyProblem {
    // Problem-specific fields
}

impl TimeDependentProblem for MyProblem {
    type State = Vec<f64>;
    type Time = f64;

    fn initial_state(&self) -> Self::State {
        // Return the initial state vector
    }

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Compute the RHS based on the current state and time
    }

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
        time
    }

    fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>> {
        // Return the system matrix if needed (for implicit methods)
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Solve the linear system for implicit methods
    }
}
```

### Selecting a Time Stepping Method

Choose a time-stepping method based on the problem's requirements:

- **Forward Euler**: For simple, non-stiff problems where computational efficiency is important.
- **Backward Euler**: For stiff problems requiring stability.

**Example**:

```rust
let mut stepper = ForwardEuler;
// or
let mut stepper = BackwardEuler;
```

### Performing Time Steps

Set up the time interval and time step size (if applicable):

```rust
stepper.set_time_interval(0.0, 10.0);
stepper.set_time_step(0.1);
```

Perform the time-stepping loop:

```rust
let mut state = problem.initial_state();
let mut time = 0.0;
let end_time = 10.0;
let dt = 0.1;

while time < end_time {
    stepper.step(&problem, time, dt, &mut state)?;
    time += dt;
}
```

**Error Handling**:

- Each `step` method returns a `Result`. Handle errors appropriately.
- Use `?` operator or match statements to manage `TimeSteppingError`.

---

## **6. Planned Features and Not Yet Implemented Components**

The `Time Stepping` module has several components and features planned for future implementation:

### Adaptive Time Stepping

- **Description**: Adjusting the time step size dynamically based on error estimates to improve efficiency and accuracy.
- **Current Status**: The `adaptive_step` method is defined in the `TimeStepper` trait but not fully implemented in existing methods.
- **Planned Components**:
  - **Error Estimation**: Modules to estimate the local truncation error.
  - **Step Size Control**: Algorithms to adjust `dt` based on error estimates.

### Higher-Order Methods

- **Crank-Nicolson Method**: A second-order implicit method combining Forward and Backward Euler.
- **Runge-Kutta Methods**: Higher-order explicit methods for improved accuracy.
- **Current Status**: These methods are listed in the module structure but not yet implemented.

### Step Size Control and Error Estimation

- **Modules**:
  - **`error_estimate.rs`**: Will provide functionalities for error estimation.
  - **`step_size_control.rs`**: Will implement algorithms for adjusting the time step size.
- **Adaptivity Module**: The `adaptivity` folder contains placeholders for these components.

**Note**: Users interested in these features should keep an eye on future releases of Hydra for updates.

---

## **7. Best Practices**

- **Choose the Right Method**: Select a time-stepping method appropriate for your problem's stiffness and accuracy requirements.
- **Implement Required Traits**: Ensure that your problem struct correctly implements all methods of the `TimeDependentProblem` trait.
- **Handle Errors**: Always handle potential errors returned by the `step` methods to avoid unexpected crashes.
- **Monitor Stability**: Be cautious with explicit methods for stiff problems; consider using implicit methods instead.
- **Stay Updated**: Keep track of updates to the module for new features and methods as they are implemented.

---

## **8. Conclusion**

The `Time Stepping` module in Hydra provides a flexible framework for integrating time-dependent problems using various numerical methods. While the current implementation includes fundamental methods like Forward and Backward Euler, the framework is designed to accommodate more advanced techniques in the future.

By defining clear interfaces through the `TimeDependentProblem` and `TimeStepper` traits, users can implement custom problems and apply different time-stepping strategies with ease. The planned features, such as adaptive time stepping and higher-order methods, will further enhance the module's capabilities.

---

**Note**: As the module is still under development, some features are not yet available. Users are encouraged to contribute to the project or check back for updates in future releases.

---

# Hydra `Solver` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Solver Module](#2-overview-of-the-solver-module)
3. [Krylov Subspace Solvers](#3-krylov-subspace-solvers)
   - [KSP Trait](#ksp-trait)
   - [Conjugate Gradient Solver](#conjugate-gradient-solver)
   - [GMRES Solver](#gmres-solver)
4. [Preconditioners](#4-preconditioners)
   - [Overview of Preconditioners](#overview-of-preconditioners)
   - [Jacobi Preconditioner](#jacobi-preconditioner)
   - [LU Preconditioner](#lu-preconditioner)
   - [ILU Preconditioner](#ilu-preconditioner)
   - [Cholesky Preconditioner](#cholesky-preconditioner)
5. [Using the Solver Module](#5-using-the-solver-module)
   - [Setting Up a Solver](#setting-up-a-solver)
   - [Applying Preconditioners](#applying-preconditioners)
   - [Solving Linear Systems](#solving-linear-systems)
6. [Examples and Usage](#6-examples-and-usage)
   - [Example with Conjugate Gradient](#example-with-conjugate-gradient)
   - [Example with GMRES](#example-with-gmres)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Solver` module of the Hydra computational framework. This module provides a suite of Krylov subspace solvers and preconditioners designed to solve large, sparse linear systems efficiently. The solvers are essential for numerical simulations in computational fluid dynamics (CFD) and other fields requiring iterative solutions to linear systems.

---

## **2. Overview of the Solver Module**

The `Solver` module in Hydra is organized into several components:

- **Krylov Subspace Solvers (KSP)**: Abstract interface for solvers like Conjugate Gradient (CG) and Generalized Minimal Residual Solver (GMRES).
- **Preconditioners**: Modules that improve convergence rates by transforming the system into a more favorable form.
- **Solver Manager**: A high-level interface that integrates solvers and preconditioners for flexible usage.

The module's design emphasizes:

- **Flexibility**: Ability to interchange solvers and preconditioners easily.
- **Performance**: Utilization of parallel computing via Rayon for efficiency.
- **Extensibility**: Support for adding new solvers and preconditioners.

---

## **3. Krylov Subspace Solvers**

### KSP Trait

The `KSP` trait defines a common interface for all Krylov subspace solvers in the Hydra framework. It ensures consistency and allows for easy interchangeability between different solver implementations.

```rust
pub trait KSP {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}
```

- **Parameters**:
  - `a`: The system matrix `A`.
  - `b`: The right-hand side vector `b`.
  - `x`: The solution vector `x`, which will be updated.
- **Returns**: A `SolverResult` containing convergence information.

### Conjugate Gradient Solver

The `ConjugateGradient` struct implements the `KSP` trait, providing an efficient solver for symmetric positive-definite (SPD) systems.

#### Key Features

- **Preconditioning Support**: Optional preconditioner can be applied.
- **Parallel Computation**: Utilizes Rayon for parallel operations.

#### Usage

```rust
let mut cg = ConjugateGradient::new(max_iter, tolerance);
cg.set_preconditioner(Box::new(Jacobi::default())); // Optional
let result = cg.solve(&a, &b, &mut x);
```

- **Methods**:
  - `new(max_iter, tol)`: Creates a new instance with specified maximum iterations and tolerance.
  - `set_preconditioner(preconditioner)`: Sets an optional preconditioner.

### GMRES Solver

The `GMRES` (Generalized Minimal Residual Solver) is suitable for non-symmetric or non-positive-definite systems.

#### Key Features

- **Restart Mechanism**: Supports restarts to prevent loss of orthogonality.
- **Preconditioning Support**: Can incorporate preconditioners.

#### Usage

```rust
let mut gmres = GMRES::new(max_iter, tolerance, restart);
gmres.set_preconditioner(Arc::new(Jacobi::default())); // Optional
let result = gmres.solve(&a, &b, &mut x);
```

- **Methods**:
  - `new(max_iter, tol, restart)`: Initializes the solver with specified parameters.
  - `set_preconditioner(preconditioner)`: Sets an optional preconditioner.

---

## **4. Preconditioners**

### Overview of Preconditioners

Preconditioners transform a linear system into an equivalent one that has more favorable properties for iterative solution methods. They aim to improve convergence rates and overall solver performance.

### Jacobi Preconditioner

The Jacobi preconditioner is one of the simplest preconditioners, utilizing the inverse of the diagonal elements of the matrix.

#### Usage

```rust
let jacobi_preconditioner = Jacobi::default();
cg.set_preconditioner(Box::new(jacobi_preconditioner));
```

#### Implementation Highlights

- **Parallelism**: Uses `rayon` for parallel computation across rows.
- **Thread Safety**: Employs `Arc<Mutex<T>>` to ensure safe concurrent access.

### LU Preconditioner

The LU preconditioner uses LU decomposition to factorize the matrix and solve the preconditioned system efficiently.

#### Usage

```rust
let lu_preconditioner = LU::new(&a);
gmres.set_preconditioner(Arc::new(lu_preconditioner));
```

#### Implementation Highlights

- **Partial Pivoting**: Utilizes partial pivot LU decomposition from the `faer` library.
- **Efficient Solving**: Provides methods for forward and backward substitution.

### ILU Preconditioner

The Incomplete LU (ILU) preconditioner approximates the LU decomposition while preserving the sparsity pattern.

#### Usage

```rust
let ilu_preconditioner = ILU::new(&a);
gmres.set_preconditioner(Arc::new(ilu_preconditioner));
```

#### Implementation Highlights

- **Sparsity Preservation**: Discards small values to maintain sparsity.
- **Custom Decomposition**: Implements a sparse ILU decomposition algorithm.

### Cholesky Preconditioner

The Cholesky preconditioner is suitable for SPD matrices and uses Cholesky decomposition for efficient solving.

#### Usage

```rust
let cholesky_preconditioner = CholeskyPreconditioner::new(&a)?;
cg.set_preconditioner(Box::new(cholesky_preconditioner));
```

#### Implementation Highlights

- **Error Handling**: Returns a `Result` to handle decomposition failures.
- **Lower Triangular Factorization**: Decomposes the matrix into lower and upper triangular matrices.

---

## **5. Using the Solver Module**

### Setting Up a Solver

To set up a solver, you need to:

1. **Choose a Solver**: Decide between `ConjugateGradient` or `GMRES` based on your system's properties.
2. **Initialize the Solver**: Create an instance with appropriate parameters.

**Example**:

```rust
let max_iter = 1000;
let tolerance = 1e-6;

let mut solver = ConjugateGradient::new(max_iter, tolerance);
```

### Applying Preconditioners

Preconditioners can significantly improve solver performance.

**Adding a Preconditioner**:

```rust
let preconditioner = Box::new(Jacobi::default());
solver.set_preconditioner(preconditioner);
```

### Solving Linear Systems

To solve the system `Ax = b`:

1. **Prepare the System Matrix and Vectors**: Ensure `A`, `b`, and `x` are properly defined.
2. **Call the Solver**:

```rust
let result = solver.solve(&a, &b, &mut x);
```

3. **Check the Result**:

```rust
if result.converged {
    println!("Solver converged in {} iterations.", result.iterations);
} else {
    println!("Solver did not converge.");
}
```

---

## **6. Examples and Usage**

### Example with Conjugate Gradient

**Problem**: Solve `Ax = b` where `A` is SPD.

**Setup**:

```rust
use faer::mat;

let a = mat![
    [4.0, 1.0],
    [1.0, 3.0],
];

let b = mat![
    [1.0],
    [2.0],
];

let mut x = Mat::<f64>::zeros(2, 1);
```

**Solver Initialization**:

```rust
let mut cg = ConjugateGradient::new(100, 1e-6);
```

**Applying Preconditioner** (Optional):

```rust
let jacobi_preconditioner = Box::new(Jacobi::default());
cg.set_preconditioner(jacobi_preconditioner);
```

**Solving**:

```rust
let result = cg.solve(&a, &b, &mut x);
```

**Result Checking**:

```rust
if result.converged {
    println!("Solution: {:?}", x);
} else {
    println!("Solver did not converge.");
}
```

### Example with GMRES

**Problem**: Solve a non-symmetric system `Ax = b`.

**Setup**:

```rust
let a = mat![
    [2.0, 1.0],
    [3.0, 4.0],
];

let b = mat![
    [1.0],
    [2.0],
];

let mut x = Mat::<f64>::zeros(2, 1);
```

**Solver Initialization**:

```rust
let mut gmres = GMRES::new(100, 1e-6, 2);
```

**Applying Preconditioner** (Optional):

```rust
let lu_preconditioner = Arc::new(LU::new(&a));
gmres.set_preconditioner(lu_preconditioner);
```

**Solving**:

```rust
let result = gmres.solve(&a, &b, &mut x);
```

**Result Checking**:

```rust
if result.converged {
    println!("Solution: {:?}", x);
} else {
    println!("Solver did not converge.");
}
```

---

## **7. Best Practices**

- **Select Appropriate Solver**: Use CG for SPD systems and GMRES for non-symmetric systems.
- **Utilize Preconditioners**: Always consider applying a preconditioner to improve convergence.
- **Monitor Convergence**: Check the `SolverResult` for convergence status and residual norms.
- **Thread Safety**: Ensure that matrices and vectors are thread-safe if using custom implementations.
- **Handle Errors**: Be prepared to handle cases where solvers do not converge within the maximum iterations.

---

## **8. Conclusion**

The `Solver` module in Hydra provides robust and flexible tools for solving large, sparse linear systems. By offering multiple solver options and preconditioners, it caters to a wide range of problems encountered in computational simulations. Proper utilization of this module can lead to significant performance improvements and more accurate results in your simulations.

---

**Note**: For advanced usage and custom implementations, refer to the official Hydra documentation and source code.

---

The `extrusion` module in HYDRA is designed to transform 2D geophysical meshes into 3D volumetric meshes. It provides a system for extruding 2D cells, like quadrilaterals and triangles, into 3D volumes such as hexahedrons and prisms. 

The module is structured into four main sections:

---

### 1. Core (`core`)

**Purpose**: Defines essential data structures and traits needed for extrusion. The `ExtrudableMesh` trait is introduced to standardize extrusion properties and behaviors across different mesh types.

- **Components**:
  - `extrudable_mesh`: Defines the `ExtrudableMesh` trait.
  - `hexahedral_mesh`: Implements `QuadrilateralMesh` for extrusion into hexahedrons.
  - `prismatic_mesh`: Implements `TriangularMesh` for extrusion into prisms.

**Key Structure: `ExtrudableMesh` Trait**  
Defines methods to:
  - Check if a mesh is extrudable.
  - Access vertices and cells.
  - Identify mesh type (quadrilateral or triangular).

**Example**:
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
use hydra::extrusion::core::extrudable_mesh::ExtrudableMesh;

let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
let cells = vec![vec![0, 1, 2, 3]];
let quad_mesh = QuadrilateralMesh::new(vertices, cells);
assert!(quad_mesh.is_valid_for_extrusion());
Ok(())
# }
```

---

### 2. Infrastructure (`infrastructure`)

**Purpose**: Manages file I/O for mesh data and provides logging utilities.

- **Components**:
  - `mesh_io`: Manages loading and saving of mesh files.
  - `logger`: Logs extrusion operations, supporting info, warning, and error levels.

**Key Structure: `MeshIO`**  
- **`load_2d_mesh(file_path: &str) -> Result<Box<dyn ExtrudableMesh>, String>`**  
  Loads a 2D mesh, supporting either quadrilateral or triangular cell structures.
  
- **`save_3d_mesh(mesh: &Mesh, file_path: &str) -> Result<(), String>`**  
  Saves a 3D extruded mesh to a specified file.

**Example**:
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use hydra::extrusion::infrastructure::mesh_io::MeshIO;
use hydra::domain::mesh::Mesh;

let mesh = Mesh::new();
MeshIO::save_3d_mesh(&mesh, "outputs/extruded_mesh.msh")?;
Ok(())
# }
```

---

### 3. Interface Adapters (`interface_adapters`)

**Purpose**: Provides the main entry point for extrusion operations through `ExtrusionService`, which determines the type of 3D extrusion (hexahedral or prismatic).

- **Component**:
  - `extrusion_service`: Manages the extrusion process based on mesh type.

**Key Method: `extrude_mesh`**  
Extrudes a 2D mesh to a 3D form, selecting hexahedral or prismatic extrusion based on the input mesh type.

**Example**:
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use hydra::extrusion::core::hexahedral_mesh::QuadrilateralMesh;
use hydra::extrusion::interface_adapters::extrusion_service::ExtrusionService;

let quad_mesh = QuadrilateralMesh::new(
    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    vec![vec![0, 1, 2, 3]],
);
let depth = 5.0;
let layers = 3;
let extruded_mesh = ExtrusionService::extrude_mesh(&quad_mesh, depth, layers)?;
Ok(())
# }
```

---

### 4. Use Cases (`use_cases`)

**Purpose**: Provides functions to execute vertex and cell extrusion, coordinating these to generate complete 3D meshes.

- **Components**:
  - `vertex_extrusion`: Extrudes vertices along the z-axis to create 3D layers.
  - `cell_extrusion`: Extrudes 2D cells into 3D volumes.
  - `extrude_mesh`: Orchestrates the full extrusion process.

**Example of Vertex Extrusion**:
```rust
use hydra::extrusion::use_cases::vertex_extrusion::VertexExtrusion;

let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
let depth = 3.0;
let layers = 2;
let extruded_vertices = VertexExtrusion::extrude_vertices(vertices, depth, layers);
```

---

### Summary

The `extrusion` module is essential for converting 2D geophysical meshes into 3D, utilizing modular design and robust error handling to ensure compatibility with various mesh types and configurations. Each component in the pipeline—from core structures to use cases—contributes to a flexible and reliable extrusion process suitable for complex simulations in geophysical contexts.

---

The `input_output` module currently comprises two primary components: `gmsh_parser` and `mesh_generation`. 

Together, these handle importing external mesh data and generating geometric mesh structures. 

Below is a detailed breakdown of each component, its purpose, and recommended improvements to enhance modularity, maintainability, and performance.

---

#### Module: `gmsh_parser`
**Purpose**:  
The `gmsh_parser` module is responsible for reading mesh files in the Gmsh format and populating an internal `Mesh` structure based on the file's contents. The parser identifies sections in the Gmsh file, reads nodes and elements, and maps relationships between elements and vertices.

**Key Functions**:
1. **`from_gmsh_file(file_path: &str) -> Result<Mesh, io::Error>`**:  
   - This main function opens a specified Gmsh file, parses it line by line, and fills a `Mesh` structure.
   - It divides parsing into `Nodes` and `Elements` sections and, using helper methods, reads and sets vertices and elements in the mesh.

2. **`parse_node(line: &str) -> Result<(usize, [f64; 3]), io::Error>`**:
   - Parses individual nodes from the file, extracting node ID and coordinates.

3. **`parse_element(line: &str) -> Result<(usize, Vec<usize>), io::Error>`**:
   - Parses elements, extracting element ID and associated vertex IDs.

4. **`parse_next` Utility Function**:
   - Provides a helper for parsing the next value in a line, with error handling.

**Recommendations for `gmsh_parser`**:
- **Modularize Parsing Logic**:
  - Consider breaking down `from_gmsh_file` into smaller, private functions for each section (`parse_nodes_section`, `parse_elements_section`). This would increase readability and simplify debugging.
  
- **Error Handling Enhancements**:
  - Currently, the module has limited error messaging. Add specific error context (e.g., file line numbers, Gmsh section identifiers) to facilitate troubleshooting.
  
- **Parallel Processing for Large Files**:
  - Implement concurrent reading for large files using threads or asynchronous I/O, particularly if nodes and elements are processed sequentially. Rust's `async-std` or `tokio` crates can be used to manage file I/O more efficiently in future extensions.

---

#### Module: `mesh_generation`
**Purpose**:  
The `mesh_generation` module generates 2D and 3D meshes, as well as circular and triangular grids, directly within the code rather than from external files.

**Key Functions**:
1. **Public Functions**:
   - **`generate_rectangle_2d`**: Generates a 2D rectangular grid of vertices and cells based on specified dimensions and resolution.
   - **`generate_rectangle_3d`**: Creates a 3D rectangular grid of vertices and hexahedral cells.
   - **`generate_circle`**: Generates a circular mesh with a specified radius and divisions.

2. **Internal Helper Functions**:
   - **`generate_grid_nodes_2d`** and **`generate_grid_nodes_3d`**: Helper functions to create nodes in a 2D or 3D grid layout.
   - **`generate_circle_nodes`**: Generates nodes in a circular layout.
   - **`generate_quadrilateral_cells`**, **`generate_hexahedral_cells`**, and **`generate_triangular_cells`**: Generates cell relationships for quadrilateral, hexahedral, and triangular meshes, respectively.
   - **`_generate_faces_3d`**: A function stub for generating faces in 3D grids; currently underused but could support adding 3D mesh boundaries in the future.

**Recommendations for `mesh_generation`**:
- **Optimize Mesh Generation**:
  - Break down `generate_rectangle_2d` and `generate_rectangle_3d` into smaller functions or iterator-based loops to reduce nested for-loops. This would improve readability and maintainability.
  
- **Add Parallelization**:
  - For large 3D grids, implementing parallel generation of vertices and cells (using `rayon` or another parallel iterator library) could significantly speed up mesh generation.

- **Dynamic Grid Types and Error Checking**:
  - Integrate error checking when creating grids (e.g., handle zero or negative dimensions for robustness).
  - Allow specifying mesh properties, such as element types or boundary types, as arguments to the generator functions to improve flexibility.

- **Implement Additional Mesh Shapes**:
  - Support additional standard shapes such as ellipsoids, polygons, and custom grid patterns. These new shapes would provide more flexibility and are especially beneficial in environmental modeling.

---

#### Module: `tests`
**Purpose**:  
The `tests` submodule includes unit tests to verify the correctness of mesh importation and generation. Key test cases check for:
- Proper parsing and mapping of Gmsh files in `gmsh_parser`.
- Correct grid generation in `mesh_generation`, including vertex and cell counts.

**Test Coverage**:
1. **Gmsh Import Tests**: Tests for several standard mesh files (e.g., circular lakes, coastal islands) by comparing imported node and element counts.
2. **Mesh Generation Tests**: Verifies vertex and cell counts for 2D and 3D grids and circular meshes.

**Recommendations for `tests`**:
- **Expand Edge Case Coverage**:
  - Include tests for empty or malformed Gmsh files.
  - Add validation tests for degenerate cases in generated meshes (e.g., zero width or height).
  
- **Parallel Testing and Benchmarks**:
  - Use `cargo bench` for benchmarking mesh generation speeds to identify performance bottlenecks, especially in high-resolution or large 3D meshes.

- **Clearer Documentation**:
  - Include detailed doc comments for each test to explain expected behaviors and edge cases. This would aid future contributors in understanding the test intent.

---

### Module: `gmsh_parser`

**Purpose**:  
The `gmsh_parser` module provides functionality to import mesh data from Gmsh-formatted files into HYDRA’s internal `Mesh` structure. It reads mesh nodes, elements, and connectivity data, parsing the file into sections and associating each section’s data with the corresponding entities in the `Mesh`. This capability is fundamental for integrating externally defined meshes, particularly for complex environmental fluid simulations.

**Current Key Components**:
1. **`GmshParser` Struct**:  
   - The primary struct in this module, `GmshParser`, encapsulates methods for reading and parsing a Gmsh mesh file.
   - This struct does not hold any state itself, acting as a wrapper around static parsing methods.

2. **Primary Method: `from_gmsh_file(file_path: &str) -> Result<Mesh, io::Error>`**:
   - **Purpose**: The `from_gmsh_file` method is the main entry point for parsing a Gmsh file and converting its contents into a `Mesh` structure.
   - **Functionality**:
     - **File Handling**: Opens the specified file and wraps it in a buffered reader for efficient line-by-line processing.
     - **Parsing Flow**:
       - Divides the parsing into `Nodes` and `Elements` sections.
       - **Nodes Section**: Parses node identifiers and coordinates, storing each node’s ID and spatial coordinates in the mesh.
       - **Elements Section**: Parses elements, creating cell entities and establishing relationships between elements and their nodes.
   - **Error Handling**: Returns a `Result` with a custom `io::Error` on failure, such as invalid node counts or missing sections in the Gmsh file.

3. **Helper Method: `parse_node(line: &str) -> Result<(usize, [f64; 3]), io::Error>`**:
   - **Purpose**: Parses a single line representing a node in the `Nodes` section, extracting the node ID and 3D coordinates.
   - **Usage**: This method is called within `from_gmsh_file` for each line in the `Nodes` section, validating and structuring node data for storage in `Mesh`.
   - **Error Handling**: Returns an `io::Error` if the expected node format is not found.

4. **Helper Method: `parse_element(line: &str) -> Result<(usize, Vec<usize>), io::Error>`**:
   - **Purpose**: Parses individual elements in the `Elements` section, extracting an element ID and associated node IDs.
   - **Usage**: This method is invoked within `from_gmsh_file` to create cell entities and build relationships between elements and vertices.
   - **Error Handling**: Throws an error if the line does not follow the expected format, which is crucial for maintaining mesh consistency.

5. **Utility Function: `parse_next`**:
   - **Purpose**: A generic utility function that parses the next item from a line, throwing a customizable error if the item is missing or invalid.
   - **Usage**: This helper is used in both `parse_node` and `parse_element` to ensure that each required component of a line is present and correctly formatted.

**Current Limitations**:
- **Sequential Parsing**: The current implementation parses nodes and elements sequentially, which may be suboptimal for large files.
- **Basic Error Handling**: Error messages provide limited context, which can hinder debugging for malformed files.

**Recommended Enhancements**:
1. **Modularize Parsing Logic**:
   - Divide `from_gmsh_file` into smaller, private functions dedicated to handling specific sections (`parse_nodes_section`, `parse_elements_section`). This will increase readability and simplify maintenance and debugging.

2. **Enhanced Error Messages**:
   - Add line-specific error messages and section context in `from_gmsh_file` to facilitate easier diagnosis of file-related errors. For example, include the current line number or section name in error messages to indicate exactly where parsing failed.

3. **Parallel Processing for Large Files**:
   - Implement a parallelized approach for handling very large files. This could involve asynchronous processing of sections or utilizing threads for concurrent parsing. Libraries such as `rayon` or Rust’s async capabilities (`async-std`, `tokio`) could be explored to facilitate efficient I/O and parsing on larger meshes.

By incorporating these improvements, `gmsh_parser` will become more resilient, maintainable, and performant, particularly in handling large Gmsh files, aligning with HYDRA's goals for efficient mesh integration in environmental simulation workflows.

---

### Module: `mesh_generation`

**Purpose**:  
The `mesh_generation` module provides an API for creating various types of standard geometric meshes directly within HYDRA. These mesh generation functions are particularly useful for defining initial conditions or testing scenarios without relying on external mesh files. The module supports generating structured 2D and 3D rectangular meshes, circular meshes, and triangular grids, each tailored to the requirements of environmental fluid dynamics modeling.

**Current Key Components**:
1. **`MeshGenerator` Struct**:  
   - The `MeshGenerator` struct serves as the central struct for this module, encapsulating functions that define and populate a `Mesh` with vertices and cells for different geometric shapes.
   - `MeshGenerator` does not store any state, functioning instead as a namespace for the various mesh generation methods.

2. **Primary Methods**:
   - **`generate_rectangle_2d(width: f64, height: f64, nx: usize, ny: usize) -> Mesh`**:
     - **Purpose**: Generates a 2D rectangular mesh based on specified width, height, and resolution parameters.
     - **Functionality**:
       - **Vertex Generation**: Creates vertices in a 2D grid pattern using helper function `generate_grid_nodes_2d`.
       - **Cell Formation**: Divides the rectangle into quadrilateral cells by connecting adjacent vertices and storing each cell’s connectivity information in `Mesh`.
   
   - **`generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh`**:
     - **Purpose**: Generates a 3D rectangular mesh as a structured hexahedral grid.
     - **Functionality**:
       - **Vertex Generation**: Arranges vertices in a 3D grid using `generate_grid_nodes_3d`.
       - **Cell Formation**: Constructs hexahedral cells by connecting vertices in adjacent grid positions using `generate_hexahedral_cells`.

   - **`generate_circle(radius: f64, num_divisions: usize) -> Mesh`**:
     - **Purpose**: Creates a circular mesh based on a given radius and number of radial divisions.
     - **Functionality**:
       - **Vertex Generation**: Places vertices along a circular boundary and one at the center, using `generate_circle_nodes`.
       - **Cell Formation**: Constructs triangular cells between the center vertex and adjacent boundary vertices with `generate_triangular_cells`.

3. **Internal Helper Functions**:
   - **Vertex Generation**:
     - **`generate_grid_nodes_2d`** and **`generate_grid_nodes_3d`**: Produce a 2D or 3D grid of vertices, returning a vector of 3D coordinates.
     - **`generate_circle_nodes`**: Generates vertices along a circular boundary, with one additional central vertex.
   
   - **Cell Generation**:
     - **`generate_quadrilateral_cells`**: Produces quadrilateral cells in a 2D grid by connecting adjacent vertices.
     - **`generate_hexahedral_cells`**: Generates hexahedral cells for 3D grids by linking vertices in adjacent positions.
     - **`generate_triangular_cells`**: Creates triangular cells in a circular mesh by connecting the central vertex with adjacent boundary vertices.

   - **Face Generation**:
     - **`_generate_faces_3d`**: This function stub suggests future plans for generating boundary faces in 3D grids, currently underutilized but essential for defining boundary conditions.

**Current Limitations**:
- **Sequential Loops**: Mesh generation relies on sequential for-loops, which may hinder performance for large meshes.
- **Lack of Shape Flexibility**: Only basic shapes (rectangles and circles) are supported, limiting the scope of environmental simulations.
- **Basic Error Handling**: The module does not validate input parameters (e.g., checking for zero or negative dimensions).

**Recommended Enhancements**:
1. **Optimize Mesh Generation**:
   - Replace nested for-loops with iterator-based approaches or parallel iterators (e.g., using `rayon`) for improved efficiency, especially when generating large meshes.

2. **Parameter Validation**:
   - Add checks for invalid parameters, such as zero or negative dimensions. Provide informative error messages to prevent runtime issues.

3. **Expand Shape Library**:
   - Implement additional shapes, such as ellipsoids, polygons, and custom grid patterns. These new shapes would offer more flexibility and are valuable in environmental modeling scenarios that require irregular or custom geometries.

4. **Dynamic Grid Customization**:
   - Enhance `generate_rectangle_2d` and `generate_rectangle_3d` to accept additional mesh properties, such as element types or boundary flags, allowing for custom configurations and integration with other HYDRA components.

By incorporating these enhancements, the `mesh_generation` module will become a robust, flexible utility for mesh creation, supporting HYDRA’s goals for customizable and efficient environmental fluid dynamics modeling.

---

### Module: `tests`

**Purpose**:  
The `tests` module provides unit tests that validate the functionality of both `gmsh_parser` and `mesh_generation`. These tests ensure that imported and generated meshes meet expected specifications, including node counts, element connectivity, and structural consistency. The module's test coverage allows developers to confirm that changes or extensions to `input_output` maintain correct behavior and performance across different mesh types and file formats.

**Current Test Coverage**:
1. **Tests for `GmshParser` (Mesh Import)**:
   - **File Import Tests**: Each test checks the parsing functionality for a specific Gmsh file, confirming that the mesh generated from each file is both structurally correct and consistent with known properties of that mesh.
   - **Examples**:
     - **`test_circle_mesh_import`**: Validates the import of a circular mesh file by checking node and element counts.
     - **`test_coastal_island_mesh_import`, `test_lagoon_mesh_import`, and similar tests**: These tests verify mesh imports for various pre-defined geographic features (e.g., coastal islands, lagoons, meandering rivers), ensuring that `GmshParser` can correctly handle different environmental configurations.

2. **Tests for `MeshGenerator` (Direct Mesh Generation)**:
   - **Mesh Generation Tests**: Each test checks the correct generation of vertices and cells for different geometries, confirming the shape, structure, and connectivity of the generated mesh.
   - **Examples**:
     - **`test_generate_rectangle_2d`**: Confirms that a 2D rectangular mesh has the correct number of vertices and quadrilateral cells based on the specified grid resolution.
     - **`test_generate_rectangle_3d`**: Validates that a 3D rectangular mesh contains the expected number of vertices and hexahedral cells.
     - **`test_generate_circle`**: Ensures that a circular mesh is generated with the correct number of boundary vertices and triangular cells.

3. **Validation of Mesh Structure**:
   - Each test verifies the internal structure of the mesh by checking counts of specific entities (e.g., vertices, cells). This validation confirms that imported and generated meshes have the expected topology.
   - The tests use helper methods (e.g., `count_entities`) to count specific entities in the mesh, providing a straightforward way to validate each mesh’s structural integrity.

**Current Limitations**:
- **Limited Edge Case Testing**: The tests currently cover expected inputs and formats, but there is limited testing for edge cases such as malformed Gmsh files, empty meshes, or invalid parameters for mesh generation.
- **Sequential Execution**: Tests are run sequentially, which can be slow for larger mesh files or complex generation tests.
- **Lack of Detailed Error Messaging**: Test failures do not always provide specific context for why a test failed, which can complicate troubleshooting.

**Recommended Enhancements**:
1. **Expand Edge Case Coverage**:
   - Add tests for malformed and incomplete Gmsh files to verify that `GmshParser` correctly identifies and handles errors in external data.
   - Include validation tests for degenerate cases in `MeshGenerator`, such as zero or negative dimensions, to ensure robustness against invalid input parameters.

2. **Parallel Testing and Benchmarks**:
   - Integrate Rust’s parallel testing capabilities (enabled with `cargo test -- --test-threads N`) to speed up test execution, especially beneficial for large meshes and performance-intensive generation tests.
   - Add benchmarks using `cargo bench` for `MeshGenerator` to profile the generation speed and identify performance bottlenecks for high-resolution or 3D meshes.

3. **Enhanced Error Reporting**:
   - Improve test assertions with detailed messages that specify the expected vs. actual values upon failure. This enhancement would provide clearer diagnostics, allowing developers to quickly identify and address issues in the test outputs.

4. **Increased Code Coverage**:
   - Add tests for additional shapes as they are integrated into `MeshGenerator`, ensuring that each new shape generation function is thoroughly tested and validated against expected outputs.

By expanding the test coverage, implementing parallel execution, and enhancing error reporting, the `tests` module will provide a more comprehensive and efficient framework for ensuring the stability and correctness of the `input_output` module as HYDRA evolves.

---

