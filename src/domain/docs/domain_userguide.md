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