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

Welcome to the user's guide for the `Domain` module of the Hydra computational framework. This module is central to managing mesh-related data and operations essential for finite volume (FVM) or finite element simulations. It provides robust abstractions for representing mesh entities—such as vertices, edges, faces, and cells—and their relationships, as well as the ability to store data on these entities.

---

## **2. Overview of the Domain Module**

The `Domain` module is designed for unstructured 2D/3D meshes typically used in computational fluid dynamics (CFD), geophysical modeling, and similar numerical simulations. It offers:

- **Mesh Entity Representation** (`MeshEntity`).
- **Mesh Connectivity Management** via a topological data structure (`Sieve`).
- **Data Association** using a generic mapping structure (`Section<T>`).
- **Domain Overlap & Boundary Data** handling in partitioned contexts (`Overlap` & `Delta`).
- **Hierarchical Mesh Refinement** for adaptivity (`MeshNode`).
- **Validation & Reordering** capabilities for correctness and performance.

---

## **3. Core Structures**

### MeshEntity Enum

```rust
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
pub enum MeshEntity {
    Vertex(usize),
    Edge(usize),
    Face(usize),
    Cell(usize),
}
```

**Purpose**: Represents a **single mesh entity**, uniquely identified by an integer.

- **Key Methods**:
  - `get_id() -> usize`: Returns the unique ID.
  - `get_entity_type() -> &str`: Returns `"Vertex"`, `"Edge"`, `"Face"`, or `"Cell"`.
  - `with_id(new_id: usize) -> Self`: Clones the entity with a new ID.

**Example**:
```rust
use hydra::domain::MeshEntity;

let vertex = MeshEntity::Vertex(1);
println!("Vertex ID: {}", vertex.get_id()); // 1
println!("Entity type: {}", vertex.get_entity_type()); // "Vertex"
```

### Arrow Struct

```rust
pub struct Arrow {
    pub from: MeshEntity,
    pub to: MeshEntity,
}
```

**Purpose**: Encodes a **directed** relationship (`from -> to`) between two `MeshEntity` objects.

- **Key Methods**:
  - `Arrow::new(from, to) -> Arrow`: Constructs a new arrow.
  - `get_relation() -> (&MeshEntity, &MeshEntity)`: Returns references to `from` and `to`.
  - `set_from(...)` / `set_to(...)`: Mutably updates the endpoints.

**Example**:
```rust
use hydra::domain::{MeshEntity, Arrow};

let from = MeshEntity::Vertex(1);
let to   = MeshEntity::Edge(2);
let arrow = Arrow::new(from, to);
let (start, end) = arrow.get_relation();
```

---

## **4. Mesh Connectivity with Sieve**

### Sieve Structure

```rust
#[derive(Clone, Debug)]
pub struct Sieve {
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}
```

**Purpose**: A **thread-safe** adjacency structure that organizes **directed relationships** between mesh entities.

- Each key is a `MeshEntity`.
- The value is a `DashMap<MeshEntity, ()>`, representing all entities that the key points to.

### Core Methods

1. **Constructing a Sieve**  
   ```rust
   let sieve = Sieve::new();
   ```
2. **Adding a Relationship**  
   ```rust
   let v = MeshEntity::Vertex(1);
   let e = MeshEntity::Edge(1);
   sieve.add_arrow(v, e);
   ```
   This states "`v` has an arrow to `e`".
3. **Topological Queries**  
   - **cone(&entity) -> Option<Vec<MeshEntity>>**: Direct “children” of `entity`.
   - **support(&entity) -> Vec<MeshEntity>**: All entities that point **to** `entity`.
   - **closure(&entity) -> DashMap<MeshEntity, ()>**: Entity plus the recursive union of its cone(s).
   - **star(&entity) -> DashMap<MeshEntity, ()>**: Entity plus all who cover it plus its cone.
   - **meet(&p, &q)**: Intersection of `closure(p)` and `closure(q)`.
   - **join(&p, &q)**: Union of `star(p)` and `star(q)`.
4. **Parallel Processing**  
   ```rust
   sieve.par_for_each_adjacent(|(entity, neighbors)| {
       // 'entity' is the key MeshEntity
       // 'neighbors' is a Vec<MeshEntity> of adjacency
   });
   ```
5. **Exporting**  
   - `to_adjacency_map() -> FxHashMap<MeshEntity, Vec<MeshEntity>>`: Converts the internal `DashMap` to a standard map.

---

## **5. Stratification of Mesh Entities**

```rust
pub fn stratify(&self) -> DashMap<usize, Vec<MeshEntity>>
```

**Purpose**: Categorizes entities by their **dimension**:
- **0** -> Vertices  
- **1** -> Edges  
- **2** -> Faces  
- **3** -> Cells  

**Example**:
```rust
let strata = sieve.stratify();
if let Some(verts) = strata.get(&0) {
    println!("Number of vertices: {}", verts.len());
}
```

---

## **6. Filling Missing Entities**

```rust
pub fn fill_missing_entities(&self)
```

- **Intent**: In **2D**, automatically create **edges** that are not yet in the mesh but implied by each cell’s vertices.
- **Note**: The current implementation **focuses on 2D** (inferring edges). Extending for 3D faces would require custom logic.

**Usage**:
```rust
// For each Cell, connect consecutive vertices to form Edges.
sieve.fill_missing_entities();
```

---

## **7. Data Association with Section**

### Section Struct

```rust
#[derive(Clone, Debug)]
pub struct Section<T> {
    pub data: DashMap<MeshEntity, T>,
}
```

**Purpose**: Stores a mapping of **`MeshEntity` -> `T`** (e.g., scalars, vectors, or custom types).

- **Key Methods**:
  - `new() -> Self`
  - `set_data(entity, value)`
  - `restrict(entity) -> Option<T>`: Retrieves a clone of the data for `entity`.
  - `entities() -> Vec<MeshEntity>`: Lists all entities for which data is stored.
  - `clear()`: Removes all data entries.
  - `update_with_derivative(...)`, `scale(...)`: Additional numeric utilities if `T` implements certain traits.

**Example**:
```rust
use hydra::domain::Section;

let section = Section::<f64>::new();
section.set_data(MeshEntity::Vertex(1), 100.0);

if let Some(val) = section.restrict(&MeshEntity::Vertex(1)) {
    println!("Value at Vertex(1): {}", val);
}
```

### Parallel Data Updates

```rust
pub fn parallel_update<F>(&self, update_fn: F)
where
    F: Fn(&mut T) + Sync + Send
```

**Usage**:
```rust
section.parallel_update(|value| {
    *value *= 2.0;
});
```
All entries in `data` are updated **in parallel** using Rayon.

---

## **8. Domain Overlap Management**

### Overlap Struct

```rust
pub struct Overlap {
    pub local_entities: Arc<DashMap<MeshEntity, ()>>,
    pub ghost_entities: Arc<DashMap<MeshEntity, ()>>,
}
```

**Purpose**: Tracks **partitioned** data ownership:
- **Local**: Entities fully owned by the current partition.
- **Ghost**: Entities shared with or owned by another partition, but needed locally for overlap consistency.

- **Key Methods**:
  - `add_local_entity(entity)`
  - `add_ghost_entity(entity)`
  - `is_local(entity) -> bool`
  - `is_ghost(entity) -> bool`
  - `merge(&other)`: Merges another `Overlap` into this one.

**Example**:
```rust
let overlap = Overlap::new();
overlap.add_local_entity(MeshEntity::Cell(10));
overlap.add_ghost_entity(MeshEntity::Cell(20));

assert!(overlap.is_local(&MeshEntity::Cell(10)));
assert!(overlap.is_ghost(&MeshEntity::Cell(20)));
```

### Delta Struct

```rust
pub struct Delta<T> {
    pub data: Arc<DashMap<MeshEntity, T>>,
}
```

**Purpose**: Holds transformation or “delta” data for entities, often used in ghost exchanges.

- **Key Methods**:
  - `set_data(entity, value)`
  - `get_data(entity) -> Option<T>`
  - `remove_data(entity) -> Option<T>`
  - `apply(|entity, value| ...)`: Read-only iteration over all data.
  - `merge(&other)`: Overwrites or adds data from `other`.

---

## **9. Mesh Management**

### Mesh Struct

```rust
#[derive(Clone, Debug)]
pub struct Mesh {
    pub sieve: Arc<Sieve>,
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,
}
```

**Purpose**: The **central** object for:
- Storing a `Sieve` for connectivity.
- Tracking a set of mesh entities.
- Storing (vertex_id -> coordinates).
- Handling boundary data channels for distributed or parallel usage.

**Typical Construction**:
```rust
let mut mesh = Mesh::new(); // Creates default channels, sets, etc.
```

### Entities Management

- **Adding Entities**:
  ```rust
  mesh.add_entity(MeshEntity::Vertex(1));
  ```
- **Relationships** (Delegates to `Sieve`):
  ```rust
  mesh.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));
  ```
- **Vertex Coordinates**:
  ```rust
  mesh.set_vertex_coordinates(1, [0.0, 1.0, 2.0]);
  let coords = mesh.get_vertex_coordinates(1); // Option<[f64;3]>
  ```
- **Filtering**:
  ```rust
  let all_cells = mesh.get_cells(); // Collects all MeshEntity::Cell
  let all_vertices = mesh.get_vetices();
  ```

### Boundary Handling

- **Channels**: `set_boundary_channels(sender, receiver)`
- **Synchronization**: `sync_boundary_data()`
  - Sends local boundary data if `boundary_data_sender` is configured.
  - Receives updates via `boundary_data_receiver`.

### Geometry Calculations

- **Methods**:
  - `get_face_area(face) -> Option<f64>`
  - `get_cell_centroid(cell) -> [f64; 3]`
  - `get_distance_between_cells(cell_i, cell_j) -> f64`
  - `get_face_normal(face, reference_cell) -> Option<Vector3>`

**Example**:
```rust
let centroid = mesh.get_cell_centroid(&MeshEntity::Cell(10));
let dist = mesh.get_distance_between_cells(&MeshEntity::Cell(10), &MeshEntity::Cell(20));
```

### Hierarchical Mesh

```rust
pub enum MeshNode<T> {
    Leaf(T),
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>,
    },
}
```

- **Refinement** (`refine(...)`): Converts a leaf to a branch.
- **Coarsening** (`coarsen()`): Branch back to leaf.
- **Iteration** (`leaf_iter()`): Traverses only leaf nodes.

### Topology Validation

```rust
pub struct TopologyValidation<'a> {
    sieve: &'a Sieve,
    entities: &'a Arc<RwLock<FxHashSet<MeshEntity>>>,
}
```

- **Methods**:
  - `validate_connectivity() -> bool`
  - `validate_unique_relationships() -> bool`

**Usage**:
```rust
let tv = TopologyValidation::new(&mesh);
assert!(tv.validate_connectivity());
```

### Reordering Algorithms

- **Cuthill-McKee** or **Reverse Cuthill-McKee**:
  ```rust
  let ordered_entities = cuthill_mckee(&entities, &adjacency_map);
  mesh.apply_reordering(&ordered_entities.iter().map(|e| e.get_id()).collect::<Vec<_>>());
  ```
- **Morton (Z-order) Reordering**:
  ```rust
  let mut elements_2d = vec![(0u32,0u32), (1,2), ...];
  mesh.reorder_by_morton_order(&mut elements_2d);
  ```

---

## **10. Testing and Validation**

### Unit Testing

- **MeshEntity**: Confirm correct IDs, entity types, etc.
- **Sieve**: Ensure `add_arrow`, `cone`, `closure`, etc. behave correctly.
- **Section**: Confirm `set_data`, `restrict`, `parallel_update` logic.

**Example**:
```rust
#[test]
fn test_mesh_entity_ids() {
    let v = MeshEntity::Vertex(42);
    assert_eq!(v.get_id(), 42);
}
```

### Integration Testing

- **Mesh Construction**: Populate a `Mesh` with vertices, edges, cells, and verify consistency.
- **Boundary Data**: Test boundary synchronization with multiple senders/receivers.
- **Reordering**: Check if performance or bandwidth improvements are observed.

---

## **11. Best Practices**

### Efficient Mesh Management

1. **Parallelization**: Use built-in parallel methods (e.g., `par_for_each_entity`, `Section::parallel_update`).
2. **Avoid Duplication**: Ensure unique IDs across `MeshEntity` within a single mesh.
3. **Regular Validation**: After adding relationships or performing reordering, use `TopologyValidation`.

### Performance Optimization

1. **Reordering**: Apply RCM or Morton order to improve memory locality in solvers.
2. **Adaptive Refinement**: Use hierarchical (`MeshNode`) structures to refine mesh regions of interest.
3. **DashMap** Efficiency**: The `DashMap` structures in `Sieve`, `Section`, and `Overlap` allow concurrent reads/writes.

### Handling Complex Mesh Structures

- **Faces vs. Edges**: Keep track of dimension differences; for 2D, edges are the boundary of cells, while in 3D you may also need faces.
- **Partitioning**: Use `Overlap` to differentiate local vs. ghost entities in multi-part setups.

---

## **12. Advanced Configurations and Extensibility**

- **Custom Entity Types**: Extend `MeshEntity` if you have specialized region types (e.g., `PolyCell`).
- **More Validation**: Add advanced rules in `TopologyValidation` or create new modules (`geometry_validation.rs` is an example).
- **Parallel/Distributed**: Integrate more sophisticated boundary data exchange patterns or custom HPC frameworks.

---

## **13. Conclusion**

The `Domain` module in Hydra offers a **powerful**, **thread-safe** framework for building, connecting, refining, and validating mesh data. By combining the `Sieve` for topological relationships, `Section` for data storage, and the high-level `Mesh` API (plus boundary/overlap utilities), you can manage a wide range of mesh configurations—2D, 3D, static, or adaptive—under one cohesive architecture. Proper use of the included validation and reordering features ensures both **correctness** and **performance** in large-scale numerical simulations.