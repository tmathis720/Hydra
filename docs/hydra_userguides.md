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

---

Below is an **updated** `Geometry` module user guide that more accurately reflects the **current source code** and the set of shapes (both 2D and 3D) supported in Hydra. All the new additions—such as the `Edge` face shape, triangular cells, and advanced features—are incorporated to ensure completeness and correctness.

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
   - [Edge](#edge)  
   - [Triangle](#triangle)  
   - [Quadrilateral](#quadrilateral)  
   - [Tetrahedron](#tetrahedron)  
   - [Hexahedron](#hexahedron)  
   - [Prism](#prism)  
   - [Pyramid](#pyramid)  
8. [Caching and Performance Optimization](#8-caching-and-performance-optimization)  
9. [Advanced Usage](#9-advanced-usage)  
   - [Updating All Cell Volumes](#updating-all-cell-volumes)  
   - [Total Volume and Centroid](#total-volume-and-centroid)  
10. [Best Practices](#10-best-practices)  
11. [Testing and Validation](#11-testing-and-validation)  
12. [Conclusion](#12-conclusion)

---

## **1. Introduction**

Welcome to the **`Geometry`** module user guide for Hydra. This module is responsible for **storing** and **computing** all geometric properties of mesh entities. Typical tasks include:

- Setting or updating **vertex** coordinates in 3D space.  
- Calculating **centroids**, **areas**, **volumes**, and **normals** for various shapes (both 2D and 3D).  
- Handling specialized shapes, such as **tetrahedrons**, **hexahedrons**, **prisms**, **pyramids**, and more.  
- Maintaining a **cache** of computed properties for improved performance.

---

## **2. Overview of the Geometry Module**

The `Geometry` module is integrated with Hydra’s **`domain`** layer (notably, with `Mesh` and `MeshEntity`) to:

- Retrieve a cell’s or face’s vertices via `Mesh`.  
- Identify the shape (`CellShape` or `FaceShape`).  
- Compute shape-specific properties (e.g., volume for a tetrahedron, area for a quadrilateral face).  
- Optionally **cache** results (centroids, volumes, normals) to speed up subsequent calculations.  

Supported shapes:

- **2D (FaceShape)**: `Edge`, `Triangle`, `Quadrilateral`  
- **3D (CellShape)**: `Triangle` (degenerate 3D?), `Quadrilateral`, `Tetrahedron`, `Hexahedron`, `Prism`, `Pyramid`.  

*(Note that `Triangle` and `Quadrilateral` can appear as either cells or faces, depending on the mesh topology.)*

---

## **3. Core Structures**

### Geometry Struct

```rust
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        
    pub cell_centroids: Vec<[f64; 3]>,  
    pub cell_volumes: Vec<f64>,         
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>,
}
```

- **Stores**:
  - A list of **vertex** coordinates.
  - Computed arrays for **cell centroids** and **cell volumes**.
  - A **thread-safe cache** (`Mutex<FxHashMap<...>>`) for geometric properties keyed by entity ID.

### GeometryCache Struct

```rust
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,
}
```

- **Purpose**: Holds optional precomputed fields—`volume`, `centroid`, `area`, `normal`.  
- Populated **lazily** as geometry calculations occur.  
- If a shape changes or a vertex is updated, the cache can be **invalidated**.

### CellShape Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellShape {
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}
```

- Identifies **3D cells** (though `Triangle`/`Quadrilateral` can be degenerate 3D surfaces or 2D domains in certain contexts).
- The geometry code references these shapes to decide which specialized formula to use for **centroids** or **volumes**.

### FaceShape Enum

```rust
#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Edge,
    Triangle,
    Quadrilateral,
}
```

- Identifies **2D faces**:
  - **Edge**: 2-vertex boundary or line segment (common in 2D meshes).
  - **Triangle**: 3 vertices.
  - **Quadrilateral**: 4 vertices.

---

## **4. Mesh Entities and Connectivity**

- `Geometry` interacts closely with the **`Mesh`** structure in `domain::mesh`.  
- **Example**:
  - `Mesh::get_cell_vertices(cell) -> Vec<[f64; 3]>`  
  - `Mesh::get_face_vertices(face) -> Vec<[f64; 3]>`
- `Geometry` uses these vertices plus a shape classification (e.g., `CellShape::Tetrahedron`) to compute properties.

---

## **5. Working with Vertices**

### Adding and Updating Vertices

```rust
pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]);
```

- If `vertex_index >= self.vertices.len()`, the `vertices` vector is **resized** automatically.
- A change here **invalidates** the geometry cache to prevent stale calculations.

```rust
let mut geometry = Geometry::new();
geometry.set_vertex(0, [1.0, 2.0, 3.0]); // If index=0 is new, a new vertex is added
geometry.set_vertex(10, [5.0, 1.0, 0.0]); // Resizes the vector up to index=10
```

---

## **6. Computing Geometric Properties**

The `Geometry` struct provides high-level methods to compute cell or face properties. It also has direct subroutines for shape-specific computations.

### Cell Centroids

```rust
pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3];
```

- Determines the cell’s shape (via `mesh.get_cell_shape(cell)`) and calls the relevant shape method:
  - **Triangle** -> `compute_triangle_centroid`
  - **Quadrilateral** -> `compute_quadrilateral_centroid`
  - **Tetrahedron** -> `compute_tetrahedron_centroid`
  - **Hexahedron** -> `compute_hexahedron_centroid`
  - **Prism** -> `compute_prism_centroid`
  - **Pyramid** -> `compute_pyramid_centroid`
- **Cached** under that cell’s ID.

### Cell Volumes

```rust
pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64;
```

- Similar logic—checks shape, then calls e.g., `compute_tetrahedron_volume`, etc.
- If the shape is `Triangle` or `Quadrilateral` but used as a cell, it interprets that 2D shape’s area as a “volume” in a degenerate sense.

### Face Areas

```rust
pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64;
```

- Distinguishes:
  - **Edge**: The area is effectively the **length** of the line segment.
  - **Triangle**: Uses `compute_triangle_area`.
  - **Quadrilateral**: Splits into two triangles, sums areas.
- Stores the result in the cache for `face_id`.

### Face Centroids

```rust
pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3];
```

- If `Edge`, midpoint of the two vertices.
- If `Triangle`, average of the three vertices.
- If `Quadrilateral`, average of the four vertices.

### Face Normals

```rust
pub fn compute_face_normal(
    &mut self,
    mesh: &Mesh,
    face: &MeshEntity,
    cell: &MeshEntity
) -> Option<[f64; 3]>;
```

- Checks `face` vertex count:  
  - 2 -> `FaceShape::Edge` -> `compute_edge_normal`  
  - 3 -> `Triangle` -> `compute_triangle_normal`  
  - 4 -> `Quadrilateral` -> `compute_quadrilateral_normal`  
- Returns a `Vector3` if supported. 2D “edge” normal is a 90-degree rotation in XY plane.

### Distance Calculations

```rust
pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64;
```

- Basic Euclidean distance:
  \[
    \sqrt{(p1_x - p2_x)^2 + (p1_y - p2_y)^2 + (p1_z - p2_z)^2}
  \]

---

## **7. Shape-Specific Computations**

The module internally delegates to shape-specific methods. For reference, each shape has its centroid, area, or volume logic:

### Edge

- **`compute_edge_length`**  
  - Expects exactly 2 vertices and calculates the distance.  
- **`compute_edge_midpoint`**  
  - Averages the two vertex coordinates to get midpoint.  
- **`compute_edge_normal`**  
  - Rotates the edge vector 90° in 2D. Normal is `[ -dy, dx, 0 ]` (normalized).

### Triangle

**File**: `triangle.rs`  
- **Centroid** (`compute_triangle_centroid`)  
  - Average of the three vertices.  
- **Area** (`compute_triangle_area`)  
  - Half of cross product magnitude: \(\frac{1}{2} |(v1 - v0) \times (v2 - v0)|\).  
- **Normal** (`compute_triangle_normal`)  
  - Cross product of edges `(v1 - v0)` and `(v2 - v0)`.

### Quadrilateral

**File**: `quadrilateral.rs`  
- **Area** (`compute_quadrilateral_area`)  
  - Split into two triangles, sum their areas.  
- **Centroid** (`compute_quadrilateral_centroid`)  
  - Average of the four vertices.  
- **Normal** (`compute_quadrilateral_normal`)  
  - Compute normals for the two triangles formed, then average them.

### Tetrahedron

**File**: `tetrahedron.rs`  
- **Centroid** (`compute_tetrahedron_centroid`)  
  - Average of the four vertices.  
- **Volume** (`compute_tetrahedron_volume`)  
  - Determinant-based formula (1/6 of the absolute determinant of the edge vectors).

### Hexahedron

**File**: `hexahedron.rs`  
- **Centroid** (`compute_hexahedron_centroid`)  
  - Average of all 8 vertex coordinates.  
- **Volume** (`compute_hexahedron_volume`)  
  - Decomposes the hexahedron into 5 tetrahedrons, sums volumes.

### Prism

**File**: `prism.rs`  
- **Centroid** (`compute_prism_centroid`)  
  - Split top and bottom triangles (3 vertices each). Compute each triangle’s centroid, then average the two centroids.  
- **Volume** (`compute_prism_volume`)  
  - Base area (triangle) * height (distance between top and bottom triangle centroids).

### Pyramid

**File**: `pyramid.rs`  
- **Centroid** (`compute_pyramid_centroid`)  
  - If triangular base (4 vertices total), treat as a tetrahedron. If square base (5 vertices), split base into two triangles. Weighted average apex & base.  
- **Volume** (`compute_pyramid_volume`)  
  - If 5 vertices, decompose into 2 tetrahedrons. If 4 vertices, 1 tetrahedron.

---

## **8. Caching and Performance Optimization**

The geometry module uses a **`cache: Mutex<FxHashMap<usize, GeometryCache>>`**.  
- **Key**: usually the entity’s integer ID (e.g., from `MeshEntity::get_id()`).  
- **Value**: a `GeometryCache` with optional `volume`, `centroid`, `area`, `normal`.  
- On any **vertex update** or major geometry change, `invalidate_cache()` is called.  

**Parallel** usage:  
- The module is parallel-friendly in many routines (e.g., `update_all_cell_volumes` uses Rayon).

---

## **9. Advanced Usage**

### Updating All Cell Volumes

```rust
pub fn update_all_cell_volumes(&mut self, mesh: &Mesh);
```

- **Parallel**: For each cell in the mesh, a temporary `Geometry` is created, then `compute_cell_volume(...)` is called.  
- Stores all new volumes in `self.cell_volumes`.

### Total Volume and Centroid

- **`compute_total_volume()`**: Summation of `self.cell_volumes`.  
- **`compute_total_centroid()`**: Averages all `self.cell_centroids`.

These rely on the arrays in `Geometry` being **up-to-date** (via your calls to compute each cell’s centroid/volume or `update_all_cell_volumes`).

---

## **10. Best Practices**

1. **Invalidate Cache**: Whenever you modify **vertices** in the `Geometry` struct, call or rely on the `invalidate_cache()` method to keep computations accurate.  
2. **Parallelization**: Leverage **Rayon**-based methods (like `update_all_cell_volumes`) for large meshes.  
3. **Shape Verification**: Each shape method uses `assert!` to confirm the correct vertex count—this is a good safeguard.  
4. **Check Return Types**: Some face computations (e.g., area of an `Edge`) yield a “length” effectively, so plan your usage accordingly.

---

## **11. Testing and Validation**

- The code includes **unit tests** for each shape, verifying:
  - Tetrahedron volumes (e.g., a simple corner tetrahedron of volume `1/6`).  
  - Triangular or quadrilateral face areas.  
  - Centroids of prisms, pyramids, etc.  
- **Degenerate Cases**: For zero-area or zero-volume shapes, expect the cross product or determinant to return zero.  
- Use **floating-point tolerances** for real-world test verifications.

**Example**:

```rust
#[test]
fn test_tetrahedron_volume() {
    let geometry = Geometry::new();
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let vol = geometry.compute_tetrahedron_volume(&vertices);
    assert!((vol - 1.0/6.0).abs() < 1e-10);
}
```

---

## **12. Conclusion**

The **`Geometry`** module forms the backbone of Hydra’s spatial computations, enabling you to:

1. **Store** and **update** vertex coordinates.  
2. **Compute** shape-based metrics like **centroids**, **areas**, **volumes**, and **normals**.  
3. **Cache** these results for performance.  
4. **Leverage** parallel updates for large meshes or frequent recomputations.

Understanding these APIs and best practices ensures that your numerical simulations run accurately and efficiently, whether you’re dealing with simple 2D edges and triangles or complex 3D solids like tetrahedrons, prisms, or hexahedrons.

---

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

---

# Hydra `Equation` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Equation Module](#2-overview-of-the-equation-module)  
3. [Core Components](#3-core-components)  
   - [PhysicalEquation Trait](#physicalequation-trait)  
   - [Fields and Fluxes](#fields-and-fluxes)  
   - [EquationManager](#equationmanager)  
4. [Equation Submodules](#4-equation-submodules)  
   - [MomentumEquation](#momentumequation)  
   - [EnergyEquation](#energyequation)  
   - [Turbulence Models](#turbulence-models)  
   - [Flux Limiters](#flux-limiters)  
   - [Gradient Computation](#gradient-computation)  
   - [Reconstruction Methods](#reconstruction-methods)  
5. [Using the Equation Module](#5-using-the-equation-module)  
   - [Defining a Physical Equation](#defining-a-physical-equation)  
   - [Managing Equations, Fields, and Fluxes](#managing-equations-fields-and-fluxes)  
   - [Integration with Other Modules](#integration-with-other-modules)  
6. [Best Practices](#6-best-practices)  
7. [Conclusion](#7-conclusion)

---

## **1. Introduction**

The **`Equation`** module in Hydra coordinates the **physical equations** governing fluid or other physical processes. It ties together:

- **Equation Definitions**: (e.g., **Momentum**, **Energy**, **Turbulence**).  
- **Fields**: Data structures storing state variables like velocity, pressure, temperature.  
- **Fluxes**: Computed results from each equation used to update the fields.  
- **Reconstruction**, **Gradients**, and **Flux Limiters**: Tools for higher-order accuracy and numerical stability.

By leveraging the **`PhysicalEquation`** trait, the module supports adding new physics while tapping into Hydra’s frameworks for domain geometry, boundary conditions, and solver time-stepping.

---

## **2. Overview of the Equation Module**

**Location**: `src/equation/`

Submodules:

- **`equation/fields.rs`**: Contains `Fields` (various field data) and `Fluxes` (various flux data) plus the `UpdateState` trait.  
- **`equation/equation.rs`**: An example utility class for flux calculations.  
- **`equation/manager.rs`**: The `EquationManager` that orchestrates multiple equations with time stepping.  
- **`energy_equation.rs`, `momentum_equation.rs`, `turbulence_models.rs`**: Implementation examples of physical equations.  
- **`flux_limiter`, `gradient`, `reconstruction`**: Supporting tools for advanced numerical methods.

---

## **3. Core Components**

### PhysicalEquation Trait

```rust
pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}
```

Defines a method `assemble(...)` that modifies or adds to **fluxes** based on **fields** and domain data. It's the core interface for PDE-based computations:

1. **domain**: The mesh describing geometry.  
2. **fields**: Current field data (e.g., velocity, pressure).  
3. **fluxes**: Outgoing flux container updated by the equation.  
4. **boundary_handler**: For applying boundary conditions.  
5. **current_time**: For time-dependent BC or physics.

**Implementers** might:

- Calculate momentum fluxes (momentum_equation).
- Calculate energy fluxes (energy_equation).
- Or other physical processes (turbulence models, chemical reactions, etc.).

### Fields and Fluxes

In **`fields.rs`**:

- **`Fields`**: Contains multiple **scalar_fields**, **vector_fields**, **tensor_fields** (backed by Hydra’s `Section<T>`).
  - `get_scalar_field_value(...)`, `set_scalar_field_value(...)`, similarly for vectors.  
  - `update_from_fluxes(...)`: A convenient method to apply flux changes to the fields.

- **`Fluxes`**: Gathers flux data (e.g., momentum_fluxes, energy_fluxes) in `Section<T>` objects.  
  - e.g., `add_momentum_flux(...)`, `add_energy_flux(...)`, `add_turbulence_flux(...)`.

**`UpdateState`** trait: Let fields do a “time step update” (`update_state`), compute difference, or measure norm.

### EquationManager

In **`manager.rs`**:

```rust
pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
    time_stepper: Box<dyn TimeStepper<Self>>,
    domain: Arc<RwLock<Mesh>>,
    boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
}
```

- Maintains a list of `PhysicalEquation`s.  
- Has a reference to a time-stepper (which uses it as a `TimeDependentProblem`):
  - `EquationManager` itself implements `TimeDependentProblem` so it can produce flux derivatives for time stepping.  
- **`assemble_all(...)`**: Runs `assemble(...)` for each equation over `fields -> fluxes`.  
- **`step(...)`**: Delegates a time-step to the assigned `time_stepper`, which calls back into `compute_rhs(...)` etc.

---

## **4. Equation Submodules**

### MomentumEquation

File: **`momentum_equation.rs`**  
Implements `PhysicalEquation` for incompressible (or general) momentum laws:

- **`assemble(...)`** calls `calculate_momentum_fluxes(...)`:
  - Collect velocity, pressure data from `fields`.
  - Compute fluxes for convection, diffusion, pressure.  
  - Apply BC adjustments.  
  - Store final flux in `fluxes.momentum_fluxes`.  

It’s a template for more advanced Navier-Stokes or specialized momentum systems.

### EnergyEquation

File: **`energy_equation.rs`**  
Implements `PhysicalEquation` for thermal energy or enthalpy equation:

- **`assemble(...)`** calls `calculate_energy_fluxes(...)`:
  - Evaluates conduction + convection flux in a manner similar to momentum.
  - BCs can impose temperature constraints or flux.  

### Turbulence Models

- **`turbulence_models.rs`**: A trait `TurbulenceModel` plus an example `GOTMModel`.
- They typically define how to compute **eddy diffusivity**, **eddy viscosity**, or extra scalar fluxes (k-ε, RANS, etc.).

**Note**: They also implement `PhysicalEquation` if they produce fluxes for e.g. TKE or dissipation.

### Flux Limiters

Directory: **`flux_limiter/`**  
- Defines a `FluxLimiter` trait with implementations: **Minmod**, **Superbee**, **VanLeer**, etc.  
- Typically used in reconstructions or slope-limited finite volume methods to maintain stability near discontinuities.

### Gradient Computation

Directory: **`gradient/`**  
- **`GradientMethod`** trait with **`FiniteVolumeGradient`** and **`LeastSquaresGradient`**.  
- A `Gradient` wrapper that calls these methods for each cell.  
- Allows the module to compute \(\nabla \phi\) (scalar gradient) for advanced flux computations or higher-order PDE solvers.

### Reconstruction Methods

Directory: **`reconstruction/`**  
- **`ReconstructionMethod`** trait for face reconstruction.  
- Implementations:
  - **`LinearReconstruction`** (simple linear interpolation).  
  - **`WENOReconstruction`**, **`PPMReconstruction`**, etc. for higher-order schemes.  
- The PDE solvers (like momentum/energy) can call these to get face values from cell-centered data plus gradients.

---

## **5. Using the Equation Module**

### Defining a Physical Equation

1. **Implement** `PhysicalEquation`:
   ```rust
   pub struct MyEquation;

   impl PhysicalEquation for MyEquation {
       fn assemble(
           &self,
           domain: &Mesh,
           fields: &Fields,
           fluxes: &mut Fluxes,
           boundary_handler: &BoundaryConditionHandler,
           current_time: f64,
       ) {
           // 1) Possibly compute gradient or field reconstructions.
           // 2) Evaluate fluxes (convective, diffusive, source).
           // 3) Use boundary_handler to apply BC modifications.
           // 4) Insert final flux into fluxes (e.g. fluxes.momentum_fluxes).
       }
   }
   ```

2. Add custom logic for your PDE or equations. The momentum and energy equation files are good references.

### Managing Equations, Fields, and Fluxes

- **Create** an `EquationManager` with domain, boundary handler, and time stepper:
  ```rust
  let manager = EquationManager::new(
      time_stepper,  // e.g., a Box<dyn TimeStepper<EquationManager>>
      domain,
      boundary_handler,
  );
  ```
- **Add** `PhysicalEquation` objects:
  ```rust
  manager.add_equation(MomentumEquation::new());
  manager.add_equation(EnergyEquation::new(thermal_conductivity));
  ```
- **Fields**: Usually start with an initial `Fields` object.  
- **Time stepping**: If the manager is used in synergy with Hydra’s time stepping, call `manager.step(&mut fields)` each iteration. This internally calls `EquationManager::compute_rhs(...)` which does `assemble_all(...)` -> fluxes -> derivative.

### Integration with Other Modules

- **Mesh**: The domain geometry from **`domain`** module.  
- **Boundary Conditions**: The **`BoundaryConditionHandler`** from **`boundary`** module.  
- **Time Stepping**: The **`TimeStepper`** trait from **`time_stepping`**.  
- **Linear Algebra**: If a PDE step requires a matrix solve, call `solve_linear_system(...)` with Hydra’s `Matrix` interface.

---

## **6. Best Practices**

1. **Keep PDE Logic in `PhysicalEquation`**: Encapsulate each PDE’s flux computations in a separate struct.  
2. **Use `EquationManager`**: Orchestrate multiple PDEs (momentum, energy, turbulence) to produce combined flux data.  
3. **Extend with `FluxLimiter`**: If high-order or slope-limited finite volume methods are used, incorporate limiters in your reconstruction steps.  
4. **Check Boundary Conditions**: The boundary handler must provide the correct BC for each face or cell boundary.  
5. **Parallelization**: Hydra’s module design allows concurrency at the level of flux assembly if carefully managed.  

---

## **7. Conclusion**

The **`Equation`** module is a crucial piece of Hydra’s PDE-solving architecture:

- **`PhysicalEquation`** provides a uniform interface for PDE flux assembly.  
- **`Fields`** and **`Fluxes`** store, update, and unify the data needed for continuous PDE integration over the mesh.  
- **`EquationManager`** merges these equations with time stepping, letting Hydra evolve complex multi-physics systems in a consistent manner.  
- **Supporting submodules** (momentum, energy, turbulence, reconstruction, flux limiting, gradient) demonstrate extensible approaches for real-world fluid or multi-physics simulations.

By following this modular approach, you can **add new physics**, **customize flux calculations**, or **switch reconstruction/gradient methods** to tailor Hydra to your numerical requirements.

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
     - [SparseMatrix](#sparsematrix)  
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

Welcome to the user's guide for the **`Linear Algebra`** module of the Hydra computational framework. This module provides the fundamental linear algebra functionalities used in **numerical simulations**, **finite volume/element methods**, and other scientific computing tasks. Key features include:

- **Vector operations**: Dot product, norms, scalings, element-wise operations, etc.  
- **Matrix operations**: Matrix-vector multiplication, trace, Frobenius norm, resizing, etc.  
- **Abstract Traits** and **Concrete Implementations**: Flexible trait system for different data structures (`Vec<f64>`, `faer::Mat<f64>`, and a `SparseMatrix`).  

---

## **2. Overview of the Linear Algebra Module**

The **`linalg`** module separates functionality into **vector** and **matrix** submodules:

- **Vectors** (`linalg::vector`)  
  - `Vector` trait plus implementations for standard Rust vectors (`Vec<f64>`) and `faer::Mat<f64>` (treated as a column vector).
  - A **`VectorBuilder`** utility to create and resize vectors.
- **Matrices** (`linalg::matrix`)  
  - `Matrix` trait plus specialized traits (`MatrixOperations`, `ExtendedMatrixOperations`) for constructing/resizing.
  - Implementations for `faer::Mat<f64>` (dense) and a custom `SparseMatrix`.
  - A **`MatrixBuilder`** utility to create, resize, and integrate with preconditioners.

This design allows Hydra to **expand** or **swap** underlying data structures (e.g., other backends for HPC).  

---

## **3. Core Components**

### Vectors

- Defined primarily by the **`Vector`** trait, located in `src/linalg/vector/traits.rs`.
- The trait enforces thread safety (`Send + Sync`) and includes standard operations:
  - Dot product, `norm`, `axpy`, `scale`, cross product (for 3D), sums, min/max, mean, variance, etc.
- **Implementations** exist for:
  - **`Vec<f64>`** (a standard Rust vector)
  - **`faer::Mat<f64>`** interpreted as a **column vector** (with `nrows() == length, ncols() == 1`).

### Matrices

- Defined primarily by the **`Matrix`** trait, located in `src/linalg/matrix/traits.rs`.
- The trait includes methods:
  - `nrows()`, `ncols()`
  - `mat_vec` (matrix-vector multiplication)
  - `trace()`, `frobenius_norm()`
  - `get(i, j)`, plus read/write slices (though not all implementations support slice mutability).
- **Implementations** exist for:
  - **`faer::Mat<f64>`** (a standard dense 2D array).
  - A custom **`SparseMatrix`** that uses a `FxHashMap` for storing non-zero entries.

---

## **4. Vector Module**

The **vector module** is organized as follows:

- **`traits.rs`**: Defines the `Vector` trait.  
- **`vec_impl.rs`**: Implements `Vector` for `Vec<f64>`.  
- **`mat_impl.rs`**: Implements `Vector` for a `faer::Mat<f64>` column.  
- **`vector_builder.rs`**: Contains the `VectorBuilder` utility and supporting traits to build or resize vectors.

### Vector Traits

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

- The **thread-safety** requirement: `Send + Sync`.  
- The **cross product** method is only valid for 3D vectors.  

### Vector Implementations

#### Implementation for `Vec<f64>`

In `vec_impl.rs`, a standard Rust vector is extended via the `Vector` trait:

- **Key methods**:
  - `dot`, `norm`, `axpy`, `scale`, `element_wise_*`, `cross` (3D), `sum`, `max`, `min`, `mean`, `variance`.

**Example**:
```rust
let mut vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];

let dot = vec1.dot(&vec2); // 32.0
vec1.scale(2.0); // vec1 becomes [2.0, 4.0, 6.0]
vec1.axpy(1.5, &vec2); // vec1 = 1.5*vec2 + vec1
```

#### Implementation for `Mat<f64>`

In `mat_impl.rs`, a **`faer::Mat<f64>`** with `ncols() == 1` is treated as a column vector:

- `len()` -> number of rows
- `get(i)`, `set(i)`, `dot(...)`, etc.
- `as_slice()` / `as_mut_slice()` use `try_as_slice()` from `faer`.

**Example**:
```rust
use faer::Mat;
use hydra::linalg::Vector;

let mut mat_vec = Mat::<f64>::zeros(3, 1); // 3x1
mat_vec.set(0, 1.0); 
mat_vec.set(1, 2.0);
mat_vec.set(2, 3.0);

let norm = mat_vec.norm(); // sqrt(1^2 + 2^2 + 3^2) = ~3.74
```

### Vector Builder

**`vector_builder.rs`** provides `VectorBuilder` to build vectors in a generic way.

- `build_vector<T: VectorOperations>(size: usize) -> T`
- `build_dense_vector(size: usize) -> Mat<f64>` 
- `resize_vector<T: VectorOperations + ExtendedVectorOperations>(vector, new_size)`

**Vector Operations** trait:
- `construct(size) -> Self`
- `set_value(index, value)`
- `get_value(index) -> f64`
- `size() -> usize`

Then `ExtendedVectorOperations` adds `resize(new_size)`.  
Implementations are provided for both `Vec<f64>` and `Mat<f64>`.

### Vector Testing

Comprehensive tests in `src/linalg/vector/tests.rs` validate:

- Indexing, dot products, norm, cross product (3D), element-wise ops, etc.
- Edge cases: empty vectors, large vectors, dimension mismatch for cross product, etc.

---

## **5. Matrix Module**

The **matrix module** is organized as follows:

- **`traits.rs`**: Defines the `Matrix`, `MatrixOperations`, and `ExtendedMatrixOperations` traits.  
- **`mat_impl.rs`**: Implements `Matrix` for `faer::Mat<f64>`.  
- **`matrix_builder.rs`**: Contains `MatrixBuilder` utility for constructing/resizing matrices and applying preconditioners.  
- **`sparse_matrix.rs`**: A simple `SparseMatrix` that also implements `Matrix`.  
- **`tests.rs`**: Test suite for matrix functionality.

### Matrix Traits

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

- `MatrixOperations` trait:
  - `construct(rows, cols) -> Self`
  - `get(...)`, `set(...)`
  - `size() -> (usize, usize)`
- `ExtendedMatrixOperations` trait adds `fn resize(&mut self, new_rows, new_cols)`.

### Matrix Implementations

#### Implementation for `Mat<f64>`

In `mat_impl.rs`, `faer::Mat<f64>` is extended:

- **`Matrix`** trait:
  - `nrows()`, `ncols()`
  - `mat_vec(x, y)`: Standard dense matrix-vector multiplication
  - `trace()`: sum of diagonal
  - `frobenius_norm()`: sqrt of sum of squares
  - `as_slice()`, `as_slice_mut()`: yields a `Box<[f64]>` copy or slice
- **`MatrixOperations`**:
  - `construct(rows, cols)` -> zero matrix
  - `set(row, col, value)`, `get(row, col)`  
- **`ExtendedMatrixOperations`**:
  - `resize(&mut self, new_rows, new_cols)` -> creates new `Mat<f64>` and copies existing entries.

**Example**:
```rust
use faer::Mat;
use hydra::linalg::{Matrix, MatrixOperations};

let mut mat = Mat::<f64>::zeros(3, 3);
mat.set(1, 2, 5.0); // matrix.write(1,2,5.0)
let trace = mat.trace();
let norm = mat.frobenius_norm();
```

#### SparseMatrix

In `sparse_matrix.rs`, a **`SparseMatrix`** using `FxHashMap<(row, col), f64>` is provided:

- Also implements the **`Matrix`** trait:
  - `mat_vec(...)`: only iterates over non-zero entries for multiplication
  - `trace()`, `frobenius_norm()`, `get(i, j)` -> zero if absent
  - `as_slice()` is **not supported** (panics), as it’s not contiguous.
- Implements **`MatrixOperations`** and **`ExtendedMatrixOperations`**:
  - `set(row, col, value)`: storing or removing near-zero entries
  - `resize(...)`: re-hash only valid entries that fit in the new dimension range.

This allows a simple **sparse** backend with minimal overhead.

### Matrix Builder

**`matrix_builder.rs`** has a `MatrixBuilder` struct:

- `build_matrix<T: MatrixOperations>(rows, cols) -> T`
- `build_dense_matrix(rows, cols) -> Mat<f64>`
- `resize_matrix<T: MatrixOperations + ExtendedMatrixOperations>(...)`
- `apply_preconditioner(preconditioner, matrix)`: Example usage with a solver preconditioner.

---

### Matrix Testing

The `src/linalg/matrix/tests.rs` typically checks:

- **mat_vec** for correctness  
- Setting/retrieving matrix elements  
- Sizing, resizing, and partial copy logic  
- Corner cases: zero rows/columns, out-of-bounds, etc.

---

## **6. Using the Linear Algebra Module**

Below are typical usage patterns using the vector and matrix abstractions:

### Creating Vectors

1. **`Vec<f64>`** (most common):
   ```rust
   let mut vector = vec![0.0; 5];
   vector.set(0, 1.0);
   ```
2. **Using `faer::Mat<f64>` as a column vector**:
   ```rust
   use faer::Mat;

   let mut mat_vec = Mat::<f64>::zeros(5, 1);
   mat_vec.set(0, 1.0);
   ```

### Performing Vector Operations

**Dot Product**:
```rust
let vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];
let dot_val = vec1.dot(&vec2); // 32.0
```

**Norm**:
```rust
let norm = vec1.norm(); // sqrt(1^2 + 2^2 + 3^2) = ~3.74
```

**Scale and AXPY**:
```rust
vec1.scale(2.0); // [2.0, 4.0, 6.0]
vec1.axpy(1.5, &vec2); // vec1 = 1.5*vec2 + vec1
```

**Element-wise**:
```rust
vec1.element_wise_add(&vec2);
vec1.element_wise_mul(&vec2);
```

### Creating Matrices

1. **Using `faer::Mat<f64>`**:
   ```rust
   let mut matrix = Mat::<f64>::zeros(3, 3);
   matrix.set(1, 1, 5.0);
   ```
2. **Using `SparseMatrix`**:
   ```rust
   use hydra::linalg::matrix::sparse_matrix::SparseMatrix;

   let mut sp_mat = SparseMatrix::new(3, 3);
   sp_mat.set(0, 0, 1.0);
   sp_mat.set(2, 1, 3.5);
   ```

### Performing Matrix Operations

**Matrix-Vector Multiplication**:
```rust
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y); // y = matrix * x
```

**Trace and Frobenius Norm**:
```rust
let trace_val = matrix.trace();
let fro_norm = matrix.frobenius_norm();
```

**Resizing**:
```rust
use hydra::linalg::matrix::MatrixBuilder;
MatrixBuilder::resize_matrix(&mut matrix, 5, 5);
```

---

## **7. Best Practices**

1. **Dimensional Consistency**: Always ensure vectors and matrices match in size when performing multiplication or element-wise operations.  
2. **Thread Safety**: The traits require `Send + Sync`; your data structures must maintain concurrency safety.  
3. **Sparse vs. Dense**: Choose `SparseMatrix` if your matrix has many zero entries. For dense computations, use `faer::Mat<f64>`.  
4. **Cross Product**: Use only for 3D vectors; otherwise, it returns an error.  
5. **Preconditioning**: The `MatrixBuilder::apply_preconditioner` demonstrates how to integrate a solver preconditioner with your matrix.  
6. **Performance**: For large vectors or matrices, ensure you do minimal copying (e.g., pass slices or references).

---

## **8. Conclusion**

The **`Linear Algebra`** module in Hydra offers a unified abstraction layer for **vector** and **matrix** operations, supporting multiple data structures from a single interface:

- **`Vector` trait** with implementations for standard Rust vectors and `faer::Mat<f64>` column vectors.  
- **`Matrix` trait** with implementations for **dense** (`faer::Mat<f64>`) and **sparse** (`SparseMatrix`) usage.  
- **Builders** (`VectorBuilder`, `MatrixBuilder`) for generic creation, resizing, and specialized operations (e.g., applying preconditioners).  

By leveraging these traits and implementations, you can write **cleaner**, **extensible** linear algebra code while mixing and matching data structures best suited to your simulation’s memory and performance requirements.

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
   - [Jacobi Preconditioner](#jacobi-preconditioner)  
   - [LU Preconditioner](#lu-preconditioner)  
   - [ILU Preconditioner](#ilu-preconditioner)  
   - [Cholesky Preconditioner](#cholesky-preconditioner)  
   - [AMG Preconditioner](#amg-preconditioner)  
5. [Using the Solver Module](#5-using-the-solver-module)  
   - [Choosing and Creating a Solver](#choosing-and-creating-a-solver)  
   - [Applying Preconditioners](#applying-preconditioners)  
   - [Solving Linear Systems](#solving-linear-systems)  
6. [Examples and Usage](#6-examples-and-usage)  
   - [Conjugate Gradient Example](#conjugate-gradient-example)  
   - [GMRES Example](#gmres-example)  
7. [Best Practices](#7-best-practices)  
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the **`Solver`** module user guide for Hydra, which provides **iterative solvers** and **preconditioners** to tackle large, sparse linear systems. Whether your matrix is **SPD (Symmetric Positive Definite)** or **general non-symmetric**, the solver framework offers:

- **Krylov Subspace Methods**:
  - Conjugate Gradient (CG)
  - GMRES (Generalized Minimal Residual)
- **Preconditioners** to improve convergence:
  - Jacobi, LU, ILU, Cholesky, AMG, etc.
- A **unified `KSP` trait** and **`SolverManager`** for flexible solver/preconditioner configuration.

This module is intended for advanced numerical simulations in **CFD**, **finite element/volume methods**, or any domain that requires **iterative** solution of \(A x = b\).

---

## **2. Overview of the Solver Module**

The solver code is split into key components:

1. **KSP** (Krylov Subspace Solvers)  
   - **`KSP` trait**: Common interface for solvers.  
   - **`ConjugateGradient`**: For SPD systems.  
   - **`GMRES`**: For general non-symmetric systems.  

2. **Preconditioners**  
   - **`Preconditioner`** trait plus multiple implementations:
     - **Jacobi**: Simple diagonal-based approach.  
     - **LU**, **ILU**, **Cholesky** (using Faer library).  
     - **AMG** (Algebraic Multigrid).
   - Significantly **reduces iteration count** by improving matrix conditioning.

3. **SolverManager**  
   - High-level adapter that unifies a chosen solver with an optional preconditioner.  
   - Offers a single `solve(a, b, x)` method to run the entire solve.

**Parallelization** is handled largely via **Rayon** for operations like dot products, mat-vec multiplications, and more.

---

## **3. Krylov Subspace Solvers**

### KSP Trait

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

- **Purpose**: A consistent interface for iterative solvers like CG or GMRES.
- **`solve(...)`**: Takes a matrix `A`, right-hand side `b`, and solution vector `x`. Returns a `SolverResult` that indicates if convergence was achieved.

### Conjugate Gradient Solver

**`ConjugateGradient`** solves **SPD** systems:  
- Uses a **preconditioner** if set (e.g., Jacobi, ILU, etc.).  
- **Rayon** is used for parallel dot products and residual updates.  
- **Key Methods**:
  - `new(max_iter, tol)`: Creates a solver with iteration limit and tolerance.  
  - `set_preconditioner(...)`: Attach a `Preconditioner`.  

**Workflow**:  
1. Compute initial residual \(r = b - A x\).  
2. Precondition the residual (optional).  
3. Update direction `p`.  
4. Iterate until norm of residual < `tol` or `max_iter` is reached.

### GMRES Solver

**`GMRES`** is for general non-symmetric systems:
- **Restart-based** (the `restart` parameter) to limit Krylov subspace size.  
- **Arnoldi process** to build an orthonormal basis.  
- **Givens rotations** to maintain upper Hessenberg form and compute the solution in a smaller subspace.  
- Also uses an **optional preconditioner**.

**Key Methods**:
- `new(max_iter, tol, restart)`: Basic constructor.  
- `set_preconditioner(...)`: Hook a preconditioner.

---

## **4. Preconditioners**

Each preconditioner implements:

```rust
pub trait Preconditioner: Send + Sync {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}
```

**Goal**: transform the residual `r` into a new vector `z` that is easier for CG/GMRES to work with.

### Jacobi Preconditioner

- **Uses only the diagonal** of matrix `A`.  
- `z[i] = r[i] / A[i, i]` (if diagonal is non-zero).  
- Parallelized with **Rayon**:
  ```rust
  let z = Arc::new(Mutex::new(z));
  (0..a.nrows()).into_par_iter().for_each(|i| { ... });
  ```
- Extremely simple but beneficial for diagonally dominant or well-scaled problems.

### LU Preconditioner

- **LU factorization** of matrix `A` (partial pivoting).  
- Solves `LU * x = r` quickly via forward/back substitution.  
- Implementation uses **`faer::PartialPivLu`**.  
- Good for smaller systems or as a “direct” method in a broader iterative approach.  
- Potentially large overhead for big systems.

### ILU Preconditioner

- **Incomplete LU** factorization preserves the **sparsity** pattern:
  - Ignores small values in factorization to reduce memory overhead.  
  - Approximates the effect of a full LU.  
- Works well for **large, sparse** systems.  
- Implementation:
  - Basic incomplete factor iteration (with possible threshold).  
  - `apply_ilu(...)` for forward/back substitution.

### Cholesky Preconditioner

- **Cholesky** for symmetric positive definite matrices only.  
- Factorizes \(A = LL^T\).  
- Uses **Faer** for decomposition + forward/back solves.  
- Typically used with CG.  
- The code might return an error if the matrix isn’t SPD or factorization fails.

### AMG Preconditioner

- **Algebraic Multigrid (AMG)**:
  - Builds a **hierarchy of coarser grids** from the original matrix.  
  - Uses strength-of-connection, coarsening, interpolation, etc.  
- Implementation includes:
  - Methods to **construct** coarser levels (`generate_operators`, `compute_strength_matrix`, etc.).  
  - A **recursive** approach that calls `apply_recursive(...)` for V-cycle or W-cycle style smoothing.  
- Very powerful for large problems with structured or unstructured grids.

---

## **5. Using the Solver Module**

### Choosing and Creating a Solver

Pick a solver best suited for the system:
- **SPD**: `ConjugateGradient`.
- **General**: `GMRES`.

**Helper**: `create_solver(solver_type, max_iter, tol, restart)` returns a `Box<dyn KSP>`:

```rust
use hydra::solver::ksp::{create_solver, SolverType};

let solver = create_solver(SolverType::GMRES, 1000, 1e-6, 50);
```

Alternatively, you can instantiate directly:
```rust
let mut cg = ConjugateGradient::new(1000, 1e-8);
let mut gmres = GMRES::new(1000, 1e-6, 30);
```

### Applying Preconditioners

Attach a preconditioner (e.g., Jacobi, LU) if desired:
```rust
use hydra::solver::preconditioner::Jacobi;

cg.set_preconditioner(Box::new(Jacobi::default()));
```

If using `SolverManager`:
```rust
use hydra::solver::ksp::{SolverManager};
use std::sync::Arc;
use hydra::solver::preconditioner::{PreconditionerFactory};

let mut manager = SolverManager::new(Box::new(cg));
manager.set_preconditioner(PreconditionerFactory::create_jacobi());
```

### Solving Linear Systems

1. Prepare `A`, `b`, and an **initial guess** `x`.  
2. Call `solve(a, b, x)`.  
3. Check the returned `SolverResult`:
   - `converged`  
   - `iterations`  
   - `residual_norm`

**Example**:
```rust
let result = manager.solve(&a, &b, &mut x);
if result.converged {
    println!("Converged in {} iterations, res norm = {}", result.iterations, result.residual_norm);
} else {
    eprintln!("Not converged after {} iterations", result.iterations);
}
```

---

## **6. Examples and Usage**

### Conjugate Gradient Example

**Context**: Suppose `A` is SPD, size = 2x2 for demonstration.

```rust
use hydra::solver::{ConjugateGradient, KSP};
use hydra::linalg::{Matrix, Vector};
use faer::Mat;

// Create A, b, x
let a = Mat::from_fn(2, 2, |i, j| {
    if i == j {
        4.0
    } else {
        1.0
    }
});
let b = vec![1.0, 2.0];
let mut x = vec![0.0, 0.0];

// Initialize CG
let mut cg = ConjugateGradient::new(100, 1e-6);

// (Optional) attach a Jacobi preconditioner
use hydra::solver::preconditioner::Jacobi;
cg.set_preconditioner(Box::new(Jacobi::default()));

// Solve
let result = cg.solve(&a, &b, &mut x);
if result.converged {
    println!("CG converged in {} iters, solution: {:?}", result.iterations, x);
}
```

### GMRES Example

**Context**: Non-symmetric or indefinite system.

```rust
use hydra::solver::{GMRES, KSP};
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| {
    match (i,j) {
        (0,0) => 2.0, (0,1) => 1.0,
        (1,0) => 3.0, (1,1) => 4.0,
        _ => 0.0,
    }
});
let b = vec![1.0, 2.0];
let mut x = vec![0.0, 0.0];

let mut gmres = GMRES::new(100, 1e-6, 10);
// Optional: LU preconditioner
use hydra::solver::preconditioner::LU;
use std::sync::Arc;
gmres.set_preconditioner(Arc::new(LU::new(&a)));

let result = gmres.solve(&a, &b, &mut x);
if result.converged {
    println!("GMRES converged. x = {:?}", x);
} else {
    println!("GMRES did not converge. residual={}", result.residual_norm);
}
```

---

## **7. Best Practices**

1. **Match Solver to Matrix**:
   - CG for SPD systems,
   - GMRES for general systems.
2. **Leverage Preconditioning**: Even a simple Jacobi can speed up convergence. More advanced (ILU, AMG) for large sparse systems.
3. **Monitor Tolerance**: Adjust `tol` to balance **accuracy vs. iteration count**.
4. **Check Residual Norm**: If the solver does not converge or stalls, consider different preconditioners or re-check matrix conditioning.
5. **Thread Safety**: The code uses **Rayon** heavily—any custom matrix or vector must be `Send + Sync`.
6. **AMG Complexity**: Algebraic multigrid can require more advanced tuning (coarsening threshold, etc.).

---

## **8. Conclusion**

The **`Solver`** module is an **integral** part of Hydra for handling large linear systems in HPC or scientific simulation contexts. By mixing:

- **Krylov solvers** (CG, GMRES),
- **Preconditioners** (Jacobi, LU, ILU, Cholesky, AMG),
- A **unified interface** (`KSP` or `SolverManager`),

…users can flexibly choose the **best** approach for their problem. Properly pairing a solver with an appropriate preconditioner often **drastically** reduces iteration count and CPU time, ensuring robust and efficient solutions for complex CFD, FE, or other large-scale simulations.

---

# Hydra `Time Stepping` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Time Stepping Module](#2-overview-of-the-time-stepping-module)  
3. [Core Components](#3-core-components)  
   - [TimeDependentProblem Trait](#timedependentproblem-trait)  
   - [TimeStepper Trait](#timestepper-trait)  
   - [FixedTimeStepper](#fixedtimestepper)  
4. [Implemented Time Stepping Methods](#4-implemented-time-stepping-methods)  
   - [Explicit Euler Method](#explicit-euler-method)  
   - [Backward Euler Method](#backward-euler-method)  
   - [Runge-Kutta (Partial)](#runge-kutta-partial)  
5. [Using the Time Stepping Module](#5-using-the-time-stepping-module)  
   - [Defining a Time-Dependent Problem](#defining-a-time-dependent-problem)  
   - [Selecting a Time Stepping Method](#selecting-a-time-stepping-method)  
   - [Performing Time Steps](#performing-time-steps)  
6. [Adaptivity and Planned Features](#6-adaptivity-and-planned-features)  
   - [Adaptive Time Stepping](#adaptive-time-stepping)  
   - [Crank-Nicolson Method](#crank-nicolson-method)  
   - [Step Size Control and Error Estimation](#step-size-control-and-error-estimation)  
7. [Best Practices](#7-best-practices)  
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the **`Time Stepping`** module in the Hydra computational framework. This module is designed for evolving **time-dependent** problems, such as ordinary differential equations (ODEs) and partial differential equations (PDEs). It aims to provide:

- **A unified interface** via the `TimeDependentProblem` and `TimeStepper` traits.
- **Multiple time integration methods** such as **explicit** and **implicit** Euler approaches, with partial or planned support for more advanced schemes (Runge-Kutta, Crank-Nicolson).
- **Optional adaptivity** to adjust the time step based on error estimates (in partial form).

The design emphasizes **modularity** and **extensibility**, allowing new methods or adaptivity techniques to be added cleanly.

---

## **2. Overview of the Time Stepping Module**

Below is a simplified file structure:

```
time_stepping/
├── mod.rs
├── ts.rs                 // Core traits: TimeDependentProblem, TimeStepper
├── methods/
│   ├── euler.rs          // Explicit Euler
│   ├── backward_euler.rs // Backward Euler
│   ├── runge_kutta.rs    // Partial Runge-Kutta
│   ├── crank_nicolson.rs // Planned
│   └── mod.rs
├── adaptivity/
│   ├── error_estimate.rs    // For local error estimation
│   ├── step_size_control.rs // For adjusting dt
│   └── mod.rs
└── tests.rs
```

Key submodules:

1. **`ts.rs`**:  
   - Defines `TimeDependentProblem` for problem specification.  
   - Defines `TimeStepper` trait for time-stepping logic.  
   - Offers a sample `FixedTimeStepper` that can step forward in fixed increments.  
   - Includes a custom error type `TimeSteppingError`.

2. **`methods/`**:  
   - `ExplicitEuler` (in `euler.rs`)  
   - `BackwardEuler` (in `backward_euler.rs`)  
   - `RungeKutta` (in `runge_kutta.rs`)  
   - `Crank-Nicolson` is **planned** but not yet implemented.  

3. **`adaptivity/`** (in partial form):  
   - `error_estimate.rs`: Demo function to estimate error by comparing single-step vs. multi-step approaches.  
   - `step_size_control.rs`: A function `adjust_step_size` that modifies dt based on the computed error.

---

## **3. Core Components**

### TimeDependentProblem Trait

Defines how a **time-dependent** system is specified:

```rust
pub trait TimeDependentProblem {
    type State: Clone + UpdateState;
    type Time: Copy + PartialOrd + Add<Output = Self::Time> + From<f64> + Into<f64>;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}
```

- **`compute_rhs(...)`**: Fills `derivative` with \(\frac{d}{dt}\) of the system at `time` and `state`.
- **`initial_state()`**: Returns the system’s initial condition.
- **`get_matrix()`** / `solve_linear_system(...)`: For **implicit** methods requiring matrix solves.

The `State` typically implements Hydra’s `UpdateState` trait (from `equation::fields`) to handle state updates like `state = state + alpha * derivative`.

### TimeStepper Trait

Specifies how to **advance** a system in time:

```rust
pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);
    fn set_time_step(&mut self, dt: P::Time);
    fn get_time_step(&self) -> P::Time;
    fn current_time(&self) -> P::Time;
    fn set_current_time(&mut self, time: P::Time);

    fn get_solver(&mut self) -> &mut dyn KSP; // Some methods might solve systems with a KSP solver
}
```

- **`step(...)`**: Does one time step of size `dt`.
- **`adaptive_step(...)`**: (Optional) for adaptive stepping. Currently partial in some methods.
- **`get_solver()`**: Access underlying solver if needed for matrix solves.

### FixedTimeStepper

An **example** implementation that maintains:

- `current_time`
- `start_time`
- `end_time`
- `time_step`
- A solver manager (though it may not be used in simple explicit steps).

This is one approach to iterating from `start_time` to `end_time` in increments of `time_step`.

---

## **4. Implemented Time Stepping Methods**

### Explicit Euler Method

**File**: `methods/euler.rs`  
**Struct**: `ExplicitEuler<P: TimeDependentProblem>`

- **Description**: Also known as “Forward Euler”, a first-order **explicit** scheme:
  \[
    y_{n+1} = y_n + dt \cdot f(t_n, y_n).
  \]
- **Implementation**:
  - `step(...)`: calls `compute_rhs(...)`, then updates the state with `state = state + dt * derivative`.
  - `adaptive_step(...)`: partial example using error estimates.  
- **Pros**: Very simple, cheap per-step.  
- **Cons**: Potentially unstable if the problem is stiff or if dt is too large.

### Backward Euler Method

**File**: `methods/backward_euler.rs`  
**Struct**: `BackwardEuler`

- **Description**: A **first-order implicit** scheme:
  \[
    y_{n+1} = y_n + dt \cdot f(t_{n+1}, y_{n+1}).
  \]
- **Implementation**:
  - **Requires** `get_matrix()` and `solve_linear_system(...)` from the problem.  
  - Calls `compute_rhs(...)`, then does an implicit solve to get the new state.  
- **Pros**: **Stable** for stiff problems.  
- **Cons**: Each step solves a linear system, so more expensive per-step.

### Runge-Kutta (Partial)

**File**: `methods/runge_kutta.rs`  
**Struct**: `RungeKutta<P>`

- **Current**: The code sets up `stages` but does not fully implement classical RK specifics.  
- **`step(...)`**: Loops over `stages` to compute intermediate states (`k` vectors).  
- **Pros**: Higher potential accuracy than Euler if completed.  
- **Status**: Partially functional. Not yet integrated with adaptivity or advanced Butcher tables, etc.

**Note**: Another partially planned method is `Crank-Nicolson` (in `methods/crank_nicolson.rs`), not currently implemented.

---

## **5. Using the Time Stepping Module**

### Defining a Time-Dependent Problem

Create a struct that implements **`TimeDependentProblem`**:

```rust
use hydra::time_stepping::{TimeDependentProblem, TimeSteppingError};
use hydra::linalg::Matrix;

#[derive(Clone)]
struct MyState {
    data: Vec<f64>,
    // possibly more fields
}

struct MyTimeDependentSystem;

impl TimeDependentProblem for MyTimeDependentSystem {
    type State = MyState;
    type Time = f64;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Fill derivative based on the ODE or PDE
        Ok(())
    }

    fn initial_state(&self) -> Self::State {
        MyState { data: vec![1.0, 2.0, 3.0] }
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar=f64>>> {
        None // or Some(...) if implicit methods need a matrix
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar=f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Solve the linear system if needed (e.g., for Backward Euler).
        Ok(())
    }
}
```

### Selecting a Time Stepping Method

Choose from:

- **`ExplicitEuler`**: `ExplicitEuler::new(dt, start, end)`
- **`BackwardEuler`**: `BackwardEuler::new(start_time, dt)`
- **`FixedTimeStepper`**: A generic approach that calls `compute_rhs`.

**Example**:
```rust
use hydra::time_stepping::methods::euler::ExplicitEuler;
let mut stepper = ExplicitEuler::new(0.01, 0.0, 1.0);
```

### Performing Time Steps

1. **Initialize**:
   ```rust
   let system = MyTimeDependentSystem;
   let mut state = system.initial_state();
   let mut time = 0.0;
   let end_time = 1.0;
   let dt = 0.01;
   ```
2. **Loop**:
   ```rust
   while time < end_time {
       stepper.step(&system, dt, time, &mut state)?;
       time += dt;
   }
   ```
3. **Error Handling**:
   ```rust
   if let Err(e) = stepper.step(&system, dt, time, &mut state) {
       eprintln!("Time stepping error: {:?}", e);
       break;
   }
   ```
4. For **adaptive** steps, call `stepper.adaptive_step(...)` if available.

---

## **6. Adaptivity and Planned Features**

The module includes placeholders for **error estimation** and **step-size control**:

### Adaptive Time Stepping

- **`adaptivity/error_estimate.rs`**: A sample function that compares single-step vs. multi-step solutions to estimate local error.  
- **`adaptivity/step_size_control.rs`**: Adjusts dt based on the ratio \(\sqrt{tol / error}\).  
- **`TimeStepper`** has an `adaptive_step(...)` method. 
  - Implementations in `ExplicitEuler` or `RungeKutta` are partially complete.

### Crank-Nicolson Method

- **File**: `methods/crank_nicolson.rs`  
- **Status**: *Not yet implemented.*  
- **Plan**: A second-order implicit scheme that averages explicit/implicit Euler steps.

### Step Size Control and Error Estimation

- **`estimate_error(...)`**: Inside `error_estimate.rs`, returns a numerical error measure.
- **`adjust_step_size(...)`**: In `step_size_control.rs`, modifies dt based on the computed error.

**Current**: Basic logic is provided, but a fully robust adaptive loop is still under development.

---

## **7. Best Practices**

1. **Method Selection**: 
   - Use **ExplicitEuler** for simpler or non-stiff equations. 
   - Prefer **BackwardEuler** for stiff systems or stability concerns.
2. **Implement Required Trait Methods**: Ensure your `TimeDependentProblem` provides everything needed by your chosen scheme (e.g., a matrix for implicit methods).
3. **Monitor Stability**: For stiff problems, explicit methods can fail unless dt is very small. 
4. **Error Checking**: 
   - The `step(...)` method returns a `Result`— handle `TimeSteppingError` carefully.
5. **Adaptive Steps**: 
   - If the problem changes rapidly, consider partial adaptivity in `ExplicitEuler` or `RungeKutta`.

---

## **8. Conclusion**

Hydra’s **`Time Stepping`** module supplies a flexible, trait-based framework for **time integration**. Users can define custom time-dependent problems, pick or implement a **TimeStepper** method (e.g., **Explicit Euler**, **Backward Euler**), and optionally incorporate **adaptivity**. The partial code for **RungeKutta** and future **Crank-Nicolson** expansions underscores ongoing improvements.

By following the guidelines and employing the module’s abstractions (particularly `TimeDependentProblem` + `TimeStepper`), it’s straightforward to integrate time-stepping logic into Hydra-based simulations—and to extend or refine these methods to meet advanced simulation requirements.

---

# Hydra `Use Cases` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the `use_cases` Module](#2-overview-of-the-use_cases-module)  
3. [Core Components](#3-core-components)  
   - [Matrix Construction](#matrix-construction)
   - [RHS Construction](#rhs-construction)
   - [PISO Solver Suite](#piso-solver-suite)
4. [Using the Matrix and RHS Builders](#4-using-the-matrix-and-rhs-builders)
   - [Building and Initializing a Matrix](#building-and-initializing-a-matrix)
   - [Building and Initializing an RHS Vector](#building-and-initializing-an-rhs-vector)
5. [PISO Algorithm Workflow](#5-piso-algorithm-workflow)
   - [Predictor Step](#predictor-step)
   - [Pressure Correction Step](#pressure-correction-step)
   - [Velocity Correction Step](#velocity-correction-step)
   - [Nonlinear Loop](#nonlinear-loop)
   - [Boundary Handling in PISO](#boundary-handling-in-piso)
6. [Example Usage](#6-example-usage)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the **`use_cases`** module of the Hydra computational framework. This module provides **higher-level use cases and workflows** that build upon Hydra’s core features (e.g., domain handling, time stepping, solver management). The primary goals are:

- To **construct and initialize** matrices and RHS vectors for linear systems.
- To demonstrate the **PISO** (Pressure-Implicit with Splitting of Operators) solver steps, often used in fluid dynamics simulations.

---

## **2. Overview of the `use_cases` Module**

The `use_cases` directory collects higher-level routines that combine or orchestrate lower-level functionalities from other Hydra modules. Its structure is:

```bash
src/use_cases/
├── matrix_construction.rs
├── rhs_construction.rs
├── piso/
│   ├── boundary.rs
│   ├── nonlinear_loop.rs
│   ├── predictor.rs
│   ├── pressure_correction.rs
│   ├── velocity_correction.rs
│   └── mod.rs
└── mod.rs
```

**Key Submodules**:

1. **`matrix_construction`**: Tools to create, resize, and initialize matrices for simulations.  
2. **`rhs_construction`**: Tools to create and initialize the right-hand side (RHS) vectors.  
3. **`piso`**: Implementation of the **PISO** solver approach, including predictor, pressure correction, velocity correction, boundary condition handling, and a nonlinear iteration loop.

---

## **3. Core Components**

### Matrix Construction

- **File**: [`matrix_construction.rs`](./matrix_construction.rs)  
- **Struct**: `MatrixConstruction`

```rust
pub struct MatrixConstruction;

impl MatrixConstruction {
    pub fn build_zero_matrix(rows: usize, cols: usize) -> Mat<f64> { ... }
    pub fn initialize_matrix_with_value<T: MatrixOperations>(matrix: &mut T, value: f64) { ... }
    pub fn resize_matrix<T: ExtendedMatrixOperations>(matrix: &mut T, new_rows: usize, new_cols: usize) { ... }
}
```

**Purpose**:  
- Build Faer-based dense matrices of specified dimensions.  
- Initialize them (set all entries to a particular value).  
- Resize while preserving data if possible.

### RHS Construction

- **File**: [`rhs_construction.rs`](./rhs_construction.rs)  
- **Struct**: `RHSConstruction`

```rust
pub struct RHSConstruction;

impl RHSConstruction {
    pub fn build_zero_rhs(size: usize) -> Mat<f64> { ... }
    pub fn initialize_rhs_with_value<T: Vector<Scalar = f64>>(vector: &mut T, value: f64) { ... }
    pub fn resize_rhs(vector: &mut Mat<f64>, new_size: usize) { ... }
}
```

**Purpose**:  
- Create a **dense vector** for the right-hand side of a linear system.  
- Fill that vector with some initial condition.  
- Resize it for changing problem sizes.

### PISO Solver Suite

- **Folder**: [`piso/`](./piso)  
- **Main**: [`mod.rs`](./piso/mod.rs) with the `PISOSolver`  
- **Submodules**: 
  - `predictor.rs`  
  - `pressure_correction.rs`  
  - `velocity_correction.rs`  
  - `nonlinear_loop.rs`  
  - `boundary.rs`  

**Purpose**: Provide a cohesive **PISO** implementation. The steps are typically:

1. **Predictor**: Solve momentum equation ignoring updated pressure.  
2. **Pressure Correction**: Solve Poisson equation for pressure.  
3. **Velocity Correction**: Adjust velocity to enforce continuity.  
4. **Nonlinear Loop**: Repeat until convergence or iteration limit.

Within the submodules:

- **`predictor`**: The velocity predictor step.  
- **`pressure_correction`**: Solves the pressure Poisson system, obtains corrected pressure.  
- **`velocity_correction`**: Uses pressure correction to fix velocity field.  
- **`nonlinear_loop`**: Repeats steps until the flow solution converges.  
- **`boundary`**: Specialized boundary condition applications for PISO steps.

---

## **4. Using the Matrix and RHS Builders**

### Building and Initializing a Matrix

```rust
use hydra::use_cases::matrix_construction::MatrixConstruction;
use faer::Mat;

fn main() {
    // Create a 5x5 zero matrix
    let mut matrix = MatrixConstruction::build_zero_matrix(5, 5);

    // Initialize all elements to 2.5
    MatrixConstruction::initialize_matrix_with_value(&mut matrix, 2.5);

    // Resize to 7x7, preserving the top-left 5x5 block
    MatrixConstruction::resize_matrix(&mut matrix, 7, 7);

    println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());
}
```

### Building and Initializing an RHS Vector

```rust
use hydra::use_cases::rhs_construction::RHSConstruction;
use faer::Mat;

fn main() {
    // Create a zero vector of length 5
    let mut rhs = RHSConstruction::build_zero_rhs(5);

    // Fill the RHS with a constant value of 1.0
    RHSConstruction::initialize_rhs_with_value(&mut rhs, 1.0);

    // Resize to length 8
    RHSConstruction::resize_rhs(&mut rhs, 8);
}
```

---

## **5. PISO Algorithm Workflow**

Below is a **high-level** explanation of the PISO approach as implemented in the `piso` submodule:

1. **Predictor Step**:  
   - Solve the momentum equation ignoring any new pressure correction.  
   - Typically updates velocity using an approximate pressure from the previous iteration.

2. **Pressure Correction Step**:  
   - Formulate and solve the pressure Poisson equation.  
   - Compute the correction field for pressure to ensure mass conservation.

3. **Velocity Correction Step**:  
   - Adjust velocity with the newly computed pressure correction to enforce divergence-free flow.

4. **Nonlinear Loop**:  
   - Repeats the predictor → pressure → velocity corrections until residuals meet a tolerance or iteration limit.

5. **Boundary Handling**:  
   - The `boundary.rs` file shows how boundary conditions are specifically adapted for the PISO steps (especially for pressure Poisson).

### Predictor Step

- **File**: [`predictor.rs`](./piso/predictor.rs)  
- **Function**: `predict_velocity(...)`

**Key Points**:  
- Receives mesh, fields, fluxes, etc.  
- Assembles momentum fluxes and updates the velocity field.  
- Ignores the new pressure correction in this stage.

### Pressure Correction Step

- **File**: [`pressure_correction.rs`](./piso/pressure_correction.rs)  
- **Function**: `solve_pressure_poisson(...)`  

**Key Points**:  
- Assembles the matrix for the pressure Poisson equation.  
- Solves it using a `KSP` solver (e.g., Conjugate Gradient).  
- Outputs a `PressureCorrectionResult` containing the **residual** measure.

### Velocity Correction Step

- **File**: [`velocity_correction.rs`](./piso/velocity_correction.rs)  
- **Function**: `correct_velocity(...)`

**Key Points**:  
- Uses the **pressure gradient** from the correction step to adjust velocity.  
- Ensures continuity / divergence-free condition.

### Nonlinear Loop

- **File**: [`nonlinear_loop.rs`](./piso/nonlinear_loop.rs)  
- **Function**: `solve_nonlinear_system(...)`

**Key Points**:  
- Orchestrates multiple predictor → correction cycles until convergence.  
- Checks residual from the pressure correction result to decide stopping.  
- If not converged within `max_iterations`, returns an error.

### Boundary Handling in PISO

- **File**: [`boundary.rs`](./piso/boundary.rs)  
- **Function**: `apply_pressure_poisson_bc(...)`

**Key Points**:  
- Specialized routine to apply boundary conditions to the matrix and RHS in the pressure Poisson step.  
- Uses Hydra’s boundary condition abstractions.

---

## **6. Example Usage**

Consider a scenario where you want to:

1. Build a system matrix and RHS.  
2. Run a partial PISO iteration.

```rust
use hydra::{
   use_cases::{
       matrix_construction::MatrixConstruction,
       rhs_construction::RHSConstruction,
       piso::{PISOSolver, PISOConfig},
   },
   // ...other Hydra modules
};

fn main() {
   // Step 1: Build and initialize a matrix
   let mut mat = MatrixConstruction::build_zero_matrix(10, 10);
   MatrixConstruction::initialize_matrix_with_value(&mut mat, 0.0);

   // Step 2: Build and initialize an RHS vector
   let mut rhs = RHSConstruction::build_zero_rhs(10);
   RHSConstruction::initialize_rhs_with_value(&mut rhs, 5.0);

   // Step 3: Create a PISO solver with a mesh, time stepper, and config
   let mesh = Mesh::new(); // Suppose we have a mesh
   let time_stepper = Box::new(...); // Provide a valid TimeStepper
   let config = PISOConfig { max_iterations: 5, tolerance: 1e-5, relaxation_factor: 0.7 };
   let mut piso_solver = PISOSolver::new(mesh, time_stepper, config);

   // Step 4: Solve with PISO for one step (requires a problem + state)
   // ...
}
```

---

## **7. Best Practices**

1. **Matrix & RHS**:  
   - Use `MatrixConstruction` and `RHSConstruction` for consistent creation/resizing.  
   - Initialize values carefully to avoid leftover data from prior simulations.

2. **PISO**:  
   - Ensure your mesh and boundary conditions are set properly before calling predictor/correction steps.  
   - Monitor the pressure Poisson solver’s **residual** to confirm convergence.  
   - Use the `nonlinear_loop` functionality for advanced iterative flows if needed.

3. **Integration**:  
   - The `use_cases` are building blocks. Combine them with Hydra’s domain, boundary, and solver modules for robust simulations.  
   - Keep track of the current simulation **time** and **time step** in PISO updates.

---

## **8. Conclusion**

The **`use_cases`** module in Hydra provides essential, higher-level building blocks for typical solver workflows:

- **Matrix construction** (`matrix_construction`) and **RHS building** (`rhs_construction`).
- A **PISO** solver suite that orchestrates the classic predictor, pressure correction, and velocity correction steps, with a nonlinear iteration loop and specialized boundary condition handling.

By combining these routines with Hydra’s domain, boundary, and solver infrastructure, users can **rapidly implement** advanced simulation pipelines for fluid dynamics, CFD, or other PDE-based problems.

---

# Hydra `Extrusion` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Extrusion Module](#2-overview-of-the-extrusion-module)  
3. [Core Components](#3-core-components)  
   - [ExtrudableMesh Trait](#extrudablemesh-trait)  
   - [QuadrilateralMesh and TriangularMesh](#quadrilateralmesh-and-triangularmesh)  
4. [Use Cases](#4-use-cases)  
   - [Extruding to Hexahedrons or Prisms](#extruding-to-hexahedrons-or-prisms)  
   - [Vertex and Cell Extrusion Utilities](#vertex-and-cell-extrusion-utilities)  
5. [Infrastructure](#5-infrastructure)  
   - [Mesh I/O](#mesh-io)  
   - [Logger](#logger)  
6. [Interface Adapters](#6-interface-adapters)  
   - [ExtrusionService](#extrusionservice)  
7. [Using the Extrusion Module](#7-using-the-extrusion-module)  
   - [Loading a 2D Mesh](#loading-a-2d-mesh)  
   - [Extruding to 3D](#extruding-to-3d)  
   - [Saving the Extruded Mesh](#saving-the-extruded-mesh)  
8. [Best Practices](#8-best-practices)  
9. [Conclusion](#9-conclusion)

---

## **1. Introduction**

The **`extrusion`** module in Hydra provides functionality to **convert** a 2D mesh into a 3D mesh by **extruding** it along a chosen axis (often the \( z \)-axis). It supports:

- **Quadrilateral** 2D meshes \(\rightarrow\) **Hexahedral** 3D meshes.  
- **Triangular** 2D meshes \(\rightarrow\) **Prismatic** 3D meshes.

The module is composed of **core** definitions for extrudable meshes, **use case** logic that actually performs the extrusion, **infrastructure** for reading/writing mesh data, and an **interface adapter** layer (like `ExtrusionService`) for a high-level API.

---

## **2. Overview of the Extrusion Module**

**Location**: `src/extrusion/`

Submodules:

- **`core/`**: Defines the `ExtrudableMesh` trait and specific mesh types (`QuadrilateralMesh`, `TriangularMesh`).  
- **`infrastructure/`**: Tools for reading/writing 2D or extruded meshes (`mesh_io`), plus a `logger`.  
- **`interface_adapters/`**: Contains services or adapters for extruding a mesh, e.g. `ExtrusionService`.  
- **`use_cases/`**: Implements specific extrusion steps—`extrude_mesh`, `vertex_extrusion`, `cell_extrusion`.

**Purpose**: Provide a **clear** path from a loaded 2D mesh to a final 3D Hydra `Mesh` object that can be used for PDE solving in Hydra’s pipeline.

---

## **3. Core Components**

### ExtrudableMesh Trait

File: **`extrudable_mesh.rs`**  
Defines a **`ExtrudableMesh`** trait with:

```rust
pub trait ExtrudableMesh: Debug {
    fn is_valid_for_extrusion(&self) -> bool;
    fn get_vertices(&self) -> Vec<[f64; 3]>;
    fn get_cells(&self) -> Vec<Vec<usize>>;
    // ...
    fn is_quad_mesh(&self) -> bool { ... }
    fn is_tri_mesh(&self) -> bool { ... }
    fn as_quad(&self) -> Option<&QuadrilateralMesh>;
    fn as_tri(&self) -> Option<&TriangularMesh>;
    fn as_any(&self) -> &dyn std::any::Any;
}
```

- **Purpose**: A 2D mesh must implement `ExtrudableMesh` to be extruded into 3D.  
- The trait determines if it’s quadrilateral or triangular, obtains vertices/cells, and can be downcast to the specialized type.

### QuadrilateralMesh and TriangularMesh

Files: **`hexahedral_mesh.rs`**, **`prismatic_mesh.rs`**  

- **`QuadrilateralMesh`**:
  - A 2D mesh with four vertices per cell.  
  - Suitable for extrusion into **hexahedrons**.  
  - Implements `ExtrudableMesh` by verifying all cells have length 4, returning vertex/cell data.

- **`TriangularMesh`**:
  - A 2D mesh with three vertices per cell.  
  - Suitable for extrusion into **prisms**.  
  - Implements `ExtrudableMesh` similarly, but each cell has length 3.

These mesh structs store:
```rust
pub struct QuadrilateralMesh {
   vertices: Vec<[f64; 3]>,
   cells: Vec<Vec<usize>>,
}

pub struct TriangularMesh {
   vertices: Vec<[f64; 3]>,
   cells: Vec<Vec<usize>>,
}
```

---

## **4. Use Cases**

### Extruding to Hexahedrons or Prisms

**`extrude_mesh.rs`** in `use_cases`:

- **`ExtrudeMeshUseCase::extrude_to_hexahedron(quad_mesh, depth, layers)`**  
- **`ExtrudeMeshUseCase::extrude_to_prism(tri_mesh, depth, layers)`**  

Both create a new Hydra `Mesh` after extruding vertices up to `depth`, subdivided into `layers`.

**Workflow**:

1. **vertex_extrusion**: Duplicate vertices along a new \( z \)-coordinate for each layer.  
2. **cell_extrusion**: Connect base and top layer vertices to form 3D cells (hexahedrons or prisms).  
3. **Build** a final Hydra `Mesh`.

### Vertex and Cell Extrusion Utilities

**`vertex_extrusion.rs`**:

- **`VertexExtrusion::extrude_vertices(...)`**: Repeats each base vertex across multiple layers, stepping in the z-axis by `depth / layers`.  

**`cell_extrusion.rs`**:

- **`CellExtrusion::extrude_quadrilateral_cells(...)`**: For each 2D quad, produce multiple 3D hexahedrons.  
- **`CellExtrusion::extrude_triangular_cells(...)`**: For each 2D triangle, produce multiple 3D prisms.

---

## **5. Infrastructure**

### Mesh I/O

**`mesh_io.rs`**:

- **`MeshIO::load_2d_mesh(file_path)`**: Reads a 2D mesh from a Gmsh file, detects if cells are tri or quad, and returns the appropriate `ExtrudableMesh` (QuadrilateralMesh or TriangularMesh).  
- **`MeshIO::save_3d_mesh(mesh, file_path)`**: Writes a Hydra `Mesh` to a Gmsh-like format (lists nodes, then elements).  

### Logger

**`logger.rs`**:

- A simple **`Logger`** struct to log messages (info, warn, error) with timestamps.  
- Can log to a file or stdout.  
- Used for debugging or general info while extruding.

---

## **6. Interface Adapters**

### ExtrusionService

File: **`extrusion_service.rs`**  
A high-level function:

```rust
pub fn extrude_mesh(mesh: &dyn ExtrudableMesh, depth: f64, layers: usize) -> Result<Mesh, String>
```

- Checks mesh type (`is_quad_mesh()` or `is_tri_mesh()`).  
- Calls `ExtrudeMeshUseCase` to produce either a hexahedral or prismatic Hydra `Mesh`.  
- Returns an error if unsupported.

**Use Cases**:
- Allows user code to simply do:
  ```rust
  let extruded_3d = ExtrusionService::extrude_mesh(&my_2d_mesh, 10.0, 3)?;
  ```

---

## **7. Using the Extrusion Module**

### Loading a 2D Mesh

1. **Load** from Gmsh file using:
   ```rust
   let extrudable_2d_mesh = MeshIO::load_2d_mesh("path/to/two_dim_mesh.msh")?;
   ```
   This returns a `Box<dyn ExtrudableMesh>`, which is either a `QuadrilateralMesh` or `TriangularMesh`.

2. **Check** the mesh validity if needed:
   ```rust
   assert!(extrudable_2d_mesh.is_valid_for_extrusion());
   ```

### Extruding to 3D

1. **Call** `ExtrusionService::extrude_mesh(extrudable_2d_mesh.as_ref(), depth, layers)`.
2. If `Quad` -> produces hexahedron cells; if `Tri` -> produces prism cells.
3. A Hydra `Mesh` is returned for 3D PDE usage.

Example:

```rust
let mesh_2d = MeshIO::load_2d_mesh("quad_example.msh")?;
let extruded_mesh_3d = ExtrusionService::extrude_mesh(&*mesh_2d, 10.0, 4)?;
```

### Saving the Extruded Mesh

Use **`MeshIO::save_3d_mesh(&extruded_mesh, "extruded_output.msh")`**. This writes:

- `$Nodes` block with vertex coordinates.  
- `$Elements` block with cell connectivity.

---

## **8. Best Practices**

1. **Validate** the 2D mesh is homogeneous (all tri or all quad) before extruding.  
2. **Decide** how many `layers` you need for your PDE accuracy. Larger `layers` = finer mesh in z.  
3. **Use Logging** for debugging large extrusions or boundary conditions.  
4. **Check** final 3D mesh with a tool that can open `.msh` (like Gmsh) if needed.  
5. **Performance**: Large extrusions can produce many 3D cells; ensure memory usage is acceptable.

---

## **9. Conclusion**

The **`extrusion`** module in Hydra simplifies generating a 3D mesh from a 2D quadrilateral or triangular mesh. By implementing **`ExtrudableMesh`**, the code can:

- Distinguish quad vs. tri,
- Build either **hexahedral** or **prismatic** 3D cells,
- Read or save from Gmsh or other formats, and
- Provide a final Hydra `Mesh` for PDE or solver processes.

By splitting logic between **core** definitions (`ExtrudableMesh`, specialized mesh types), **use cases** (actual extrusion steps), and **infrastructure** (`MeshIO` I/O, `Logger`), the module maintains a **clean, extensible** design for extruding 2D to 3D in Hydra.

---

# Hydra `Interface Adapters` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Interface Adapters Module](#2-overview-of-the-interface-adapters-module)  
3. [Core Adapters](#3-core-adapters)  
   - [VectorAdapter](#vectoradapter)  
   - [MatrixAdapter](#matrixadapter)  
   - [SectionMatVecAdapter](#sectionmatvecadapter)  
4. [Domain and System Solvers](#4-domain-and-system-solvers)  
   - [DomainBuilder (`domain_adapter.rs`)](#domainbuilder-domain_adapterrs)  
   - [SystemSolver (`system_solver.rs`)](#systemsolver-system_solverrs)  
5. [Using the Interface Adapters](#5-using-the-interface-adapters)  
   - [Mapping Hydra Sections to Dense Vectors](#mapping-hydra-sections-to-dense-vectors)  
   - [Converting Matrices for External Solvers](#converting-matrices-for-external-solvers)  
   - [Building a Domain Programmatically](#building-a-domain-programmatically)  
   - [Solving Systems from MatrixMarket Files](#solving-systems-from-matrixmarket-files)  
6. [Best Practices](#6-best-practices)  
7. [Conclusion](#7-conclusion)

---

## **1. Introduction**

The **`interface_adapters`** module provides **utility classes** and **adapters** that simplify:

- Conversion between Hydra’s `Section<T>` or mesh-based data and external representations like `faer::Mat<f64>` (dense matrices) or standard vectors.
- Bridging the gap between Hydra’s PDE/mesh-centered approach and external solver or domain-building functionalities (like reading/writing matrix data, domain construction, etc.).

Key features:

- **`VectorAdapter`**: Helps create, resize, and manipulate dense column vectors in a consistent way with Hydra’s vector traits.  
- **`MatrixAdapter`**: Similar bridging for Hydra’s `Matrix`/`MatrixOperations` with dense matrices from `faer`.  
- **`SectionMatVecAdapter`**: Translates Hydra’s `Section<T>` to/from standard linear algebra objects (vectors, matrices).  
- **`DomainBuilder`**: Programmatic construction of a domain/mesh entity with vertices, edges, faces, and cells, plus reordering or geometry validation.  
- **`SystemSolver`**: Provides functionalities to parse MatrixMarket files, build solvers, and solve linear systems with user-chosen solver or preconditioner.

---

## **2. Overview of the Interface Adapters Module**

**Location**: `src/interface_adapters/`

Submodules:

- **`vector_adapter.rs`**  
- **`matrix_adapter.rs`**  
- **`section_matvec_adapter.rs`**  
- **`domain_adapter.rs`**  
- **`system_solver.rs`**  

These modules define how Hydra data structures relate to external or more generic data structures, e.g., `faer::Mat<f64>`, external linear solvers, or a domain-building approach that can be used in a typical user script.

---

## **3. Core Adapters**

### VectorAdapter

File: **`vector_adapter.rs`**  
**Purpose**: Helper for creating/setting/resizing **dense vectors** in Hydra.

- **`new_dense_vector(size) -> Mat<f64>`**: Returns a column vector (size x 1).  
- **`resize_vector(...)`**: If a given vector implements Hydra’s `Vector` trait, can resize it in place.  
- **`set_element(...)`**: Assign a value at a given index.  
- **`get_element(...)`**: Retrieve a value from a vector.

**Use Cases**:  
- Creating a new vector for the solver’s right-hand side or solution.  
- Updating a single element in a `faer::Mat<f64>` used as a column vector.

### MatrixAdapter

File: **`matrix_adapter.rs`**  
**Purpose**: Helps create or manipulate **dense** 2D arrays (`faer::Mat<f64>`) and integrate with Hydra’s `Matrix` trait or preconditioners.

- **`new_dense_matrix(rows, cols) -> Mat<f64>`**: Creates an empty matrix.  
- **`resize_matrix(...)`**: If the type supports `ExtendedMatrixOperations`, resizes the matrix.  
- **`set_element(...)`**, **`get_element(...)`**: Basic element-level manipulation.  
- **`apply_preconditioner(...)`**: Demonstrates how to invoke a Hydra **`Preconditioner`** on a matrix + vector.

**Use Cases**:  
- Building or reading a matrix for a system solve.  
- Converting from Hydra’s internal `Matrix` representation to a standard dense matrix for external libraries.

### SectionMatVecAdapter

File: **`section_matvec_adapter.rs`**  
**Purpose**: Convert between Hydra’s `Section<T>` (mesh-based data) and standard linear algebra objects (vectors or matrices).

Examples:

- **`section_to_dense_vector(...)`**: Takes a `Section<Scalar>` and returns `Vec<f64>`.  
- **`dense_vector_to_section(...)`**: The inverse.  
- **`section_to_dense_matrix(...)`** or `sparse_to_dense_matrix(...)`: For converting a `Section<Tensor3x3>` or `Section<Scalar>` into a `faer::Mat<f64>`.  
- **`matmut_to_section(...)`**: The inverse direction from a `faer::Mat<f64>` to Hydra’s `Section<Scalar>`.  

These functionalities are crucial when *the PDE-based data in Hydra’s mesh sections must be used in a standard solver or when reading/writing data from an external format.*

---

## **4. Domain and System Solvers**

### DomainBuilder (`domain_adapter.rs`)

**Struct**: `DomainBuilder`  
Provides a **procedural** approach to building a mesh:

- **`add_vertex(id, coords)`**: Insert a new vertex.  
- **`add_edge(vertex1, vertex2)`**: Connect existing vertices with an edge.  
- **`add_cell(...)`** or `add_tetrahedron_cell(...)` / `add_hexahedron_cell(...)`: Insert new cells, creating appropriate faces and mesh relationships.  
- **`apply_reordering()`**: (Optional) uses the **Cuthill-McKee** algorithm to reorder the mesh for performance.  
- **`validate_geometry()`**: Runs geometry checks (like verifying no duplicate vertex coords).  
- **`build()`**: Finalizes and returns the `Mesh`.

**Use Cases**:  
- Constructing a mesh from user data or a script.  
- Testing small or custom domain topologies.

### SystemSolver (`system_solver.rs`)

**Struct**: `SystemSolver`  
**Focus**: Reading a **MatrixMarket** file to get matrix + optional RHS, then using a Hydra **KSP** solver to solve.

- **`solve_from_file_with_solver(...)`**: 
  1. Parse a `.mtx` file with `mmio::read_matrix_market(...)`.  
  2. Build a dense matrix using `MatrixAdapter`.  
  3. Derive the `_rhs1.mtx` filename, read the RHS, build a vector with `VectorAdapter`.  
  4. Set up a `SolverManager` with optional preconditioner, solve, and return `SolverResult`.

**Use Cases**:

- Interfacing with external data or benchmarks in MatrixMarket format.  
- Demonstrating Hydra’s KSP solvers with standard matrix input.

---

## **5. Using the Interface Adapters**

### Mapping Hydra Sections to Dense Vectors

1. You have a `Section<Scalar>` storing mesh-based values.  
2. Build an index mapping or use entity IDs directly.  
3. **`SectionMatVecAdapter::section_to_dense_vector()`** to produce `Vec<f64>` or `faer::Mat<f64>` from the section.  
4. Possibly pass that vector to an external library.  
5. After solving or modifying, convert back with **`dense_vector_to_section()`** or `matmut_to_section(...)`.

### Converting Matrices for External Solvers

- Start with a Hydra-based **`Section<Scalar>`** that acts like a sparse matrix (row/column from `MeshEntity` IDs).  
- **`section_to_dense_matrix(...)`** or **`sparse_to_dense_matrix(...)`** to produce a `faer::Mat<f64>`.  
- Solve or process externally.  
- If the solution or updates need to be returned, call **`dense_matrix_to_section(...)`** to map back.

### Building a Domain Programmatically

**`DomainBuilder`**:

```rust
let mut builder = DomainBuilder::new();
builder
    .add_vertex(0, [0.0, 0.0, 0.0])
    .add_vertex(1, [1.0, 0.0, 0.0])
    .add_edge(0, 1)
    .add_cell(vec![0, 1, 2]) // e.g., a triangular cell in 2D
    .apply_reordering()
    .validate_geometry();
let mesh = builder.build();
```

Now you have a Hydra **`Mesh`**. This is helpful for quick test domains or specialized geometry generation.

### Solving Systems from MatrixMarket Files

**`SystemSolver`**:

```rust
use hydra::interface_adapters::system_solver::SystemSolver;
use hydra::solver::ksp::KSP;
use hydra::solver::{cg::ConjugateGradient};

let mm_file = "path/to/matrix.mtx";
let solver = ConjugateGradient::new(1000, 1e-8);

let result = SystemSolver::solve_from_file_with_solver(
    mm_file,
    solver,
    None, // or Some(preconditioner_factory) if needed
).unwrap();

if result.converged {
    println!("Converged in {} iterations", result.iterations);
}
```

This code:

1. Reads a `.mtx` file, builds a dense matrix.  
2. Guesses a `_rhs1.mtx` for the RHS.  
3. Creates a solver (e.g., CG) and possibly a preconditioner.  
4. Solves, returning a `SolverResult`.

---

## **6. Best Practices**

1. **Keep Adapters Modular**: Each adapter focuses on a single transformation (e.g., `Section` <-> `faer::Mat`).  
2. **Validate Indices**: Ensure consistent entity ID or index mapping when using `SectionMatVecAdapter`.  
3. **Use Reordering**: If building a domain with `DomainBuilder`, you can reorder for better solver performance.  
4. **MatrixMarket**: Provide well-formed `.mtx` and `_rhs1.mtx` pairs for the `SystemSolver`.  
5. **Performance**: Conversions can be done frequently; cache or reuse your index mappings to avoid overhead in repeated calls.

---

## **7. Conclusion**

The **`interface_adapters`** module in Hydra seamlessly integrates Hydra’s mesh-based PDE approach with **standard** linear algebra or domain building approaches. Its submodules:

- **`VectorAdapter`** and **`MatrixAdapter`**: Bridge Hydra’s solver interfaces and `faer::Mat<f64>`.  
- **`SectionMatVecAdapter`**: Translates Hydra `Section<T>` data into classical `Vec<f64>`/`Mat<f64>` forms, vital for external solver usage.  
- **`DomainBuilder`**: Simplifies building or reordering mesh-based domains.  
- **`SystemSolver`**: Provides a convenient interface for loading `.mtx` files and running Hydra’s `KSP` solvers.

Using these adapters, developers can **mix** Hydra’s domain and PDE logic with external data formats (MatrixMarket, custom domain building) or external HPC tools, while maintaining Hydra’s flexible architecture for PDE-based simulations.

---

# Hydra `Input/Output` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Input/Output Module](#2-overview-of-the-inputoutput-module)  
3. [Core Components](#3-core-components)  
   - [GmshParser](#gmshparser)  
   - [MeshGenerator](#meshgenerator)  
   - [MatrixMarket I/O (MMIO)](#matrixmarket-io-mmio)  
4. [Using the Gmsh Parser](#4-using-the-gmsh-parser)  
5. [Mesh Generation Tools](#5-mesh-generation-tools)  
6. [Matrix I/O with MMIO](#6-matrix-io-with-mmio)  
   - [Reading MatrixMarket Files](#reading-matrixmarket-files)  
   - [Writing MatrixMarket Files](#writing-matrixmarket-files)  
7. [Example Workflow](#7-example-workflow)  
8. [Best Practices](#8-best-practices)  
9. [Conclusion](#9-conclusion)

---

## **1. Introduction**

The **`input_output`** module in Hydra streamlines the **reading** and **writing** of mesh and matrix data. It handles importing data from Gmsh files to create Hydra `Mesh` objects, generating common geometric meshes, and converting to/from **MatrixMarket** (\*.mtx) files for matrix-based linear algebra tasks. This is helpful when:

- Loading a **2D or 3D** mesh from a Gmsh format.  
- Generating standard mesh shapes (rectangles/cuboids, circles) for quick testing.  
- Reading/writing **MatrixMarket** files to interface with external solvers or data sets.

---

## **2. Overview of the Input/Output Module**

**Directory**: `src/input_output/`

- **`gmsh_parser.rs`**: The core Gmsh parsing logic that creates a Hydra `Mesh` from a `.msh` file.  
- **`mesh_generation.rs`**: Automated mesh-building routines for rectangular or circular domains (2D/3D).  
- **`mmio.rs`**: MatrixMarket I/O (load a matrix from disk or save it back).  

**Purpose**: Provide **IO** and **mesh-building** features that are commonly needed in Hydra workflows.

---

## **3. Core Components**

### GmshParser

- **File**: `gmsh_parser.rs`  
- **Struct**: `GmshParser`

```rust
pub struct GmshParser;

impl GmshParser {
    pub fn from_gmsh_file(file_path: &str) -> Result<Mesh, io::Error> { ... }
    // ...
}
```

**Role**:  
- Reads a `.msh` file line by line.  
- Builds a Hydra `Mesh` by parsing:
  - `$Nodes` section → sets vertex coordinates.  
  - `$Elements` section → adds cells and relationships to those vertices.  

**Supported Element Types**:  
- Triangles (type `2`)  
- Quadrilaterals (type `3`)  
- Other element types are ignored.

**Result**: A fully instantiated `Mesh` with vertex positions and cell connectivity.

### MeshGenerator

- **File**: `mesh_generation.rs`  
- **Struct**: `MeshGenerator`

```rust
pub struct MeshGenerator;

impl MeshGenerator {
    pub fn generate_rectangle_2d(width: f64, height: f64, nx: usize, ny: usize) -> Mesh { ... }
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh { ... }
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh { ... }
    // ...
}
```

**Role**:  
- Creates sample meshes **programmatically** rather than reading from a file.  
- Supports:
  - 2D Rectangles → A grid of quadrilaterals.  
  - 3D Rectangles → A grid of hexahedrons.  
  - Circles (2D) → A radial mesh with triangular cells.  

**Usage**: For quick testing or standardized domain generation.

### MatrixMarket I/O (MMIO)

- **File**: `mmio.rs`  
- **Provides**: 
  - **`read_matrix_market`**: Load a matrix from a `.mtx` file in either **coordinate** or **array** format.  
  - **`write_matrix_market`**: Save a matrix in coordinate or array format.

**Coordinate Format**: Typically used for **sparse** matrices (\(row, col, value\)).  
**Array Format**: Typically used for **dense** arrays (row-major listing of values).

---

## **4. Using the Gmsh Parser**

**Step-by-Step**:

1. **Call**:
   ```rust
   let mesh_result = GmshParser::from_gmsh_file("my_mesh.msh");
   ```
2. **Check** the returned `Result<Mesh, io::Error>`:
   ```rust
   if let Ok(mesh) = mesh_result {
       // Use `mesh` in Hydra simulation
   } else {
       eprintln!("Failed to load mesh");
   }
   ```
3. The resulting **`Mesh`** object has:
   - Vertex coordinates
   - Cells referencing those vertices
   - Sieve relationships to represent connectivity

---

## **5. Mesh Generation Tools**

To build a mesh without a file:

1. **Generate a 2D rectangle**:
   ```rust
   let mesh_2d = MeshGenerator::generate_rectangle_2d(10.0, 5.0, 4, 2);
   ```
   - Produces a grid of cells: Nx * Ny quadrilaterals.

2. **Generate a 3D rectangle**:
   ```rust
   let mesh_3d = MeshGenerator::generate_rectangle_3d(10.0, 5.0, 3.0, 4, 2, 2);
   ```
   - Produces Nx * Ny * Nz hexahedrons.

3. **Generate a circular 2D mesh**:
   ```rust
   let circle_mesh = MeshGenerator::generate_circle(1.0, 16);
   ```
   - Creates 16 divisions around a center, with triangular cells from the origin.

**Result**: A Hydra `Mesh` object with cells, vertices, and connectivity.

---

## **6. Matrix I/O with MMIO**

### Reading MatrixMarket Files

**Function**:  
```rust
fn read_matrix_market<P: AsRef<Path>>(
   file_path: P
) -> io::Result<(usize, usize, usize, Vec<usize>, Vec<usize>, Vec<f64>)>
```

**Steps**:

1. **Parses** the `%%MatrixMarket` header → determines if coordinate (sparse) or array (dense).  
2. **Reads** the size line → gets `rows, cols, nonzeros`.  
3. **Iterates** through lines:
   - **Array**: Each line is a value in row-major order.  
   - **Coordinate**: Each line has `(row col value)`.  
4. Returns a tuple with:
   1. `rows`, `cols`
   2. `nonzeros`
   3. `row_indices`, `col_indices`
   4. `values`

### Writing MatrixMarket Files

**Function**:  
```rust
fn write_matrix_market<P: AsRef<Path>>(
   file_path: P,
   rows: usize,
   cols: usize,
   nonzeros: usize,
   row_indices: &[usize],
   col_indices: &[usize],
   values: &[f64],
   is_array_format: bool,
) -> io::Result<()>
```

1. If `is_array_format`, write **array** format.  
2. Otherwise, write **coordinate** format.  
3. Writes:
   - The **header** line `%%MatrixMarket` plus format details.  
   - The **size** line(s).  
   - Each value either in row-major order (array) or `(row+1, col+1, value)` for coordinate.

**Important**: Indices in coordinate format are **1-based** in the `.mtx` specification.

---

## **7. Example Workflow**

**Scenario**: You have a 2D Gmsh file and want to read it, generate a matrix, or do some typical tasks.

```rust
use hydra::input_output::gmsh_parser::GmshParser;
use hydra::input_output::mmio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read a Gmsh mesh
    let mesh = GmshParser::from_gmsh_file("2dmesh.msh")?;

    // 2. (Optional) Generate an alternative mesh
    let test_mesh = hydra::input_output::mesh_generation::MeshGenerator
                    ::generate_rectangle_2d(10.0, 5.0, 4, 2);

    // 3. Load a matrix from a MatrixMarket file
    let (rows, cols, nnz, row_idx, col_idx, vals) = 
        mmio::read_matrix_market("stiffness.mtx")?;

    // 4. Write the same matrix to a new file
    mmio::write_matrix_market("copy_of_stiffness.mtx",
        rows, cols, nnz,
        &row_idx, &col_idx, &vals,
        /* is_array_format */ false)?;

    Ok(())
}
```

---

## **8. Best Practices**

1. **Validate** Gmsh files: Ensure elements are correct for your simulation. Unhandled element types are ignored.  
2. **Mesh Generation**:
   - Keep Nx, Ny, (Nz) at a scale feasible for your solver.  
   - Use `generate_circle(...)` carefully for large `num_divisions` to avoid too many cells.  
3. **MatrixMarket**:
   - Confirm if your matrix is dense or sparse for correct format usage.  
   - Remember coordinate format is **1-based** indexing in the .mtx file.

---

## **9. Conclusion**

The **`input_output`** module in Hydra provides:

- **Gmsh** parsing to build Hydra `Mesh` objects.  
- **Mesh generation** for rectangles or circles in 2D/3D.  
- **MatrixMarket** reading/writing for external linear algebra data.

By following these utilities, users can easily import standard or custom geometry into Hydra and save or load matrices for solver interoperability. This module significantly **streamlines** the pipeline for setting up geometry and matrix data for subsequent numerical methods in Hydra.