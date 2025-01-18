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