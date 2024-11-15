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