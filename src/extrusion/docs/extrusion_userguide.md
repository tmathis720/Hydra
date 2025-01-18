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