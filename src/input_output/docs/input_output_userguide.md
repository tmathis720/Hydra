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