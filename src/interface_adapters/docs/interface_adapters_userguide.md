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