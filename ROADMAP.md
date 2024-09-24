# Project Roadmap

## Overview

The primary goal of this project is to develop a robust Finite Volume Method (FVM) solver using Rust. The solver aims to:

- **Accurately Solve Partial Differential Equations (PDEs):** Implement numerical methods to solve PDEs over complex geometries and meshes.
- **Flexible Mesh Representation:** Create a mesh data structure capable of representing various elements like cells, faces, edges, and vertices, along with their hierarchical relationships.
- **Boundary Condition Handling:** Incorporate Dirichlet and Neumann boundary conditions to define values and fluxes on the boundaries of the computational domain.
- **Geometric Computations:** Develop a geometry module to compute areas, volumes, centroids, and distances essential for the FVM discretization.
- **Efficient Solver Implementation:** Assemble and solve linear systems resulting from the discretization efficiently, ensuring scalability and performance.
- **Testing and Validation:** Establish a suite of unit tests and validation cases to ensure the correctness and reliability of the solver.

---

## Progress So Far

### 1. Geometry Handling (`src/geometry/` and Its Modules)

We have successfully made significant strides in the geometry handling module, specifically targeting the representation and computation of centroids and volumes for various mesh elements such as tetrahedrons, prisms, pyramids, and hexahedrons. These elements are critical in defining complex unstructured meshes and setting the groundwork for computational operations like the Finite Volume Method (FVM) and Finite Element Method (FEM).

**Key Accomplishments:**

- **Basic Geometry Handling Module (`src/geometry/mod.rs`):**
  - Structured geometric operations around various 3D shapes (tetrahedrons, hexahedrons, prisms, pyramids).
  - Implemented computation of centroids and volumes for each shape using both analytical and numerical methods (e.g., splitting pyramids into tetrahedrons for more accurate volume and centroid calculations).
  - Developed methods for computing face areas and centroids, including handling of triangles and quadrilaterals.

- **Shape-Specific Modules:**
  - Each 3D shape module (e.g., `tetrahedron.rs`, `pyramid.rs`) contains:
    - Functions to compute centroid and volume.
    - Methods to divide complex elements into simpler sub-elements for reliable calculations.
  - Robust unit tests ensure the correctness of these calculations, though further adjustments and debugging are required for edge cases (like degenerate shapes).

- **Testing and Debugging:**
  - Extensive testing helped identify failures in pyramid centroid and volume calculations. Issues were traced back to improper weighting and centroid calculations, particularly in degenerate cases.
  - Adopted numerical approaches, such as splitting shapes into simpler sub-elements, for more reliable calculations.

### 2. Mesh Entity Management and Sieve Data Structure

We have solidified the management of mesh entities and defined relationships between these entities.

**Core Operations:**

- **Mesh Entities:**
  - Defined mesh entities such as vertices, edges, faces, and cells using enums and arrows (incidence relationships between entities).
  - Organized entities using a sieve data structure, effectively handling hierarchical relationships between mesh entities.

- **Sieve Data Structure:**
  - Implemented operations such as `cone`, `closure`, `support`, and `star` to capture hierarchical and topological relationships.
  - These operations are critical for PDE solvers requiring efficient access to neighboring or related entities.
  - Implemented `meet` and `join` operations to handle minimal separators for closures and stars, useful in stratified meshes.

### 3. Boundary Conditions (`src/boundary/`)

We developed modules to handle boundary conditions, integrating them with the solver.

**Key Functionality:**

- **Dirichlet Boundary Conditions (`DirichletBC`):**
  - Utilized `HashMap<MeshEntity, f64>` to map mesh entities to prescribed values.
  - Implemented methods `is_bc` and `get_value` to check and retrieve boundary condition values.
  - Updated the `apply_bc` method to modify the system matrix and RHS vector appropriately.

- **Neumann Boundary Conditions (`NeumannBC`):**
  - Used `HashMap<MeshEntity, f64>` to associate mesh entities with flux values.
  - Added methods `is_bc` and `get_flux` for checking and retrieving fluxes.
  - Adjusted the `apply_bc` method to correctly update the RHS vector based on flux values and face areas.

### 4. Section Data Management (`section.rs`)

The `Section` structure was implemented to allow association of data (e.g., vertex coordinates, element values) with mesh entities.

**Key Functionality:**

- Implemented `set`, `restrict`, and `update` data functions to associate, retrieve, and update data related to specific entities.
- Ensured efficient storage by using contiguous arrays, improving performance when dealing with large meshes.

### 5. Solver Integration (`src/solver/fvm_solver.rs`)

We have integrated the mesh infrastructure with the FVM solver.

**Key Accomplishments:**

- **FVMSolver Structure:**
  - Developed the `FVMSolver` struct to handle assembling and solving the linear system resulting from FVM discretization.
  - Implemented methods for assembling the system matrix and RHS vector, incorporating geometry computations and boundary conditions.

- **Geometry Computations in Solver:**
  - Integrated the geometry module with the solver to compute necessary geometric quantities like cell volumes, face areas, and centroids.

- **Boundary Condition Application:**
  - Applied Dirichlet and Neumann boundary conditions within the solver, ensuring they correctly modify the system matrix and RHS vector.

### 6. Error Handling and Robustness

- **Compilation and Runtime Errors:**
  - Resolved borrowing and ownership issues by adjusting function return types and dereferencing appropriately.
  - Ensured that geometry computation methods are correctly called on instances of the `Geometry` struct.

- **Test Failures:**
  - Addressed test failures by refining mesh definitions in tests and adjusting geometry computations to handle 1D elements.
  - Updated the `FVMSolver` and geometry functions to process 1D mesh elements appropriately.

### 7. Unit Tests and Validation

**Progress in Testing and Validation:**

- Unit tests have been written for the geometry module and other modules like sieve, section, and solver.
- Debugging and adjustments continue for edge cases, particularly with degenerate geometries.
- Test coverage ensures that changes in code are validated against expected outcomes.

---

## Next Steps

### Immediate Priorities

1. **Finalize Geometry Handling:**
   - Complete debugging of pyramid centroid and volume calculations to robustly handle edge cases.
   - Implement optimized methods for handling prisms and hexahedrons using analytical methods for efficiency.

2. **Enhance Boundary Condition Handling:**
   - Expand boundary condition modules to support more complex scenarios and mixed types.
   - Implement error handling to gracefully manage missing or invalid boundary condition data.

3. **Extend Solver Capabilities:**
   - Incorporate support for 2D and 3D meshes in the solver.
   - Implement more efficient solvers or integrate with external libraries for larger systems (e.g., iterative solvers).

4. **Performance Optimization:**
   - Profile key areas such as adjacency lookups, reordering algorithms, and mesh partitioning to ensure scalability.
   - Optimize data structures and algorithms to improve computational efficiency.

### Longer-Term Objectives

1. **Parallelization and Scalability:**
   - Leverage Rust's concurrency features to parallelize computations where possible.
   - Explore distributed computing options for handling large-scale simulations, potentially using MPI or other frameworks.

2. **Advanced Mesh Features:**
   - Implement adaptive mesh refinement techniques to improve solution accuracy in regions with high gradients.
   - Add support for more complex cell and face shapes, such as polygons and polyhedra.

3. **Comprehensive Testing and Validation:**
   - Develop integration tests that combine multiple components to ensure they work together in real-world scenarios.
   - Validate solver results against analytical solutions or benchmark problems to verify accuracy.

4. **Documentation and Usability:**
   - Continue to expand inline documentation for each module to ensure clear usage and facilitate future maintenance.
   - Create a user guide with examples of mesh creation, data association, and geometry computations.
   - Develop a user-friendly interface or API for setting up simulations, defining meshes, and applying boundary conditions.

5. **Community Engagement:**
   - Consider open-sourcing the project to invite collaboration, code reviews, and contributions from the community.
   - Engage with potential users or stakeholders to gather feedback on features, usability, and performance.

6. **Error Handling and Robustness:**
   - Replace `panic!` calls with proper error handling using `Result` and custom error types.
   - Implement input validation to ensure robustness against invalid or corrupted data.

---

## Challenges and Focus Areas

- **Parallelization and Performance:** Managing data consistency and efficiency at scale will be key challenges as we move into solver integration and larger meshes.
- **Handling Complex Geometries:** Ensuring that degenerate and complex shapes (e.g., concave polyhedra) are handled efficiently is a focus for the geometry module.
- **Error Handling:** Implementing comprehensive error handling to make the solver robust and user-friendly.

---

By focusing on these areas, we will develop a robust, scalable, and efficient framework for unstructured mesh management and computation. The project aims to provide a valuable tool for solving PDEs using the Finite Volume Method in Rust, suitable for both academic research and practical engineering applications.

---

*This roadmap is intended to provide a clear understanding of the project's framework, progress to date, and future plans. It serves as a guide for contributors and stakeholders to align efforts and expectations.*