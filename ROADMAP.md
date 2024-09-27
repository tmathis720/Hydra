# Project Roadmap

## Overview

The HYDRA project aims to develop a robust **Finite Volume Method (FVM)** solver using Rust, specifically designed for **geophysical fluid dynamics problems** such as those found in **coastal areas, estuaries, rivers, lakes, and reservoirs**. The solver focuses on solving the **Reynolds-Averaged Navier-Stokes (RANS) equations**, which are essential for simulating fluid flow in these complex environments.

Our approach is inspired by the structural and functional organization of the **PETSc** library. Key PETSc modules—such as `DMPlex` for mesh management, `KSP` for linear solvers, `PC` for preconditioners, and `TS` for time-stepping—serve as models for organizing our solvers, mesh infrastructure, and preconditioners. The ultimate goal is to create a **modular, scalable solver framework** that can handle the complexity of RANS equations for geophysical applications.

---

## Progress So Far

### 1. Geometry Handling (`src/geometry/`)

We have made significant strides in the geometry handling module, which is crucial for defining complex unstructured meshes and performing computational operations in the FVM.

**Key Accomplishments:**

- **Shape-Specific Modules:**
  - Implemented modules for various 3D shapes (`tetrahedron.rs`, `hexahedron.rs`, `prism.rs`, `pyramid.rs`) containing functions to compute centroids and volumes.
  - Developed methods for computing face areas and centroids, including handling of triangles and quadrilaterals.
  - Ensured robustness by handling edge cases and degenerate geometries through numerical approaches.

- **Testing and Debugging:**
  - Extensive unit tests validate the correctness of geometric computations.
  - Adjusted methods to handle 1D and 2D elements appropriately, ensuring flexibility across different mesh dimensions.

### 2. Mesh Entity Management and Sieve Data Structure (`src/domain/`)

We have solidified the management of mesh entities and defined relationships between them, mirroring PETSc's `DMPlex` module.

**Core Components:**

- **Mesh Entities (`MeshEntity`):**
  - Represents vertices, edges, faces, and cells using a lightweight enum structure.
  - Each entity is uniquely identified by its type and ID.
  - **Note:** We intentionally kept `MeshEntity` lightweight without additional data fields or tags.

- **Sieve Data Structure (`Sieve`):**
  - Manages adjacency and incidence relationships between mesh entities.
  - Implements hierarchical operations like `cone`, `closure`, `support`, and `star` to navigate the mesh topology.
  - Facilitates efficient traversal and manipulation of the mesh structure.

- **Sections (`section.rs`):**
  - A generic data structure used to associate arbitrary data with mesh entities.
  - Acts as a mapping from `MeshEntity` to data of any type `T`.
  - Enables the association of **tags**, **functions**, **physical properties**, and other metadata with mesh entities without modifying the entities themselves.

- **Overlap and Parallelism (`overlap.rs`):**
  - Manages relationships between local and ghost entities for parallel computations.
  - Ensures data consistency across partitions in distributed computing environments.

- **Stratification and Reordering:**
  - Organizes mesh entities into strata based on their dimension.
  - Implements reordering algorithms (e.g., Cuthill-McKee) to improve memory locality and solver performance.

**Key Accomplishments:**

- **Data Association via Sections:**
  - Leveraged `Section` to associate data such as coefficient functions, boundary conditions, and tags with mesh entities.
  - Enabled flexible tagging and function association without modifying `MeshEntity` or `Mesh`.

- **Regions and Tags:**
  - Defined regions and boundaries by mapping sets of `MeshEntity` to region names or tags using `Section` or mapping structures.
  - Facilitated the application of different physical properties or boundary conditions based on regions.

### 3. Boundary Conditions (`src/boundary/`)

We developed modules to handle boundary conditions, integrating them seamlessly with the solver.

**Key Functionality:**

- **Dirichlet Boundary Conditions (`DirichletBC`):**
  - Utilized `Section` to map mesh entities to prescribed values.
  - Implemented methods to check and retrieve boundary condition values.
  - Applied boundary conditions by modifying the system matrix and RHS vector appropriately.

- **Neumann Boundary Conditions (`NeumannBC`):**
  - Used `Section` to associate mesh entities with flux values.
  - Adjusted the RHS vector based on flux values and face areas during assembly.

### 4. Solver Integration (`src/solver/`)

We have integrated the mesh infrastructure with the solver modules, mirroring PETSc's `KSP` module.

**Core Components:**

- **KSP Trait (`KSP`):**
  - Defines a common interface for all Krylov subspace solvers.
  - Ensures that different solvers can be used interchangeably.

- **Conjugate Gradient Solver (`ConjugateGradient`):**
  - Implements the Conjugate Gradient method for symmetric positive-definite systems.
  - Supports optional preconditioning through the `Preconditioner` trait.
  - Integrates with the `Domain` module for matrix and vector operations.

- **Preconditioners (`preconditioner/`):**
  - Defines an interface for preconditioners used to accelerate solver convergence.
  - Implemented **Jacobi** and **LU** preconditioners.
  - Can be applied without modifying the underlying matrix or solver structures.

**Key Accomplishments:**

- **Solver Implementation:**
  - Developed the `ConjugateGradient` solver with support for preconditioners.
  - Created unit tests to validate solver functionality with and without preconditioners.
  - Ensured robust error handling for singular or ill-conditioned matrices.

- **Matrix and Vector Traits:**
  - Abstracted over different matrix and vector types.
  - Allowed the solver to work with various data structures, including those from external crates like `faer`.

### 5. Time-Stepping Framework (`src/time_stepping/`)

We have introduced a PETSc `TS`-like framework for time-stepping, integrating it with both the `Domain` and `KSP` modules.

**Core Components:**

- **TimeStepper Trait (`TimeStepper`):**
  - Defines the interface for time-stepping methods.
  - Supports both explicit and implicit schemes.

- **TimeDependentProblem Trait (`TimeDependentProblem`):**
  - Represents the ODE/DAE problem to be solved.
  - Allows users to define custom functions for initial conditions, boundary conditions, source terms, and coefficients.

- **Implementations of Time-Stepping Methods:**
  - **Forward Euler**: An explicit method suitable for simple problems.
  - **Runge-Kutta Methods**: Higher-order explicit methods.
  - **Implicit Methods**: Such as Backward Euler and Crank-Nicolson, which require solving linear systems at each time step.

**Key Accomplishments:**

- **Integration with Mesh and Solvers:**
  - The `TimeDependentProblem` accesses mesh entities and associated data via `Section`.
  - Supports spatially varying coefficients and complex physical behaviors.
  - Implicit methods utilize the `KSP` module to solve linear systems arising from discretization.

- **User-Defined Functions:**
  - Enabled users to define custom functions for physical properties, associated with mesh entities through `Section`.

- **Example Usage:**
  - Demonstrated how to set up and run a time-stepping simulation using the framework.

### 6. Error Handling and Robustness

- **Error Handling:**
  - Replaced `panic!` calls with proper error handling using `Result` and custom error types.
  - Implemented input validation to ensure robustness against invalid or corrupted data.
  - Ensured that the solver reports non-convergence appropriately.

- **Testing and Validation:**
  - Developed a suite of unit tests for the geometry, domain, solver, and time-stepping modules.
  - Validated the correctness and performance of the framework through extensive testing.

---

## Next Steps

### Immediate Priorities

1. **Further Testing and Validation:**

   - **Expand Test Coverage:**
     - Test the solver and time-stepping methods on larger and more complex problems.
     - Validate solver performance and convergence rates with realistic geophysical scenarios.

   - **Integration Tests:**
     - Develop tests that combine multiple components to ensure they work together seamlessly.

2. **Performance Optimization:**

   - **Profiling:**
     - Profile the code to identify bottlenecks in computation and memory usage.
     - Optimize data structures and algorithms for better performance.

   - **Efficient Data Structures:**
     - Explore more efficient storage and access patterns in `Section` and `Sieve`.

3. **Solver Extensions:**

   - **Additional Solvers:**
     - Implement solvers like GMRES for non-symmetric systems.
     - Explore multi-grid or other advanced iterative methods.

   - **Advanced Preconditioners:**
     - Develop more complex preconditioners (e.g., ILU, multigrid) to improve convergence.

4. **Enhance Time-Stepping Framework:**

   - **Adaptive Time-Stepping:**
     - Implement error estimation and step-size control for adaptive methods.

   - **Event Handling:**
     - Add capabilities to handle events and control the integration process.

5. **Boundary Condition Handling:**

   - **Modular Boundary Conditions:**
     - Develop flexible mechanisms to apply various boundary conditions using `Section`.
     - Support mixed boundary conditions and complex scenarios.

### Longer-Term Objectives

1. **Parallelization and Scalability:**

   - **Distributed Computing:**
     - Integrate parallel computation features using MPI or Rust's concurrency tools.
     - Ensure that `Overlap` and `Delta` structures effectively manage data across processes.

   - **Scalability Testing:**
     - Test the framework on large-scale simulations to assess scalability.

2. **Geophysical Fluid Dynamics Application:**

   - **Incorporate RANS Equations:**
     - Extend `TimeDependentProblem` implementations to include the RANS equations.
     - Handle turbulence models and other complexities inherent in geophysical flows.

   - **Mesh Generation and Adaptivity:**
     - Implement tools for mesh generation tailored to geophysical domains.
     - Explore adaptive mesh refinement techniques to improve solution accuracy.

3. **Documentation and User Guidance:**

   - **Comprehensive Documentation:**
     - Document APIs, traits, and modules thoroughly.
     - Provide usage examples and best practices.

   - **Tutorials and Guides:**
     - Write tutorials to help new users get started.
     - Explain how to define custom problems and integrate them with the solver.

4. **Community Engagement:**

   - **Open Source Collaboration:**
     - Consider open-sourcing the project to invite contributions and feedback.
     - Engage with the community through forums, workshops, or conferences.

5. **Error Handling and Robustness:**

   - **Robust Error Propagation:**
     - Implement comprehensive error handling to make the solver robust and user-friendly.
     - Use custom error types and `Result` to propagate errors gracefully.

6. **Advanced Mesh Features:**

   - **Complex Geometries:**
     - Add support for more complex cell and face shapes, such as polygons and polyhedra.
     - Ensure that degenerate and complex shapes are handled efficiently.

---

## Challenges and Focus Areas

- **Parallelization and Performance:**
  - Managing data consistency and efficiency at scale is a key challenge.
  - Ensuring that the framework scales well with increasing problem size and computational resources.

- **Handling Complex Geometries:**
  - Accurately representing and computing on complex or degenerate geometries.
  - Balancing computational efficiency with geometric accuracy.

- **User-Friendly Interfaces:**
  - Developing intuitive APIs and interfaces for setting up simulations.
  - Providing tools and documentation to lower the barrier to entry for new users.

---

## Project Milestones

1. **Milestone 1: Foundation Completion**
   - Finalize the core modules (`Domain`, `KSP`, `Time-Stepping`).
   - Achieve stable versions of the key components with comprehensive unit tests.

2. **Milestone 2: Solver Integration**
   - Successfully integrate the solver with test problems, demonstrating end-to-end functionality.
   - Validate solver performance and correctness on benchmark problems.

3. **Milestone 3: Parallelization**
   - Implement distributed computing capabilities.
   - Demonstrate scalability on multi-node systems.

4. **Milestone 4: Geophysical Application**
   - Incorporate the RANS equations and associated physics.
   - Run simulations relevant to geophysical fluid dynamics with realistic scenarios.

5. **Milestone 5: Community Release**
   - Prepare the codebase for open-source release.
   - Provide documentation, tutorials, and user support channels.

---

## Conclusion

By focusing on these areas, we aim to develop a robust, scalable, and efficient framework for unstructured mesh management and computation, suitable for solving complex geophysical fluid dynamics problems using the Finite Volume Method in Rust. The project's modular design ensures flexibility, extensibility, and performance, making it valuable for both academic research and practical engineering applications.

---

*This roadmap provides a clear understanding of the project's framework, progress to date, and future plans. It serves as a guide for contributors and stakeholders to align efforts and expectations.*