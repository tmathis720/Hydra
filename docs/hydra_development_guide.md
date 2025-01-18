# **Hydra Comprehensive Development Guide**

---

## **1. Introduction**

Hydra is a framework designed for **numerical simulations**, particularly those involving **partial differential equations (PDEs)** in **computational fluid dynamics (CFD)** or related fields. By combining modular structures—**Domain**, **Boundary**, **Geometry**, **Linear Algebra**, **Solver**, **Time Stepping**, **Equation**, **Interface Adapters**, **Extrusion**, **Use Cases**, and **Input/Output**—Hydra aspires to offer a cohesive environment for setting up, solving, and analyzing multi-physics problems.

This development guide serves two main purposes:

1. Summarize **current capabilities** across Hydra’s modules—what’s stable and what’s partially implemented.
2. Outline a **roadmap**—the core areas of planned or desirable improvements, expansions, and refinements.

---

## **2. Overall Hydra Architecture**

Hydra organizes functionality into **modular crates** or modules, each devoted to a specific concern:

- **Domain**: Mesh management, entity definitions (vertices, edges, faces, cells), topological connectivity (Sieve), data storage in Sections.
- **Boundary**: Defining and applying boundary conditions to mesh entities.
- **Geometry**: Calculating volumes, areas, centroids, and normals for shapes (2D/3D).
- **Linear Algebra**: Unified interfaces for vectors and matrices, bridging to `faer::Mat`, supporting sparse or dense usage.
- **Solver**: Iterative Krylov solvers (CG, GMRES) with preconditioners (Jacobi, ILU, Cholesky, LU, AMG).
- **Time Stepping**: Methods for evolving time-dependent PDEs, including explicit/implicit Euler, partial Runge-Kutta, etc.
- **Equation**: PDE definitions (momentum, energy, turbulence), plus a manager to orchestrate multiple equations and fields.
- **Interface Adapters**: Converting Hydra’s internal data (Section, Mesh) to external formats (MatrixMarket, column vectors), domain building helpers, system solver adapters.
- **Extrusion**: Converting 2D extrudable meshes into 3D (quad→hex or tri→prism).
- **Use Cases**: Higher-level patterns like matrix/RHS construction, PISO solver steps.
- **Input/Output**: Reading Gmsh meshes, generating simple shapes, loading/saving MatrixMarket files, etc.

Integration among these components is designed to be **extensible** and **thread-safe** for HPC contexts.

---

## **3. Current State of Each Module**

### 3.1 Domain

- **Status**: Feature-complete for unstructured mesh representations (vertices, edges, faces, cells).  
- **Key Features**:
  - `MeshEntity` enumerates entity types; `Sieve` organizes topological relationships.  
  - `Section<T>` stores data associated with entities.  
  - Basic reordering algorithms (Cuthill-McKee) and geometry validation exist.
- **Planned**:
  - More advanced topological refinements or 3D face inference.  
  - Better parallel domain decomposition or distributed Overlap support.

### 3.2 Boundary

- **Status**: Solid single interface (`BoundaryConditionHandler`) with an **enum** for multiple boundary condition types (Dirichlet, Neumann, Robin, etc.).  
- **Key Features**:
  - Function-based BCs for time-dependent or coordinate-based conditions.  
  - Straightforward integration with solver matrix modifications in each PDE step.
- **Planned**:
  - Potential expansions for additional specialized BCs (e.g., advanced inflow/outflow, advanced slip conditions).  
  - More thorough unit testing of boundary condition applications.

### 3.3 Geometry

- **Status**: Provides shape-based computations: 2D edges, triangles, quads; 3D tets, hexes, prisms, pyramids.  
- **Key Features**:
  - Methods to compute areas, volumes, centroids, normals.  
  - `GeometryCache` to store partial results.  
- **Planned**:
  - Additional shapes or improved caching invalidation logic.  
  - Possibly unify more thoroughly with domain Sieve for direct shape classification without user duplication.

### 3.4 Linear Algebra

- **Status**: Has a robust `Vector` and `Matrix` trait system with standard implementations (`Vec<f64>`, `faer::Mat<f64>`, `SparseMatrix`).  
- **Key Features**:
  - `VectorBuilder` / `MatrixBuilder` for creation and resizing.  
  - Basic operations: dot, norm, scale, axpy, mat-vec.  
  - Parallel operations using Rayon (dot product, etc.).
- **Planned**:
  - More advanced sparse matrix storage or better interoperability with external HPC libraries.  
  - Additional vector/matrix data structures if needed (block-sparse, GPU-based, etc.).

### 3.5 Solver

- **Status**: Krylov Subspace Solvers (CG, GMRES) plus Preconditioners (Jacobi, ILU, LU, Cholesky, AMG).  
- **Key Features**:
  - `KSP` trait unifies solver usage, `SolverManager` integrates solver + preconditioner.  
  - Uses parallel loops with Rayon in matrix-vector ops.  
- **Planned**:
  - More advanced error detection, flexible restarts in GMRES, or multi-right-hand-side solvers.  
  - More robust incomplete factorization thresholds, improved AMG coarsening heuristics.

### 3.6 Time Stepping

- **Status**: Forward Euler, Backward Euler, partial Runge-Kutta; a skeleton for adaptivity.  
- **Key Features**:
  - `TimeStepper` trait (`step`, `adaptive_step`), used by PDE managers.  
  - Some partial examples of local error estimation for adaptive dt.  
- **Planned**:
  - Completion of Runge-Kutta methods with full Butcher table support.  
  - Implementation of Crank-Nicolson or multi-step integrators.  
  - More robust adaptive stepping.

### 3.7 Equation

- **Status**: Example PDEs (MomentumEquation, EnergyEquation), plus `EquationManager`.  
- **Key Features**:
  - `PhysicalEquation` trait, flux-based assembly for momentum, energy, turbulence.  
  - Fields, fluxes, gradient reconstructions, flux limiters.  
- **Planned**:
  - Expand PDE library (e.g. compressible Navier–Stokes, multi-phase flows).  
  - Further gradient methods or advanced flux limiters (WENO, PPM) integration.  
  - Additional PDE coupling (e.g., magnetohydrodynamics, chemical reactions).

### 3.8 Interface Adapters

- **Status**: `vector_adapter`, `matrix_adapter`, `section_matvec_adapter` for bridging Hydra data with standard HPC data structures. `system_solver` for reading MatrixMarket. `domain_adapter` for programmatic domain building.  
- **Key Features**:
  - Simplifies external solver usage or domain creation.  
  - `SystemSolver` can parse .mtx and run Hydra’s KSP.  
- **Planned**:
  - Additional adapter logic for more HPC libraries or direct domain partitioners.  
  - Possibly unify domain adapter with advanced geometry validation or doping in boundary conditions automatically.

### 3.9 Extrusion

- **Status**: Allows 2D quadrilateral or triangular meshes to be extruded into 3D hexahedrons or prisms.  
- **Key Features**:
  - `ExtrudableMesh` trait, `QuadrilateralMesh` / `TriangularMesh`, plus a `ExtrusionService`.  
  - Utility to extrude in the z-direction with a fixed number of layers.  
- **Planned**:
  - More flexible extrusions (non-uniform layering, revolve extrusions, etc.).  
  - Additional dimension checks or advanced geometry transformations.

### 3.10 Use Cases

- **Status**: Contains higher-level workflows:
  - `matrix_construction` / `rhs_construction` for building typical system matrices or vectors.  
  - `piso` submodule for the Pressure-Implicit with Splitting of Operators approach (predictor, pressure correction, velocity correction).  
- **Key Features**:
  - `piso::nonlinear_loop` for iterative solution of incompressible flows.  
  - `piso::boundary` for specialized boundary modifications.  
- **Planned**:
  - Additional “one-click” solution patterns, e.g. a *domain + PDE + boundary + solver* script.  
  - Extend PISO to more robust multi-phase or compressible flows if needed.

### 3.11 Input/Output

- **Status**: Gmsh reading, standard shape generation, MatrixMarket I/O.  
- **Key Features**:
  - `GmshParser` for .msh → Hydra `Mesh`.  
  - `MeshGenerator` for simple domains.  
  - `mmio` for reading/writing .mtx files.  
- **Planned**:
  - Expand Gmsh parser to handle more element types (tetrahedron, wedge, etc. in 3D).  
  - Additional or improved writing tools for 3D if .msh version requires volume cell definitions.

---

## **4. Roadmap for Future Development**

### 4.1 Short-Term Goals

1. **Complete Adaptive Time Stepping** in `time_stepping`:
   - Implement step-size control fully in `ExplicitEuler` or `RungeKutta`.
   - Provide example PDE codes demonstrating adaptivity.

2. **Refine/Expand PDEs in `Equation`**:
   - Possibly add compressible Navier–Stokes example or multi-phase momentum equation.
   - Strengthen the turbulence models (k–ε, k–ω) with robust calibrations.

3. **Enhance Preconditioners**:
   - In ILU/AMG, add threshold-based or drop-tolerance strategies.
   - Include advanced partial pivot strategies or robust block-based approaches for larger HPC contexts.

4. **Extended Gmsh Support**:
   - Parse more 3D element types (tetrahedron, wedge, pyramid) directly from .msh without ignoring them.
   - Validate the topology is consistent.

### 4.2 Medium-Term Goals

1. **Runge-Kutta + Crank-Nicolson**:
   - Provide fully tested 2nd–4th order Runge-Kutta schemes, error controllers, Butcher table support.
   - Implement Crank-Nicolson for a second-order implicit approach.

2. **Parallel/Distributed Domain**:
   - Possibly integrate domain partitioners, enabling Hydra to handle multi-rank MPI or GPU-accelerated contexts.
   - Extend Overlap data structures or boundary exchanges for domain decomposition.

3. **Advanced HPC**:
   - Incorporate specialized HPC libraries for better sparse matrix (Block ILU, advanced AMG frameworks, GPU backends).
   - Deeper concurrency at PDE assembly or multi-level domain expansions.

4. **Higher-Order Reconstruction**:
   - Expand on flux limiters, WENO, PPM in `equation::reconstruction` for more advanced fluid dynamics solutions.

### 4.3 Long-Term Aspirations

1. **Robust Multi-Physics Coupling**:
   - Provide standard modules for heat transfer, electromagnetics, chemistry, etc., integrated with momentum and turbulence.
2. **Large Library of PDEs**:
   - Offer a “one-stop” PDE solver environment covering a wide range of physical processes.
3. **Full HPC Ecosystem**:
   - Automatic domain partitioning, multi-GPU support, advanced I/O pipelines for extremely large simulations.

---

## **5. Development Process and Collaboration**

- Each module has a dedicated **user guide** and **tests**.  
- **Regular merges** with an emphasis on thread safety, HPC performance, and PDE correctness.  
- For new PDEs or advanced features (like a new preconditioner or time step method), follow:
  1. **Prototype** in a branch.
  2. Add or update **user guides** and **tests**.
  3. Undergo code review for style, concurrency, integration, and HPC readiness.
- Documentation remains in sync with code changes, ensuring the user’s first experience is consistent.

---

## **6. Conclusion**

Hydra is in a **solid** intermediate stage, offering a well-structured approach to domain-based PDE solving:

- **Core** modules are functional and tested (Domain, Boundary, Geometry, Linalg, Solver).  
- **Advanced** methods (time stepping adaptivity, HPC distributed domain, higher-order PDE expansions) are partially done or planned.  
- The framework is highly **extensible**, letting developers add new PDEs, new reconstruction or solver components, or specialized boundary treatments.

Moving forward, the roadmap highlights more robust **time adaptivity**, broader PDE coverage, improved HPC distribution, and expanded I/O and domain handling for 3D unstructured meshes. By following these development goals, Hydra will continue evolving into a more comprehensive, high-performance environment for complex numerical simulations.