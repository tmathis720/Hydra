# HYDRA: Finite Volume Solver for Geophysical Fluid Dynamics

## Project Overview

The HYDRA project aims to develop a **Finite Volume Method (FVM)** solver for **geophysical fluid dynamics (GFD)**, focusing on environments like coastal zones, estuaries, rivers, lakes, and reservoirs. The solver is designed to solve the **Reynolds-Averaged Navier-Stokes (RANS) equations**, a widely-used framework for simulating fluid flow in these domains. The ultimate goal is to model complex GFD problems efficiently and accurately, leveraging state-of-the-art numerical techniques and parallel computing.

HYDRA takes inspiration from **PETSc**, adopting a similar structure and functional approach to handle tasks like mesh management, parallelization, and solver routines. The project emphasizes flexibility, modularity, and scalability to address the computational challenges of geophysical simulations.

## Key Features and Goals

- **FVM Solver**: Targeting 3D fluid dynamics problems in complex domains using the FVM approach.
- **RANS Equations**: Solving the Reynolds-Averaged Navier-Stokes equations for turbulent flow.
- **Geophysical Focus**: Applications in environmental fluid dynamics, including modeling of natural water bodies.
- **Modular and Extensible**: Designed with flexibility in mind, allowing users to easily extend the framework with additional solvers, preconditioners, and boundary conditions.
- **Parallel Computing**: Future versions will support parallel computing using MPI and Rust’s concurrency features.
- **PETSc-Inspired**: Following PETSc’s approach, we adopt structured modules for solvers (e.g., KSP for Krylov subspace solvers), preconditioners (e.g., Jacobi), and mesh management.

## Repository Structure

- `.github/`: GitHub-specific configuration files.
- `.vscode/`: Visual Studio Code configuration.
- `Cargo.toml`: Cargo configuration file for Rust.
- `Cargo.lock`: Dependency lock file.
- `src/`: Contains the source code for the HYDRA project.
  - **solver/**: Implements various solvers and preconditioners.
    - `cg.rs`: Conjugate Gradient solver implementation.
    - `jacobi.rs`: Jacobi preconditioner.
    - `ksp.rs`: Krylov subspace solver manager.
  - **domain/**: Contains modules for mesh management, including:
    - `mesh_entity.rs`: Defines the basic geometric entities.
    - `sieve.rs`: For managing the structure and relationship between entities.
    - `section.rs`: For relating data to mesh entities.
  - **boundary/**: Handles boundary conditions (e.g., Dirichlet, Neumann).
- `test.msh2`: Sample mesh file for testing.
- `README.md`: This file.
- `ROADMAP.md`: Future development plans and goals for HYDRA.

## Current Progress

### Implemented Components

1. **Conjugate Gradient (CG) Solver**: A Krylov subspace solver for solving symmetric positive definite systems, integrated with the option to use preconditioners.
   - **Jacobi Preconditioner**: Applied to improve the convergence of the CG solver, especially for ill-conditioned systems.
   - **Singular Matrix Detection**: Added detection for non-convergence when the matrix is singular or nearly singular.
   
2. **Unit Testing**: Comprehensive tests for both the CG solver and the Jacobi preconditioner have been implemented.
   - **Tests for singular matrices**: Verifying that the solver correctly fails for non-invertible systems.
   - **Preconditioner tests**: Validating the functionality of the Jacobi preconditioner on simple matrix systems.

### Work in Progress

- **Boundary Conditions**: Implementing modules for handling various types of boundary conditions (Dirichlet, Neumann).
- **Solver Extensions**: Exploring additional solvers like GMRES for non-symmetric systems and other preconditioners.
- **Performance Optimization**: Profiling and optimizing key components, including solver convergence and matrix assembly efficiency.

## Future Plans

The next steps for HYDRA include:

1. **Parallel Computing**: Implementing support for distributed meshes and parallel solvers using MPI or Rust concurrency.
2. **Mesh Infrastructure**: Developing a robust mesh management system inspired by PETSc's `DMPlex`, allowing for efficient handling of complex geometries and partitioning for parallel processing.
3. **RANS Equations**: Integrating the full Reynolds-Averaged Navier-Stokes equations into the solver framework for modeling turbulent fluid flow.
4. **Advanced Preconditioning**: Adding more advanced preconditioners like ILU and multi-grid methods.
5. **Documentation and User Guide**: Expanding documentation for users and developers, including a detailed user guide and API references.

## Getting Started

### Prerequisites

To build and run the HYDRA project, you need the following:

- **Rust**: The programming language used for HYDRA. Install it from [rust-lang.org](https://www.rust-lang.org/).
- **Cargo**: The Rust package manager, used to build and manage dependencies.
- **MPI (optional)**: For parallel computing support in future releases.

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/tmathis720/HYDRA.git
   cd HYDRA
   ```

2. Build the project:
   ```bash
   cargo build
   ```

3. Run tests to verify the setup:
   ```bash
   cargo test
   ```

## Contributions

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Submit a pull request with detailed descriptions of changes.

For more details, see the `CONTRIBUTING.md` file.

## License

HYDRA is licensed under the MIT License. See `LICENSE` for more information.

---

**HYDRA** is actively developed and maintained with a focus on advancing the state of fluid dynamics simulation, particularly for complex geophysical applications.