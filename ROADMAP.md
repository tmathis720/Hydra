# HYDRA: Short-Term Roadmap for Incremental Test-Driven Development (TDD)

HYDRA is a high-performance Rust-based hydrodynamic modeling system designed to simulate complex geophysical phenomena, particularly for surface water bodies. The focus of the project is on ensuring computational efficiency, stability, and flexibility through Rust's concurrency and memory safety features. The project employs a staggered unstructured finite-volume method (FVM) to solve the Reynolds-averaged Navier-Stokes equations (RANS) in two- or three-dimensional boundary-fitted domains.

## Key Priorities:

1. Enhance core simulation stability by refining numerical methods.
2. Extend boundary conditions and solvers for a broader range of geophysical phenomena.
3. Develop testing strategies to ensure model robustness.
4. Optimize computational performance, leveraging Rust's unique capabilities.

## Short-Term Milestones

**Numerical Core Development**

Status: Basic solvers and boundary handling implemented.
Tasks:

    Refactor the numerical module to enhance modularity and readability.
    Extend the solver module to handle adaptive time-stepping and implicit solver methodologies.
    Improve timestep module to ensure compatibility with new solver methods.
    Add turbulence models for simulating marine and atmospheric flows (linking with approaches like k-ε models)​.

**Boundary and Domain Handling**

Status: Basic boundary condition implementation in the boundary module.
Tasks:

    Expand boundary module to support more complex boundary conditions, such as non-slip, free-surface, and open boundary conditions commonly used in marine simulations​.
    Refactor domain to support multi-region simulations for handling complex geophysical domains.

**Input Handling and Data Management**

Status: Initial input framework in the input directory.
Tasks:

    Enhance the input module to support multiple file formats (e.g., NetCDF, HDF5).
    Add preprocessing tools for input data validation and grid generation.

**Transport and Solver Updates**

Status: Initial implementation exists in the transport and solver modules.
Tasks:

    Add support for advanced transport models, including diffusive and dispersive processes for contaminants and tracers.
    Integrate with external libraries or tools for large-scale linear solvers, potentially leveraging PETSc​.
    Implement adaptive solver techniques, enhancing model performance on multi-core and distributed systems.

**Time-Stepping Methods**

Status: Basic time-stepping strategy implemented.
Tasks:

    Integrate error-controlled adaptive time-stepping methods.
    Ensure stability and efficiency of solvers in combination with multi-scale phenomena like turbulence and coastal boundary layer effects​​.

**Testing and Continuous Integration**

Status: Basic tests exist.
Tasks:

    Expand unit and integration tests in the tests module, focusing on solver robustness and accuracy.
    Automate testing for a variety of inputs, including edge cases like complex bathymetries and extreme boundary conditions.
    Create benchmark tests for comparison with legacy systems like FVCOM and Delft3D​.

## Long-Term Vision

**Full 3D Model Support**

Extend from 2D shallow-water models to full 3D RANS models.
Add multi-threaded support and MPI parallelism for large-scale distributed computing environments.

**GPU Acceleration**

Investigate GPU acceleration for heavy computational tasks (e.g., linear solvers and matrix assembly).
Leverage Rust’s interoperability with CUDA or OpenCL for optimized performance.

**Advanced Physical Models**

Incorporate advanced geophysical models, such as sediment transport, biogeochemical cycles, and ecosystem dynamics, with the transport and domain modules.