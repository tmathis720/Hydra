### Project Overview and Goal

The overall purpose of the HYDRA project is to develop a **Finite Volume Method (FVM)** solver for **geophysical fluid dynamics problems**, specifically targeting environments like **coastal, estuary, riverine, lakes, and reservoirs**. The project involves solving the **Reynolds-Averaged Navier-Stokes (RANS) equations**, which are used to simulate fluid flow in these environments.

Our approach closely follows the structural and functional organization of the **PETSc** library, particularly in the areas of mesh management, parallelization, and solver routines. PETSc’s key modules (like `DMPlex`, `KSP`, `PC`, and `TS`) serve as inspiration for organizing our solvers, mesh infrastructure, and preconditioners. The ultimate goal is to create a modular, scalable solver framework that can handle the complexity of RANS equations for geophysical applications.

### Summary of Work Done

In this conversation, our focus has been on implementing key components of the **Conjugate Gradient (CG)** solver and integrating it with **Jacobi preconditioning**, and revising our approach using the `faer` crate for linear algebra capabilities, with the aim of developing a flexible and extensible solver infrastructure. The following tasks have been completed:

1. **Conjugate Gradient Solver Implementation**:
   - Developed the core `ConjugateGradient` solver (`cg.rs`) to iteratively solve linear systems.
   - Integrated an option for **preconditioners**, which can be passed as closures to the solver.
   - Ensured modularity in solver design, following the structure of PETSc’s `KSP` module.

2. **Jacobi Preconditioner**:
   - Implemented a **Jacobi preconditioner** in `jacobi.rs`, both as an instance method and a static method.
   - Developed functionality for the preconditioner to handle simple diagonal matrices, including cases where diagonal entries are zero (handled by panicking or error reporting).

3. **Unit Testing**:
   - **Jacobi Preconditioner**: Developed tests for applying the Jacobi preconditioner, including tests for edge cases like division by zero.
   - **Conjugate Gradient Solver**:
     - Tested the CG solver on small symmetric positive definite (SPD) systems without preconditioning.
     - Added tests for the CG solver with the Jacobi preconditioner.
     - Created a test to ensure the solver fails gracefully (i.e., does not converge) when dealing with **singular matrices** (ill-conditioned systems).

4. **Error Detection in Singular Matrices**:
   - Enhanced the `ConjugateGradient` solver to detect **non-convergence** due to singular or ill-conditioned matrices.
   - Added checks for:
     - Small denominator values in the CG algorithm, which can indicate a singular matrix.
     - Stagnation in residuals, where no meaningful progress is made over iterations.

### Next Steps

The following tasks should be the focus of future development:

1. **Further Testing**:
   - Expand testing to include more **complex preconditioners** (e.g., ILU).
   - Increase test coverage for larger and more realistic matrices that resemble the types encountered in RANS simulations.
   - Add tests for performance benchmarking, focusing on convergence rates and iterations required for various matrix sizes and conditions.

2. **Parallelization and Scalability**:
   - Begin integrating **parallel computation** features using **MPI** or Rust's native concurrency tools to handle large-scale distributed systems.
   - Design the infrastructure for distributed meshes and distributed matrix assembly.

3. **Solver Extensions**:
   - Implement other Krylov subspace solvers, such as **GMRES**, to handle non-symmetric systems.
   - Explore multi-grid or other preconditioning strategies to accelerate convergence for complex RANS systems.

4. **Implement `TS`-like Framework for Solving discretized time-dependent PDEs**:
   - Integrate the linear solvers (`KSP`) routines with the **mesh management** system (`DMPlex`-like) to build a PETSc `TS`-like framework for scalable solutions for ODEs and DAEs arising from discretization of time-dependent PDEs, ensuring seamless interaction between mesh entities and linear system assembly, along with time integration.
      - Consult the `petsc_ts_reference.md` for background information on the PETSc `TS` system and how it interacts with the analogous components we have already developed for HYDRA.
      - Refer to the PETSc `TS` implementation for more precise details on the structures, enums, and implementation functions they use.

5. **Geophysical Fluid Dynamics Application**:
   - Begin expanding from simple test systems to fully incorporate the **RANS equations** and associated boundary conditions (e.g., Dirichlet, Neumann) into the solver framework.
   - Develop a modular approach for handling **boundary conditions** and ensuring flexibility for different geophysical domains.
   - For a clearer concept of how we want the ultimate interface to come together at the integration-test level, review the following specific example code developed with PETSc, which nicely encapsulates how we want HYDRA to function in terms of capabilities to define a diverse range of physical processes using `TS`, `DMPlex`, `KSP`, etc.: https://petsc.org/release/src/ts/tutorials/ex11.c.html

By continuing this structured approach, we aim to build a scalable FVM solver that can handle the computational challenges of geophysical fluid dynamics problems.