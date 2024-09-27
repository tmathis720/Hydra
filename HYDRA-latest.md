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

---

Currently, I am focused on troubleshooting the LU preconditioner. I will upload my current working versions of the `src/solver/` folder contents for you to review for appropriate context to produce an accurate and relevant response to advance the project effectively. Here is the source code for src/solver/preconditioner/lu.rs:

   ```rust
   use faer_core::{Mat, MatRef, solve};
use crate::solver::preconditioner::Preconditioner;
use crate::solver::{Matrix, Vector};

pub struct LU {
    lu: Mat<f64>,  // LU factorization matrix
}

impl LU {
    pub fn new(lu: Mat<f64>) -> Self {
        LU { lu }
    }

    /// Solve L * y = r using Faer's unit lower triangular solve
    fn forward_substitution(&self, r: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        let r_as_matrix = r.as_matrix();  // Convert the vector to a 1-column matrix
        let mut y_as_matrix = y.as_matrix_mut();  // Convert the result vector to a mutable 1-column matrix

        solve::solve_unit_lower_triangular_in_place(
            MatRef::from(&self.lu),    // Lower triangular part of the LU matrix
            y_as_matrix.rb_mut(),      // Output vector (y)
            r_as_matrix.rb(),          // Right-hand side vector (r)
            faer_core::Parallelism::None,
        ).expect("Forward substitution failed");
    }

    /// Solve U * z = y using Faer's upper triangular solve
    fn backward_substitution(&self, y: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let y_as_matrix = y.as_matrix();  // Convert the vector to a 1-column matrix
        let mut z_as_matrix = z.as_matrix_mut();  // Convert the result vector to a mutable 1-column matrix

        solve::solve_upper_triangular_in_place(
            MatRef::from(&self.lu),    // Upper triangular part of the LU matrix
            z_as_matrix.rb_mut(),      // Output vector (z)
            y_as_matrix.rb(),          // Right-hand side vector (y)
            faer_core::Parallelism::None,
        ).expect("Backward substitution failed");
    }
}

impl Preconditioner for LU {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        // First step: Forward substitution L * y = r
        let mut y = z.clone();  // Initialize y as a temporary vector to store intermediate result
        self.forward_substitution(r, &mut y);

        // Second step: Backward substitution U * z = y
        self.backward_substitution(&y, z);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{Matrix, Vector};
    use faer_core::{mat, Mat};

    #[test]
    fn test_lu_preconditioner_simple() {
        // A simple 3x3 LU-factored matrix
        let lu = mat![
            [2.0, 3.0, 1.0],  // U
            [0.5, 0.5, 0.5],  // L and U
            [0.5, 1.0, 0.5]   // L and U
        ];

        let r = mat![
            [5.0],  // RHS vector
            [4.5],
            [1.0]
        ];

        // Expected solution z = [1.0, 1.0, -1.0]
        let expected_z = mat![
            [1.0],
            [1.0],
            [-1.0]
        ];

        let mut z = Mat::<f64>::zeros(3, 1);  // Initialize result vector

        // Create LU preconditioner and apply it
        let lu_preconditioner = LU::new(lu);
        lu_preconditioner.apply(&lu_preconditioner.lu, &r, &mut z);

        // Verify the result
        for i in 0..z.nrows() {
            assert!((z.read(i, 0) - expected_z.read(i, 0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lu_preconditioner_identity() {
        let lu = mat![
            [1.0, 0.0, 0.0],  // Identity matrix
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        let r = mat![
            [3.0],  // RHS vector
            [5.0],
            [7.0]
        ];

        let expected_z = r.clone();

        let mut z = Mat::<f64>::zeros(3, 1);  // Initialize result vector

        let lu_preconditioner = LU::new(lu);
        lu_preconditioner.apply(&lu_preconditioner.lu, &r, &mut z);

        for i in 0..z.nrows() {
            assert!((z.read(i, 0) - expected_z.read(i, 0)).abs() < 1e-6);
        }
    }
}


   ```
And in this case, the compiler errors reported:

```bash
error[E0599]: no method named `as_matrix` found for reference `&dyn Vector<Scalar = f64>` in the current scope
  --> src\solver\preconditioner\lu.rs:16:29
   |
16 |         let r_as_matrix = r.as_matrix();  // Convert the vector to a 1-column matrix
   |                             ^^^^^^^^^ method not found in `&dyn Vector<Scalar = f64>`

error[E0599]: no method named `as_matrix_mut` found for mutable reference `&mut dyn Vector<Scalar = f64>` in the current scope
  --> src\solver\preconditioner\lu.rs:17:33
   |
17 |         let mut y_as_matrix = y.as_matrix_mut();  // Convert the result vector to a mutable 1-column matrix
   |                                 ^^^^^^^^^^^^^ method not found in `&mut dyn Vector<Scalar = f64>`

error[E0308]: mismatched types
   --> src\solver\preconditioner\lu.rs:20:26
    |
20  |             MatRef::from(&self.lu),    // Lower triangular part of the LU matrix
    |             ------------ ^^^^^^^^ expected `Matrix<DenseRef<'_, _>>`, found `&Matrix<DenseOwn<f64>>`
    |             |
    |             arguments to this function are incorrect
    |
    = note: expected struct `faer_core::Matrix<DenseRef<'_, _>>`
            found reference `&faer_core::Matrix<DenseOwn<f64>>`
note: associated function defined here
   --> C:\Users\Tea\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\convert\mod.rs:585:8
    |
585 |     fn from(value: T) -> Self;
    |        ^^^^

error[E0061]: this function takes 3 arguments but 4 arguments were supplied
   --> src\solver\preconditioner\lu.rs:19:9
    |
19  |           solve::solve_unit_lower_triangular_in_place(
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
22  |               r_as_matrix.rb(),          // Right-hand side vector (r)
    |  _____________________________-
23  | |             faer_core::Parallelism::None,
    | |             ----------------------------
    | |_____________|__________________________|
    |               |                          help: remove the extra argument
    |               unexpected argument of type `Parallelism`
    |
note: function defined here
   --> C:\Users\Tea\.cargo\registry\src\index.crates.io-6f17d22bba15001f\faer-core-0.17.1\src\solve.rs:496:8
    |
496 | pub fn solve_unit_lower_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

error[E0599]: no method named `expect` found for unit type `()` in the current scope
  --> src\solver\preconditioner\lu.rs:24:11
   |
19 | /         solve::solve_unit_lower_triangular_in_place(
20 | |             MatRef::from(&self.lu),    // Lower triangular part of the LU matrix
21 | |             y_as_matrix.rb_mut(),      // Output vector (y)
22 | |             r_as_matrix.rb(),          // Right-hand side vector (r)
23 | |             faer_core::Parallelism::None,
24 | |         ).expect("Forward substitution failed");
   | |          -^^^^^^ method not found in `()`
   | |__________|
   |

error[E0599]: no method named `as_matrix` found for reference `&dyn Vector<Scalar = f64>` in the current scope
  --> src\solver\preconditioner\lu.rs:29:29
   |
29 |         let y_as_matrix = y.as_matrix();  // Convert the vector to a 1-column matrix
   |                             ^^^^^^^^^ method not found in `&dyn Vector<Scalar = f64>`

error[E0599]: no method named `as_matrix_mut` found for mutable reference `&mut dyn Vector<Scalar = f64>` in the current scope
  --> src\solver\preconditioner\lu.rs:30:33
   |
30 |         let mut z_as_matrix = z.as_matrix_mut();  // Convert the result vector to a mutable 1-column matrix
   |                                 ^^^^^^^^^^^^^ method not found in `&mut dyn Vector<Scalar = f64>`

error[E0308]: mismatched types
   --> src\solver\preconditioner\lu.rs:33:26
    |
33  |             MatRef::from(&self.lu),    // Upper triangular part of the LU matrix
    |             ------------ ^^^^^^^^ expected `Matrix<DenseRef<'_, _>>`, found `&Matrix<DenseOwn<f64>>`
    |             |
    |             arguments to this function are incorrect
    |
    = note: expected struct `faer_core::Matrix<DenseRef<'_, _>>`
            found reference `&faer_core::Matrix<DenseOwn<f64>>`
note: associated function defined here
   --> C:\Users\Tea\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\convert\mod.rs:585:8
    |
585 |     fn from(value: T) -> Self;
    |        ^^^^

error[E0061]: this function takes 3 arguments but 4 arguments were supplied
   --> src\solver\preconditioner\lu.rs:32:9
    |
32  |           solve::solve_upper_triangular_in_place(
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
35  |               y_as_matrix.rb(),          // Right-hand side vector (y)
    |  _____________________________-
36  | |             faer_core::Parallelism::None,
    | |             ----------------------------
    | |_____________|__________________________|
    |               |                          help: remove the extra argument
    |               unexpected argument of type `Parallelism`
    |
note: function defined here
   --> C:\Users\Tea\.cargo\registry\src\index.crates.io-6f17d22bba15001f\faer-core-0.17.1\src\solve.rs:406:8
    |
406 | pub fn solve_upper_triangular_in_place<E: ComplexField, TriE: Conjugate<Canonical = E>>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

error[E0599]: no method named `expect` found for unit type `()` in the current scope
  --> src\solver\preconditioner\lu.rs:37:11
   |
32 | /         solve::solve_upper_triangular_in_place(
33 | |             MatRef::from(&self.lu),    // Upper triangular part of the LU matrix
34 | |             z_as_matrix.rb_mut(),      // Output vector (z)
35 | |             y_as_matrix.rb(),          // Right-hand side vector (y)
36 | |             faer_core::Parallelism::None,
37 | |         ).expect("Backward substitution failed");
   | |          -^^^^^^ method not found in `()`
   | |__________|
   |

error[E0599]: no method named `clone` found for mutable reference `&mut dyn Vector<Scalar = f64>` in the current scope  
  --> src\solver\preconditioner\lu.rs:44:23
   |
44 |         let mut y = z.clone();  // Initialize y as a temporary vector to store intermediate result
   |                       ^^^^^ method not found in `&mut dyn Vector<Scalar = f64>`
   |
   = help: items from traits can only be used if the trait is implemented and in scope
   = note: the trait `Clone` defines an item `clone`, but is explicitly unimplemented
```