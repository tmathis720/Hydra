# Hydra `Solver` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Solver Module](#2-overview-of-the-solver-module)  
3. [Krylov Subspace Solvers](#3-krylov-subspace-solvers)  
   - [KSP Trait](#ksp-trait)  
   - [Conjugate Gradient Solver](#conjugate-gradient-solver)  
   - [GMRES Solver](#gmres-solver)  
4. [Preconditioners](#4-preconditioners)  
   - [Jacobi Preconditioner](#jacobi-preconditioner)  
   - [LU Preconditioner](#lu-preconditioner)  
   - [ILU Preconditioner](#ilu-preconditioner)  
   - [Cholesky Preconditioner](#cholesky-preconditioner)  
   - [AMG Preconditioner](#amg-preconditioner)  
5. [Using the Solver Module](#5-using-the-solver-module)  
   - [Choosing and Creating a Solver](#choosing-and-creating-a-solver)  
   - [Applying Preconditioners](#applying-preconditioners)  
   - [Solving Linear Systems](#solving-linear-systems)  
6. [Examples and Usage](#6-examples-and-usage)  
   - [Conjugate Gradient Example](#conjugate-gradient-example)  
   - [GMRES Example](#gmres-example)  
7. [Best Practices](#7-best-practices)  
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the **`Solver`** module user guide for Hydra, which provides **iterative solvers** and **preconditioners** to tackle large, sparse linear systems. Whether your matrix is **SPD (Symmetric Positive Definite)** or **general non-symmetric**, the solver framework offers:

- **Krylov Subspace Methods**:
  - Conjugate Gradient (CG)
  - GMRES (Generalized Minimal Residual)
- **Preconditioners** to improve convergence:
  - Jacobi, LU, ILU, Cholesky, AMG, etc.
- A **unified `KSP` trait** and **`SolverManager`** for flexible solver/preconditioner configuration.

This module is intended for advanced numerical simulations in **CFD**, **finite element/volume methods**, or any domain that requires **iterative** solution of \(A x = b\).

---

## **2. Overview of the Solver Module**

The solver code is split into key components:

1. **KSP** (Krylov Subspace Solvers)  
   - **`KSP` trait**: Common interface for solvers.  
   - **`ConjugateGradient`**: For SPD systems.  
   - **`GMRES`**: For general non-symmetric systems.  

2. **Preconditioners**  
   - **`Preconditioner`** trait plus multiple implementations:
     - **Jacobi**: Simple diagonal-based approach.  
     - **LU**, **ILU**, **Cholesky** (using Faer library).  
     - **AMG** (Algebraic Multigrid).
   - Significantly **reduces iteration count** by improving matrix conditioning.

3. **SolverManager**  
   - High-level adapter that unifies a chosen solver with an optional preconditioner.  
   - Offers a single `solve(a, b, x)` method to run the entire solve.

**Parallelization** is handled largely via **Rayon** for operations like dot products, mat-vec multiplications, and more.

---

## **3. Krylov Subspace Solvers**

### KSP Trait

```rust
pub trait KSP {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}
```

- **Purpose**: A consistent interface for iterative solvers like CG or GMRES.
- **`solve(...)`**: Takes a matrix `A`, right-hand side `b`, and solution vector `x`. Returns a `SolverResult` that indicates if convergence was achieved.

### Conjugate Gradient Solver

**`ConjugateGradient`** solves **SPD** systems:  
- Uses a **preconditioner** if set (e.g., Jacobi, ILU, etc.).  
- **Rayon** is used for parallel dot products and residual updates.  
- **Key Methods**:
  - `new(max_iter, tol)`: Creates a solver with iteration limit and tolerance.  
  - `set_preconditioner(...)`: Attach a `Preconditioner`.  

**Workflow**:  
1. Compute initial residual \(r = b - A x\).  
2. Precondition the residual (optional).  
3. Update direction `p`.  
4. Iterate until norm of residual < `tol` or `max_iter` is reached.

### GMRES Solver

**`GMRES`** is for general non-symmetric systems:
- **Restart-based** (the `restart` parameter) to limit Krylov subspace size.  
- **Arnoldi process** to build an orthonormal basis.  
- **Givens rotations** to maintain upper Hessenberg form and compute the solution in a smaller subspace.  
- Also uses an **optional preconditioner**.

**Key Methods**:
- `new(max_iter, tol, restart)`: Basic constructor.  
- `set_preconditioner(...)`: Hook a preconditioner.

---

## **4. Preconditioners**

Each preconditioner implements:

```rust
pub trait Preconditioner: Send + Sync {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}
```

**Goal**: transform the residual `r` into a new vector `z` that is easier for CG/GMRES to work with.

### Jacobi Preconditioner

- **Uses only the diagonal** of matrix `A`.  
- `z[i] = r[i] / A[i, i]` (if diagonal is non-zero).  
- Parallelized with **Rayon**:
  ```rust
  let z = Arc::new(Mutex::new(z));
  (0..a.nrows()).into_par_iter().for_each(|i| { ... });
  ```
- Extremely simple but beneficial for diagonally dominant or well-scaled problems.

### LU Preconditioner

- **LU factorization** of matrix `A` (partial pivoting).  
- Solves `LU * x = r` quickly via forward/back substitution.  
- Implementation uses **`faer::PartialPivLu`**.  
- Good for smaller systems or as a “direct” method in a broader iterative approach.  
- Potentially large overhead for big systems.

### ILU Preconditioner

- **Incomplete LU** factorization preserves the **sparsity** pattern:
  - Ignores small values in factorization to reduce memory overhead.  
  - Approximates the effect of a full LU.  
- Works well for **large, sparse** systems.  
- Implementation:
  - Basic incomplete factor iteration (with possible threshold).  
  - `apply_ilu(...)` for forward/back substitution.

### Cholesky Preconditioner

- **Cholesky** for symmetric positive definite matrices only.  
- Factorizes \(A = LL^T\).  
- Uses **Faer** for decomposition + forward/back solves.  
- Typically used with CG.  
- The code might return an error if the matrix isn’t SPD or factorization fails.

### AMG Preconditioner

- **Algebraic Multigrid (AMG)**:
  - Builds a **hierarchy of coarser grids** from the original matrix.  
  - Uses strength-of-connection, coarsening, interpolation, etc.  
- Implementation includes:
  - Methods to **construct** coarser levels (`generate_operators`, `compute_strength_matrix`, etc.).  
  - A **recursive** approach that calls `apply_recursive(...)` for V-cycle or W-cycle style smoothing.  
- Very powerful for large problems with structured or unstructured grids.

---

## **5. Using the Solver Module**

### Choosing and Creating a Solver

Pick a solver best suited for the system:
- **SPD**: `ConjugateGradient`.
- **General**: `GMRES`.

**Helper**: `create_solver(solver_type, max_iter, tol, restart)` returns a `Box<dyn KSP>`:

```rust
use hydra::solver::ksp::{create_solver, SolverType};

let solver = create_solver(SolverType::GMRES, 1000, 1e-6, 50);
```

Alternatively, you can instantiate directly:
```rust
let mut cg = ConjugateGradient::new(1000, 1e-8);
let mut gmres = GMRES::new(1000, 1e-6, 30);
```

### Applying Preconditioners

Attach a preconditioner (e.g., Jacobi, LU) if desired:
```rust
use hydra::solver::preconditioner::Jacobi;

cg.set_preconditioner(Box::new(Jacobi::default()));
```

If using `SolverManager`:
```rust
use hydra::solver::ksp::{SolverManager};
use std::sync::Arc;
use hydra::solver::preconditioner::{PreconditionerFactory};

let mut manager = SolverManager::new(Box::new(cg));
manager.set_preconditioner(PreconditionerFactory::create_jacobi());
```

### Solving Linear Systems

1. Prepare `A`, `b`, and an **initial guess** `x`.  
2. Call `solve(a, b, x)`.  
3. Check the returned `SolverResult`:
   - `converged`  
   - `iterations`  
   - `residual_norm`

**Example**:
```rust
let result = manager.solve(&a, &b, &mut x);
if result.converged {
    println!("Converged in {} iterations, res norm = {}", result.iterations, result.residual_norm);
} else {
    eprintln!("Not converged after {} iterations", result.iterations);
}
```

---

## **6. Examples and Usage**

### Conjugate Gradient Example

**Context**: Suppose `A` is SPD, size = 2x2 for demonstration.

```rust
use hydra::solver::{ConjugateGradient, KSP};
use hydra::linalg::{Matrix, Vector};
use faer::Mat;

// Create A, b, x
let a = Mat::from_fn(2, 2, |i, j| {
    if i == j {
        4.0
    } else {
        1.0
    }
});
let b = vec![1.0, 2.0];
let mut x = vec![0.0, 0.0];

// Initialize CG
let mut cg = ConjugateGradient::new(100, 1e-6);

// (Optional) attach a Jacobi preconditioner
use hydra::solver::preconditioner::Jacobi;
cg.set_preconditioner(Box::new(Jacobi::default()));

// Solve
let result = cg.solve(&a, &b, &mut x);
if result.converged {
    println!("CG converged in {} iters, solution: {:?}", result.iterations, x);
}
```

### GMRES Example

**Context**: Non-symmetric or indefinite system.

```rust
use hydra::solver::{GMRES, KSP};
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| {
    match (i,j) {
        (0,0) => 2.0, (0,1) => 1.0,
        (1,0) => 3.0, (1,1) => 4.0,
        _ => 0.0,
    }
});
let b = vec![1.0, 2.0];
let mut x = vec![0.0, 0.0];

let mut gmres = GMRES::new(100, 1e-6, 10);
// Optional: LU preconditioner
use hydra::solver::preconditioner::LU;
use std::sync::Arc;
gmres.set_preconditioner(Arc::new(LU::new(&a)));

let result = gmres.solve(&a, &b, &mut x);
if result.converged {
    println!("GMRES converged. x = {:?}", x);
} else {
    println!("GMRES did not converge. residual={}", result.residual_norm);
}
```

---

## **7. Best Practices**

1. **Match Solver to Matrix**:
   - CG for SPD systems,
   - GMRES for general systems.
2. **Leverage Preconditioning**: Even a simple Jacobi can speed up convergence. More advanced (ILU, AMG) for large sparse systems.
3. **Monitor Tolerance**: Adjust `tol` to balance **accuracy vs. iteration count**.
4. **Check Residual Norm**: If the solver does not converge or stalls, consider different preconditioners or re-check matrix conditioning.
5. **Thread Safety**: The code uses **Rayon** heavily—any custom matrix or vector must be `Send + Sync`.
6. **AMG Complexity**: Algebraic multigrid can require more advanced tuning (coarsening threshold, etc.).

---

## **8. Conclusion**

The **`Solver`** module is an **integral** part of Hydra for handling large linear systems in HPC or scientific simulation contexts. By mixing:

- **Krylov solvers** (CG, GMRES),
- **Preconditioners** (Jacobi, LU, ILU, Cholesky, AMG),
- A **unified interface** (`KSP` or `SolverManager`),

…users can flexibly choose the **best** approach for their problem. Properly pairing a solver with an appropriate preconditioner often **drastically** reduces iteration count and CPU time, ensuring robust and efficient solutions for complex CFD, FE, or other large-scale simulations.