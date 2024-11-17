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
   - [Overview of Preconditioners](#overview-of-preconditioners)
   - [Jacobi Preconditioner](#jacobi-preconditioner)
   - [LU Preconditioner](#lu-preconditioner)
   - [ILU Preconditioner](#ilu-preconditioner)
   - [Cholesky Preconditioner](#cholesky-preconditioner)
5. [Using the Solver Module](#5-using-the-solver-module)
   - [Setting Up a Solver](#setting-up-a-solver)
   - [Applying Preconditioners](#applying-preconditioners)
   - [Solving Linear Systems](#solving-linear-systems)
6. [Examples and Usage](#6-examples-and-usage)
   - [Example with Conjugate Gradient](#example-with-conjugate-gradient)
   - [Example with GMRES](#example-with-gmres)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Solver` module of the Hydra computational framework. This module provides a suite of Krylov subspace solvers and preconditioners designed to solve large, sparse linear systems efficiently. The solvers are essential for numerical simulations in computational fluid dynamics (CFD) and other fields requiring iterative solutions to linear systems.

---

## **2. Overview of the Solver Module**

The `Solver` module in Hydra is organized into several components:

- **Krylov Subspace Solvers (KSP)**: Abstract interface for solvers like Conjugate Gradient (CG) and Generalized Minimal Residual Solver (GMRES).
- **Preconditioners**: Modules that improve convergence rates by transforming the system into a more favorable form.
- **Solver Manager**: A high-level interface that integrates solvers and preconditioners for flexible usage.

The module's design emphasizes:

- **Flexibility**: Ability to interchange solvers and preconditioners easily.
- **Performance**: Utilization of parallel computing via Rayon for efficiency.
- **Extensibility**: Support for adding new solvers and preconditioners.

---

## **3. Krylov Subspace Solvers**

### KSP Trait

The `KSP` trait defines a common interface for all Krylov subspace solvers in the Hydra framework. It ensures consistency and allows for easy interchangeability between different solver implementations.

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

- **Parameters**:
  - `a`: The system matrix `A`.
  - `b`: The right-hand side vector `b`.
  - `x`: The solution vector `x`, which will be updated.
- **Returns**: A `SolverResult` containing convergence information.

### Conjugate Gradient Solver

The `ConjugateGradient` struct implements the `KSP` trait, providing an efficient solver for symmetric positive-definite (SPD) systems.

#### Key Features

- **Preconditioning Support**: Optional preconditioner can be applied.
- **Parallel Computation**: Utilizes Rayon for parallel operations.

#### Usage

```rust
let mut cg = ConjugateGradient::new(max_iter, tolerance);
cg.set_preconditioner(Box::new(Jacobi::default())); // Optional
let result = cg.solve(&a, &b, &mut x);
```

- **Methods**:
  - `new(max_iter, tol)`: Creates a new instance with specified maximum iterations and tolerance.
  - `set_preconditioner(preconditioner)`: Sets an optional preconditioner.

### GMRES Solver

The `GMRES` (Generalized Minimal Residual Solver) is suitable for non-symmetric or non-positive-definite systems.

#### Key Features

- **Restart Mechanism**: Supports restarts to prevent loss of orthogonality.
- **Preconditioning Support**: Can incorporate preconditioners.

#### Usage

```rust
let mut gmres = GMRES::new(max_iter, tolerance, restart);
gmres.set_preconditioner(Arc::new(Jacobi::default())); // Optional
let result = gmres.solve(&a, &b, &mut x);
```

- **Methods**:
  - `new(max_iter, tol, restart)`: Initializes the solver with specified parameters.
  - `set_preconditioner(preconditioner)`: Sets an optional preconditioner.

---

## **4. Preconditioners**

### Overview of Preconditioners

Preconditioners transform a linear system into an equivalent one that has more favorable properties for iterative solution methods. They aim to improve convergence rates and overall solver performance.

### Jacobi Preconditioner

The Jacobi preconditioner is one of the simplest preconditioners, utilizing the inverse of the diagonal elements of the matrix.

#### Usage

```rust
let jacobi_preconditioner = Jacobi::default();
cg.set_preconditioner(Box::new(jacobi_preconditioner));
```

#### Implementation Highlights

- **Parallelism**: Uses `rayon` for parallel computation across rows.
- **Thread Safety**: Employs `Arc<Mutex<T>>` to ensure safe concurrent access.

### LU Preconditioner

The LU preconditioner uses LU decomposition to factorize the matrix and solve the preconditioned system efficiently.

#### Usage

```rust
let lu_preconditioner = LU::new(&a);
gmres.set_preconditioner(Arc::new(lu_preconditioner));
```

#### Implementation Highlights

- **Partial Pivoting**: Utilizes partial pivot LU decomposition from the `faer` library.
- **Efficient Solving**: Provides methods for forward and backward substitution.

### ILU Preconditioner

The Incomplete LU (ILU) preconditioner approximates the LU decomposition while preserving the sparsity pattern.

#### Usage

```rust
let ilu_preconditioner = ILU::new(&a);
gmres.set_preconditioner(Arc::new(ilu_preconditioner));
```

#### Implementation Highlights

- **Sparsity Preservation**: Discards small values to maintain sparsity.
- **Custom Decomposition**: Implements a sparse ILU decomposition algorithm.

### Cholesky Preconditioner

The Cholesky preconditioner is suitable for SPD matrices and uses Cholesky decomposition for efficient solving.

#### Usage

```rust
let cholesky_preconditioner = CholeskyPreconditioner::new(&a)?;
cg.set_preconditioner(Box::new(cholesky_preconditioner));
```

#### Implementation Highlights

- **Error Handling**: Returns a `Result` to handle decomposition failures.
- **Lower Triangular Factorization**: Decomposes the matrix into lower and upper triangular matrices.

---

## **5. Using the Solver Module**

### Setting Up a Solver

To set up a solver, you need to:

1. **Choose a Solver**: Decide between `ConjugateGradient` or `GMRES` based on your system's properties.
2. **Initialize the Solver**: Create an instance with appropriate parameters.

**Example**:

```rust
let max_iter = 1000;
let tolerance = 1e-6;

let mut solver = ConjugateGradient::new(max_iter, tolerance);
```

### Applying Preconditioners

Preconditioners can significantly improve solver performance.

**Adding a Preconditioner**:

```rust
let preconditioner = Box::new(Jacobi::default());
solver.set_preconditioner(preconditioner);
```

### Solving Linear Systems

To solve the system `Ax = b`:

1. **Prepare the System Matrix and Vectors**: Ensure `A`, `b`, and `x` are properly defined.
2. **Call the Solver**:

```rust
let result = solver.solve(&a, &b, &mut x);
```

3. **Check the Result**:

```rust
if result.converged {
    println!("Solver converged in {} iterations.", result.iterations);
} else {
    println!("Solver did not converge.");
}
```

---

## **6. Examples and Usage**

### Example with Conjugate Gradient

**Problem**: Solve `Ax = b` where `A` is SPD.

**Setup**:

```rust
use faer::mat;

let a = mat![
    [4.0, 1.0],
    [1.0, 3.0],
];

let b = mat![
    [1.0],
    [2.0],
];

let mut x = Mat::<f64>::zeros(2, 1);
```

**Solver Initialization**:

```rust
let mut cg = ConjugateGradient::new(100, 1e-6);
```

**Applying Preconditioner** (Optional):

```rust
let jacobi_preconditioner = Box::new(Jacobi::default());
cg.set_preconditioner(jacobi_preconditioner);
```

**Solving**:

```rust
let result = cg.solve(&a, &b, &mut x);
```

**Result Checking**:

```rust
if result.converged {
    println!("Solution: {:?}", x);
} else {
    println!("Solver did not converge.");
}
```

### Example with GMRES

**Problem**: Solve a non-symmetric system `Ax = b`.

**Setup**:

```rust
let a = mat![
    [2.0, 1.0],
    [3.0, 4.0],
];

let b = mat![
    [1.0],
    [2.0],
];

let mut x = Mat::<f64>::zeros(2, 1);
```

**Solver Initialization**:

```rust
let mut gmres = GMRES::new(100, 1e-6, 2);
```

**Applying Preconditioner** (Optional):

```rust
let lu_preconditioner = Arc::new(LU::new(&a));
gmres.set_preconditioner(lu_preconditioner);
```

**Solving**:

```rust
let result = gmres.solve(&a, &b, &mut x);
```

**Result Checking**:

```rust
if result.converged {
    println!("Solution: {:?}", x);
} else {
    println!("Solver did not converge.");
}
```

---

## **7. Best Practices**

- **Select Appropriate Solver**: Use CG for SPD systems and GMRES for non-symmetric systems.
- **Utilize Preconditioners**: Always consider applying a preconditioner to improve convergence.
- **Monitor Convergence**: Check the `SolverResult` for convergence status and residual norms.
- **Thread Safety**: Ensure that matrices and vectors are thread-safe if using custom implementations.
- **Handle Errors**: Be prepared to handle cases where solvers do not converge within the maximum iterations.

---

## **8. Conclusion**

The `Solver` module in Hydra provides robust and flexible tools for solving large, sparse linear systems. By offering multiple solver options and preconditioners, it caters to a wide range of problems encountered in computational simulations. Proper utilization of this module can lead to significant performance improvements and more accurate results in your simulations.

---

**Note**: For advanced usage and custom implementations, refer to the official Hydra documentation and source code.