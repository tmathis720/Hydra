# Hydra `Linear Algebra` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Linear Algebra Module](#2-overview-of-the-linear-algebra-module)  
3. [Core Components](#3-core-components)  
   - [Vectors](#vectors)  
   - [Matrices](#matrices)  
4. [Vector Module](#4-vector-module)  
   - [Vector Traits](#vector-traits)  
   - [Vector Implementations](#vector-implementations)  
     - [Implementation for `Vec<f64>`](#implementation-for-vecf64)  
     - [Implementation for `Mat<f64>`](#implementation-for-matf64)  
   - [Vector Builder](#vector-builder)  
   - [Vector Testing](#vector-testing)  
5. [Matrix Module](#5-matrix-module)  
   - [Matrix Traits](#matrix-traits)  
   - [Matrix Implementations](#matrix-implementations)  
     - [Implementation for `Mat<f64>`](#implementation-for-matf64-1)  
     - [SparseMatrix](#sparsematrix)  
   - [Matrix Builder](#matrix-builder)  
   - [Matrix Testing](#matrix-testing)  
6. [Using the Linear Algebra Module](#6-using-the-linear-algebra-module)  
   - [Creating Vectors](#creating-vectors)  
   - [Performing Vector Operations](#performing-vector-operations)  
   - [Creating Matrices](#creating-matrices)  
   - [Performing Matrix Operations](#performing-matrix-operations)  
7. [Best Practices](#7-best-practices)  
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the **`Linear Algebra`** module of the Hydra computational framework. This module provides the fundamental linear algebra functionalities used in **numerical simulations**, **finite volume/element methods**, and other scientific computing tasks. Key features include:

- **Vector operations**: Dot product, norms, scalings, element-wise operations, etc.  
- **Matrix operations**: Matrix-vector multiplication, trace, Frobenius norm, resizing, etc.  
- **Abstract Traits** and **Concrete Implementations**: Flexible trait system for different data structures (`Vec<f64>`, `faer::Mat<f64>`, and a `SparseMatrix`).  

---

## **2. Overview of the Linear Algebra Module**

The **`linalg`** module separates functionality into **vector** and **matrix** submodules:

- **Vectors** (`linalg::vector`)  
  - `Vector` trait plus implementations for standard Rust vectors (`Vec<f64>`) and `faer::Mat<f64>` (treated as a column vector).
  - A **`VectorBuilder`** utility to create and resize vectors.
- **Matrices** (`linalg::matrix`)  
  - `Matrix` trait plus specialized traits (`MatrixOperations`, `ExtendedMatrixOperations`) for constructing/resizing.
  - Implementations for `faer::Mat<f64>` (dense) and a custom `SparseMatrix`.
  - A **`MatrixBuilder`** utility to create, resize, and integrate with preconditioners.

This design allows Hydra to **expand** or **swap** underlying data structures (e.g., other backends for HPC).  

---

## **3. Core Components**

### Vectors

- Defined primarily by the **`Vector`** trait, located in `src/linalg/vector/traits.rs`.
- The trait enforces thread safety (`Send + Sync`) and includes standard operations:
  - Dot product, `norm`, `axpy`, `scale`, cross product (for 3D), sums, min/max, mean, variance, etc.
- **Implementations** exist for:
  - **`Vec<f64>`** (a standard Rust vector)
  - **`faer::Mat<f64>`** interpreted as a **column vector** (with `nrows() == length, ncols() == 1`).

### Matrices

- Defined primarily by the **`Matrix`** trait, located in `src/linalg/matrix/traits.rs`.
- The trait includes methods:
  - `nrows()`, `ncols()`
  - `mat_vec` (matrix-vector multiplication)
  - `trace()`, `frobenius_norm()`
  - `get(i, j)`, plus read/write slices (though not all implementations support slice mutability).
- **Implementations** exist for:
  - **`faer::Mat<f64>`** (a standard dense 2D array).
  - A custom **`SparseMatrix`** that uses a `FxHashMap` for storing non-zero entries.

---

## **4. Vector Module**

The **vector module** is organized as follows:

- **`traits.rs`**: Defines the `Vector` trait.  
- **`vec_impl.rs`**: Implements `Vector` for `Vec<f64>`.  
- **`mat_impl.rs`**: Implements `Vector` for a `faer::Mat<f64>` column.  
- **`vector_builder.rs`**: Contains the `VectorBuilder` utility and supporting traits to build or resize vectors.

### Vector Traits

```rust
pub trait Vector: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Self::Scalar;
    fn set(&mut self, i: usize, value: Self::Scalar);
    fn as_slice(&self) -> &[Self::Scalar];
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;
    fn norm(&self) -> Self::Scalar;
    fn scale(&mut self, scalar: Self::Scalar);
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);
    fn cross(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str>;
    fn sum(&self) -> Self::Scalar;
    fn max(&self) -> Self::Scalar;
    fn min(&self) -> Self::Scalar;
    fn mean(&self) -> Self::Scalar;
    fn variance(&self) -> Self::Scalar;
}
```

- The **thread-safety** requirement: `Send + Sync`.  
- The **cross product** method is only valid for 3D vectors.  

### Vector Implementations

#### Implementation for `Vec<f64>`

In `vec_impl.rs`, a standard Rust vector is extended via the `Vector` trait:

- **Key methods**:
  - `dot`, `norm`, `axpy`, `scale`, `element_wise_*`, `cross` (3D), `sum`, `max`, `min`, `mean`, `variance`.

**Example**:
```rust
let mut vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];

let dot = vec1.dot(&vec2); // 32.0
vec1.scale(2.0); // vec1 becomes [2.0, 4.0, 6.0]
vec1.axpy(1.5, &vec2); // vec1 = 1.5*vec2 + vec1
```

#### Implementation for `Mat<f64>`

In `mat_impl.rs`, a **`faer::Mat<f64>`** with `ncols() == 1` is treated as a column vector:

- `len()` -> number of rows
- `get(i)`, `set(i)`, `dot(...)`, etc.
- `as_slice()` / `as_mut_slice()` use `try_as_slice()` from `faer`.

**Example**:
```rust
use faer::Mat;
use hydra::linalg::Vector;

let mut mat_vec = Mat::<f64>::zeros(3, 1); // 3x1
mat_vec.set(0, 1.0); 
mat_vec.set(1, 2.0);
mat_vec.set(2, 3.0);

let norm = mat_vec.norm(); // sqrt(1^2 + 2^2 + 3^2) = ~3.74
```

### Vector Builder

**`vector_builder.rs`** provides `VectorBuilder` to build vectors in a generic way.

- `build_vector<T: VectorOperations>(size: usize) -> T`
- `build_dense_vector(size: usize) -> Mat<f64>` 
- `resize_vector<T: VectorOperations + ExtendedVectorOperations>(vector, new_size)`

**Vector Operations** trait:
- `construct(size) -> Self`
- `set_value(index, value)`
- `get_value(index) -> f64`
- `size() -> usize`

Then `ExtendedVectorOperations` adds `resize(new_size)`.  
Implementations are provided for both `Vec<f64>` and `Mat<f64>`.

### Vector Testing

Comprehensive tests in `src/linalg/vector/tests.rs` validate:

- Indexing, dot products, norm, cross product (3D), element-wise ops, etc.
- Edge cases: empty vectors, large vectors, dimension mismatch for cross product, etc.

---

## **5. Matrix Module**

The **matrix module** is organized as follows:

- **`traits.rs`**: Defines the `Matrix`, `MatrixOperations`, and `ExtendedMatrixOperations` traits.  
- **`mat_impl.rs`**: Implements `Matrix` for `faer::Mat<f64>`.  
- **`matrix_builder.rs`**: Contains `MatrixBuilder` utility for constructing/resizing matrices and applying preconditioners.  
- **`sparse_matrix.rs`**: A simple `SparseMatrix` that also implements `Matrix`.  
- **`tests.rs`**: Test suite for matrix functionality.

### Matrix Traits

```rust
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>);
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    fn trace(&self) -> Self::Scalar;
    fn frobenius_norm(&self) -> Self::Scalar;
    fn as_slice(&self) -> Box<[Self::Scalar]>;
    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}
```

- `MatrixOperations` trait:
  - `construct(rows, cols) -> Self`
  - `get(...)`, `set(...)`
  - `size() -> (usize, usize)`
- `ExtendedMatrixOperations` trait adds `fn resize(&mut self, new_rows, new_cols)`.

### Matrix Implementations

#### Implementation for `Mat<f64>`

In `mat_impl.rs`, `faer::Mat<f64>` is extended:

- **`Matrix`** trait:
  - `nrows()`, `ncols()`
  - `mat_vec(x, y)`: Standard dense matrix-vector multiplication
  - `trace()`: sum of diagonal
  - `frobenius_norm()`: sqrt of sum of squares
  - `as_slice()`, `as_slice_mut()`: yields a `Box<[f64]>` copy or slice
- **`MatrixOperations`**:
  - `construct(rows, cols)` -> zero matrix
  - `set(row, col, value)`, `get(row, col)`  
- **`ExtendedMatrixOperations`**:
  - `resize(&mut self, new_rows, new_cols)` -> creates new `Mat<f64>` and copies existing entries.

**Example**:
```rust
use faer::Mat;
use hydra::linalg::{Matrix, MatrixOperations};

let mut mat = Mat::<f64>::zeros(3, 3);
mat.set(1, 2, 5.0); // matrix.write(1,2,5.0)
let trace = mat.trace();
let norm = mat.frobenius_norm();
```

#### SparseMatrix

In `sparse_matrix.rs`, a **`SparseMatrix`** using `FxHashMap<(row, col), f64>` is provided:

- Also implements the **`Matrix`** trait:
  - `mat_vec(...)`: only iterates over non-zero entries for multiplication
  - `trace()`, `frobenius_norm()`, `get(i, j)` -> zero if absent
  - `as_slice()` is **not supported** (panics), as it’s not contiguous.
- Implements **`MatrixOperations`** and **`ExtendedMatrixOperations`**:
  - `set(row, col, value)`: storing or removing near-zero entries
  - `resize(...)`: re-hash only valid entries that fit in the new dimension range.

This allows a simple **sparse** backend with minimal overhead.

### Matrix Builder

**`matrix_builder.rs`** has a `MatrixBuilder` struct:

- `build_matrix<T: MatrixOperations>(rows, cols) -> T`
- `build_dense_matrix(rows, cols) -> Mat<f64>`
- `resize_matrix<T: MatrixOperations + ExtendedMatrixOperations>(...)`
- `apply_preconditioner(preconditioner, matrix)`: Example usage with a solver preconditioner.

---

### Matrix Testing

The `src/linalg/matrix/tests.rs` typically checks:

- **mat_vec** for correctness  
- Setting/retrieving matrix elements  
- Sizing, resizing, and partial copy logic  
- Corner cases: zero rows/columns, out-of-bounds, etc.

---

## **6. Using the Linear Algebra Module**

Below are typical usage patterns using the vector and matrix abstractions:

### Creating Vectors

1. **`Vec<f64>`** (most common):
   ```rust
   let mut vector = vec![0.0; 5];
   vector.set(0, 1.0);
   ```
2. **Using `faer::Mat<f64>` as a column vector**:
   ```rust
   use faer::Mat;

   let mut mat_vec = Mat::<f64>::zeros(5, 1);
   mat_vec.set(0, 1.0);
   ```

### Performing Vector Operations

**Dot Product**:
```rust
let vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];
let dot_val = vec1.dot(&vec2); // 32.0
```

**Norm**:
```rust
let norm = vec1.norm(); // sqrt(1^2 + 2^2 + 3^2) = ~3.74
```

**Scale and AXPY**:
```rust
vec1.scale(2.0); // [2.0, 4.0, 6.0]
vec1.axpy(1.5, &vec2); // vec1 = 1.5*vec2 + vec1
```

**Element-wise**:
```rust
vec1.element_wise_add(&vec2);
vec1.element_wise_mul(&vec2);
```

### Creating Matrices

1. **Using `faer::Mat<f64>`**:
   ```rust
   let mut matrix = Mat::<f64>::zeros(3, 3);
   matrix.set(1, 1, 5.0);
   ```
2. **Using `SparseMatrix`**:
   ```rust
   use hydra::linalg::matrix::sparse_matrix::SparseMatrix;

   let mut sp_mat = SparseMatrix::new(3, 3);
   sp_mat.set(0, 0, 1.0);
   sp_mat.set(2, 1, 3.5);
   ```

### Performing Matrix Operations

**Matrix-Vector Multiplication**:
```rust
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y); // y = matrix * x
```

**Trace and Frobenius Norm**:
```rust
let trace_val = matrix.trace();
let fro_norm = matrix.frobenius_norm();
```

**Resizing**:
```rust
use hydra::linalg::matrix::MatrixBuilder;
MatrixBuilder::resize_matrix(&mut matrix, 5, 5);
```

---

## **7. Best Practices**

1. **Dimensional Consistency**: Always ensure vectors and matrices match in size when performing multiplication or element-wise operations.  
2. **Thread Safety**: The traits require `Send + Sync`; your data structures must maintain concurrency safety.  
3. **Sparse vs. Dense**: Choose `SparseMatrix` if your matrix has many zero entries. For dense computations, use `faer::Mat<f64>`.  
4. **Cross Product**: Use only for 3D vectors; otherwise, it returns an error.  
5. **Preconditioning**: The `MatrixBuilder::apply_preconditioner` demonstrates how to integrate a solver preconditioner with your matrix.  
6. **Performance**: For large vectors or matrices, ensure you do minimal copying (e.g., pass slices or references).

---

## **8. Conclusion**

The **`Linear Algebra`** module in Hydra offers a unified abstraction layer for **vector** and **matrix** operations, supporting multiple data structures from a single interface:

- **`Vector` trait** with implementations for standard Rust vectors and `faer::Mat<f64>` column vectors.  
- **`Matrix` trait** with implementations for **dense** (`faer::Mat<f64>`) and **sparse** (`SparseMatrix`) usage.  
- **Builders** (`VectorBuilder`, `MatrixBuilder`) for generic creation, resizing, and specialized operations (e.g., applying preconditioners).  

By leveraging these traits and implementations, you can write **cleaner**, **extensible** linear algebra code while mixing and matching data structures best suited to your simulation’s memory and performance requirements.