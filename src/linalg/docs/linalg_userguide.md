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

Welcome to the user's guide for the `Linear Algebra` module of the Hydra computational framework. This module provides essential linear algebra functionalities, including vector and matrix operations, which are fundamental for numerical simulations and computational methods used in finite volume methods (FVM) and computational fluid dynamics (CFD).

---

## **2. Overview of the Linear Algebra Module**

The `Linear Algebra` module in Hydra is designed to offer:

- **Abstract Traits**: Define common interfaces for vectors and matrices, allowing for flexible implementations.
- **Implementations**: Provide concrete implementations for standard data structures such as `Vec<f64>` and `Mat<f64>`.
- **Builders**: Facilitate the construction and manipulation of vectors and matrices.
- **Operations**: Support essential linear algebra operations like dot products, norms, matrix-vector multiplication, etc.
- **Testing**: Ensure reliability through comprehensive unit tests.

This modular design allows users to integrate various underlying data structures and optimize for performance and memory usage.

---

## **3. Core Components**

### Vectors

- **Traits**: Define the `Vector` trait, which includes methods for vector operations.
- **Implementations**: Provide implementations for common vector types, such as Rust's `Vec<f64>` and the `Mat<f64>` type from the `faer` library.
- **Operations**: Include methods for dot product, scaling, addition, element-wise operations, cross product, and statistical functions.

### Matrices

- **Traits**: Define the `Matrix` trait, which includes methods for matrix operations.
- **Implementations**: Provide implementations for matrix types, particularly `Mat<f64>` from the `faer` library.
- **Operations**: Include methods for matrix-vector multiplication, trace, Frobenius norm, and access to elements.

---

## **4. Vector Module**

The vector module is organized into several components:

- **Traits** (`traits.rs`)
- **Implementations** (`vec_impl.rs` and `mat_impl.rs`)
- **Vector Builder** (`vector_builder.rs`)
- **Testing** (`tests.rs`)

### Vector Traits

The `Vector` trait defines a set of common operations for vectors:

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

- **Thread Safety**: All implementations must be `Send` and `Sync`.
- **Scalar Type**: The `Scalar` associated type allows for flexibility in the numeric type used.

### Vector Implementations

#### Implementation for `Vec<f64>`

The standard Rust `Vec<f64>` type implements the `Vector` trait, providing methods for vector operations.

- **Key Methods**:
  - `len()`: Returns the length of the vector.
  - `get(i)`: Retrieves the element at index `i`.
  - `set(i, value)`: Sets the element at index `i` to `value`.
  - `dot(other)`: Computes the dot product with another vector.
  - `norm()`: Calculates the Euclidean norm.
  - `scale(scalar)`: Scales the vector by a scalar.
  - `axpy(a, x)`: Performs the operation `self = a * x + self`.
  - `cross(other)`: Computes the cross product (only for 3D vectors).

**Example Usage**:

```rust
let mut vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];

// Dot product
let dot = vec1.dot(&vec2);

// Scaling
vec1.scale(2.0);

// Element-wise addition
vec1.element_wise_add(&vec2);
```

#### Implementation for `Mat<f64>`

The `Mat<f64>` type from the `faer` library is used to represent column vectors and implements the `Vector` trait.

- **Key Methods**:
  - `len()`: Returns the number of rows (since it's a column vector).
  - `get(i)`: Retrieves the element at row `i`.
  - `set(i, value)`: Sets the element at row `i` to `value`.
  - Supports all other methods defined in the `Vector` trait.

**Example Usage**:

```rust
use faer::Mat;

// Creating a column vector with 3 elements
let mut mat_vec = Mat::<f64>::zeros(3, 1);
mat_vec.set(0, 1.0);
mat_vec.set(1, 2.0);
mat_vec.set(2, 3.0);

// Computing the norm
let norm = mat_vec.norm();
```

### Vector Builder

The `VectorBuilder` struct provides methods to build and manipulate vectors generically.

- **Methods**:
  - `build_vector(size)`: Constructs a vector of a specified type and size.
  - `build_dense_vector(size)`: Constructs a dense vector using `Mat<f64>`.
  - `resize_vector(vector, new_size)`: Resizes a vector while maintaining memory safety.

**Example Usage**:

```rust
let size = 5;
let vector = VectorBuilder::build_vector::<Vec<f64>>(size);

// Resizing the vector
VectorBuilder::resize_vector(&mut vector, 10);
```

### Vector Testing

Comprehensive tests are provided to ensure the correctness of vector operations.

- **Test Cases**:
  - Length retrieval
  - Element access and modification
  - Dot product calculation
  - Norm computation
  - Scaling and axpy operations
  - Element-wise addition, multiplication, and division
  - Cross product
  - Statistical functions: sum, max, min, mean, variance

---

## **5. Matrix Module**

The matrix module includes:

- **Traits** (`traits.rs`)
- **Implementations** (`mat_impl.rs`)
- **Matrix Builder** (`matrix_builder.rs`)
- **Testing** (`tests.rs`)

### Matrix Traits

The `Matrix` trait defines essential matrix operations:

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

Additional traits for matrix construction and manipulation:

- **`MatrixOperations`**: For constructing and accessing matrix elements.
- **`ExtendedMatrixOperations`**: For resizing matrices.

### Matrix Implementations

#### Implementation for `Mat<f64>`

The `Mat<f64>` type from the `faer` library implements the `Matrix` trait.

- **Key Methods**:
  - `nrows()`: Returns the number of rows.
  - `ncols()`: Returns the number of columns.
  - `mat_vec(x, y)`: Performs matrix-vector multiplication.
  - `get(i, j)`: Retrieves the element at row `i`, column `j`.
  - `set(i, j, value)`: Sets the element at row `i`, column `j` to `value`.
  - `trace()`: Calculates the trace of the matrix.
  - `frobenius_norm()`: Computes the Frobenius norm.

**Example Usage**:

```rust
use faer::Mat;

// Creating a 3x3 zero matrix
let mut matrix = Mat::<f64>::zeros(3, 3);

// Setting elements
matrix.set(0, 0, 1.0);
matrix.set(1, 1, 1.0);
matrix.set(2, 2, 1.0);

// Matrix-vector multiplication
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y);
```

### Matrix Builder

The `MatrixBuilder` struct provides methods to build and manipulate matrices generically.

- **Methods**:
  - `build_matrix(rows, cols)`: Constructs a matrix of a specified type and dimensions.
  - `build_dense_matrix(rows, cols)`: Constructs a dense matrix using `Mat<f64>`.
  - `resize_matrix(matrix, new_rows, new_cols)`: Resizes a matrix while maintaining memory safety.
  - `apply_preconditioner(preconditioner, matrix)`: Demonstrates compatibility with preconditioners.

**Example Usage**:

```rust
let rows = 4;
let cols = 4;
let matrix = MatrixBuilder::build_matrix::<Mat<f64>>(rows, cols);

// Resizing the matrix
MatrixBuilder::resize_matrix(&mut matrix, 5, 5);
```

### Matrix Testing

Comprehensive tests are provided to ensure the correctness of matrix operations.

- **Test Cases**:
  - Dimension retrieval
  - Element access and modification
  - Matrix-vector multiplication with different vector types
  - Trace and Frobenius norm calculations
  - Thread safety
  - Handling of edge cases (e.g., out-of-bounds access)

---

## **6. Using the Linear Algebra Module**

This section provides practical examples of how to use the `Linear Algebra` module in Hydra.

### Creating Vectors

**Using `Vec<f64>`**:

```rust
let mut vector = vec![0.0; 5]; // Creates a vector of length 5 initialized with zeros.
vector.set(0, 1.0); // Sets the first element to 1.0.
```

**Using `Mat<f64>` from `faer`**:

```rust
use faer::Mat;

let mut mat_vector = Mat::<f64>::zeros(5, 1); // Creates a column vector with 5 rows.
mat_vector.set(0, 1.0); // Sets the first element to 1.0.
```

### Performing Vector Operations

**Dot Product**:

```rust
let vec1 = vec![1.0, 2.0, 3.0];
let vec2 = vec![4.0, 5.0, 6.0];
let dot = vec1.dot(&vec2); // Computes the dot product.
```

**Norm Calculation**:

```rust
let norm = vec1.norm(); // Calculates the Euclidean norm of vec1.
```

**Scaling and AXPY Operation**:

```rust
vec1.scale(2.0); // Scales vec1 by 2.0.
vec1.axpy(1.5, &vec2); // Performs vec1 = 1.5 * vec2 + vec1.
```

**Element-wise Operations**:

```rust
vec1.element_wise_add(&vec2); // Adds vec2 to vec1 element-wise.
vec1.element_wise_mul(&vec2); // Multiplies vec1 by vec2 element-wise.
```

### Creating Matrices

**Using `Mat<f64>`**:

```rust
use faer::Mat;

// Creating a 3x3 zero matrix
let mut matrix = Mat::<f64>::zeros(3, 3);

// Setting elements
matrix.set(0, 0, 1.0);
matrix.set(1, 1, 2.0);
matrix.set(2, 2, 3.0);
```

### Performing Matrix Operations

**Matrix-Vector Multiplication**:

```rust
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0; 3];
matrix.mat_vec(&x, &mut y); // Computes y = matrix * x.
```

**Trace and Norm Calculations**:

```rust
let trace = matrix.trace(); // Calculates the trace of the matrix.
let fro_norm = matrix.frobenius_norm(); // Calculates the Frobenius norm.
```

---

## **7. Best Practices**

- **Thread Safety**: Ensure that vectors and matrices used across threads implement `Send` and `Sync`.
- **Consistent Dimensions**: Always verify that vector and matrix dimensions are compatible for operations like multiplication and addition.
- **Error Handling**: Handle potential errors, such as out-of-bounds access or invalid dimensions for operations (e.g., cross product requires 3D vectors).
- **Performance Optimization**: Utilize efficient data structures and avoid unnecessary copies by using slices and references where appropriate.
- **Testing**: Incorporate unit tests to verify the correctness of custom implementations or extensions.

---

## **8. Conclusion**

The `Linear Algebra` module in Hydra provides a flexible and robust framework for vector and matrix operations essential in computational simulations. By defining abstract traits and providing concrete implementations, it allows for extensibility and optimization based on specific needs. Proper utilization of this module ensures that numerical computations are accurate, efficient, and maintainable.