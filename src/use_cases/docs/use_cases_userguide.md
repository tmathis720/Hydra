# Hydra `Use Cases` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the `use_cases` Module](#2-overview-of-the-use_cases-module)  
3. [Core Components](#3-core-components)  
   - [Matrix Construction](#matrix-construction)
   - [RHS Construction](#rhs-construction)
   - [PISO Solver Suite](#piso-solver-suite)
4. [Using the Matrix and RHS Builders](#4-using-the-matrix-and-rhs-builders)
   - [Building and Initializing a Matrix](#building-and-initializing-a-matrix)
   - [Building and Initializing an RHS Vector](#building-and-initializing-an-rhs-vector)
5. [PISO Algorithm Workflow](#5-piso-algorithm-workflow)
   - [Predictor Step](#predictor-step)
   - [Pressure Correction Step](#pressure-correction-step)
   - [Velocity Correction Step](#velocity-correction-step)
   - [Nonlinear Loop](#nonlinear-loop)
   - [Boundary Handling in PISO](#boundary-handling-in-piso)
6. [Example Usage](#6-example-usage)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the **`use_cases`** module of the Hydra computational framework. This module provides **higher-level use cases and workflows** that build upon Hydra’s core features (e.g., domain handling, time stepping, solver management). The primary goals are:

- To **construct and initialize** matrices and RHS vectors for linear systems.
- To demonstrate the **PISO** (Pressure-Implicit with Splitting of Operators) solver steps, often used in fluid dynamics simulations.

---

## **2. Overview of the `use_cases` Module**

The `use_cases` directory collects higher-level routines that combine or orchestrate lower-level functionalities from other Hydra modules. Its structure is:

```bash
src/use_cases/
├── matrix_construction.rs
├── rhs_construction.rs
├── piso/
│   ├── boundary.rs
│   ├── nonlinear_loop.rs
│   ├── predictor.rs
│   ├── pressure_correction.rs
│   ├── velocity_correction.rs
│   └── mod.rs
└── mod.rs
```

**Key Submodules**:

1. **`matrix_construction`**: Tools to create, resize, and initialize matrices for simulations.  
2. **`rhs_construction`**: Tools to create and initialize the right-hand side (RHS) vectors.  
3. **`piso`**: Implementation of the **PISO** solver approach, including predictor, pressure correction, velocity correction, boundary condition handling, and a nonlinear iteration loop.

---

## **3. Core Components**

### Matrix Construction

- **File**: [`matrix_construction.rs`](./matrix_construction.rs)  
- **Struct**: `MatrixConstruction`

```rust
pub struct MatrixConstruction;

impl MatrixConstruction {
    pub fn build_zero_matrix(rows: usize, cols: usize) -> Mat<f64> { ... }
    pub fn initialize_matrix_with_value<T: MatrixOperations>(matrix: &mut T, value: f64) { ... }
    pub fn resize_matrix<T: ExtendedMatrixOperations>(matrix: &mut T, new_rows: usize, new_cols: usize) { ... }
}
```

**Purpose**:  
- Build Faer-based dense matrices of specified dimensions.  
- Initialize them (set all entries to a particular value).  
- Resize while preserving data if possible.

### RHS Construction

- **File**: [`rhs_construction.rs`](./rhs_construction.rs)  
- **Struct**: `RHSConstruction`

```rust
pub struct RHSConstruction;

impl RHSConstruction {
    pub fn build_zero_rhs(size: usize) -> Mat<f64> { ... }
    pub fn initialize_rhs_with_value<T: Vector<Scalar = f64>>(vector: &mut T, value: f64) { ... }
    pub fn resize_rhs(vector: &mut Mat<f64>, new_size: usize) { ... }
}
```

**Purpose**:  
- Create a **dense vector** for the right-hand side of a linear system.  
- Fill that vector with some initial condition.  
- Resize it for changing problem sizes.

### PISO Solver Suite

- **Folder**: [`piso/`](./piso)  
- **Main**: [`mod.rs`](./piso/mod.rs) with the `PISOSolver`  
- **Submodules**: 
  - `predictor.rs`  
  - `pressure_correction.rs`  
  - `velocity_correction.rs`  
  - `nonlinear_loop.rs`  
  - `boundary.rs`  

**Purpose**: Provide a cohesive **PISO** implementation. The steps are typically:

1. **Predictor**: Solve momentum equation ignoring updated pressure.  
2. **Pressure Correction**: Solve Poisson equation for pressure.  
3. **Velocity Correction**: Adjust velocity to enforce continuity.  
4. **Nonlinear Loop**: Repeat until convergence or iteration limit.

Within the submodules:

- **`predictor`**: The velocity predictor step.  
- **`pressure_correction`**: Solves the pressure Poisson system, obtains corrected pressure.  
- **`velocity_correction`**: Uses pressure correction to fix velocity field.  
- **`nonlinear_loop`**: Repeats steps until the flow solution converges.  
- **`boundary`**: Specialized boundary condition applications for PISO steps.

---

## **4. Using the Matrix and RHS Builders**

### Building and Initializing a Matrix

```rust
use hydra::use_cases::matrix_construction::MatrixConstruction;
use faer::Mat;

fn main() {
    // Create a 5x5 zero matrix
    let mut matrix = MatrixConstruction::build_zero_matrix(5, 5);

    // Initialize all elements to 2.5
    MatrixConstruction::initialize_matrix_with_value(&mut matrix, 2.5);

    // Resize to 7x7, preserving the top-left 5x5 block
    MatrixConstruction::resize_matrix(&mut matrix, 7, 7);

    println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());
}
```

### Building and Initializing an RHS Vector

```rust
use hydra::use_cases::rhs_construction::RHSConstruction;
use faer::Mat;

fn main() {
    // Create a zero vector of length 5
    let mut rhs = RHSConstruction::build_zero_rhs(5);

    // Fill the RHS with a constant value of 1.0
    RHSConstruction::initialize_rhs_with_value(&mut rhs, 1.0);

    // Resize to length 8
    RHSConstruction::resize_rhs(&mut rhs, 8);
}
```

---

## **5. PISO Algorithm Workflow**

Below is a **high-level** explanation of the PISO approach as implemented in the `piso` submodule:

1. **Predictor Step**:  
   - Solve the momentum equation ignoring any new pressure correction.  
   - Typically updates velocity using an approximate pressure from the previous iteration.

2. **Pressure Correction Step**:  
   - Formulate and solve the pressure Poisson equation.  
   - Compute the correction field for pressure to ensure mass conservation.

3. **Velocity Correction Step**:  
   - Adjust velocity with the newly computed pressure correction to enforce divergence-free flow.

4. **Nonlinear Loop**:  
   - Repeats the predictor → pressure → velocity corrections until residuals meet a tolerance or iteration limit.

5. **Boundary Handling**:  
   - The `boundary.rs` file shows how boundary conditions are specifically adapted for the PISO steps (especially for pressure Poisson).

### Predictor Step

- **File**: [`predictor.rs`](./piso/predictor.rs)  
- **Function**: `predict_velocity(...)`

**Key Points**:  
- Receives mesh, fields, fluxes, etc.  
- Assembles momentum fluxes and updates the velocity field.  
- Ignores the new pressure correction in this stage.

### Pressure Correction Step

- **File**: [`pressure_correction.rs`](./piso/pressure_correction.rs)  
- **Function**: `solve_pressure_poisson(...)`  

**Key Points**:  
- Assembles the matrix for the pressure Poisson equation.  
- Solves it using a `KSP` solver (e.g., Conjugate Gradient).  
- Outputs a `PressureCorrectionResult` containing the **residual** measure.

### Velocity Correction Step

- **File**: [`velocity_correction.rs`](./piso/velocity_correction.rs)  
- **Function**: `correct_velocity(...)`

**Key Points**:  
- Uses the **pressure gradient** from the correction step to adjust velocity.  
- Ensures continuity / divergence-free condition.

### Nonlinear Loop

- **File**: [`nonlinear_loop.rs`](./piso/nonlinear_loop.rs)  
- **Function**: `solve_nonlinear_system(...)`

**Key Points**:  
- Orchestrates multiple predictor → correction cycles until convergence.  
- Checks residual from the pressure correction result to decide stopping.  
- If not converged within `max_iterations`, returns an error.

### Boundary Handling in PISO

- **File**: [`boundary.rs`](./piso/boundary.rs)  
- **Function**: `apply_pressure_poisson_bc(...)`

**Key Points**:  
- Specialized routine to apply boundary conditions to the matrix and RHS in the pressure Poisson step.  
- Uses Hydra’s boundary condition abstractions.

---

## **6. Example Usage**

Consider a scenario where you want to:

1. Build a system matrix and RHS.  
2. Run a partial PISO iteration.

```rust
use hydra::{
   use_cases::{
       matrix_construction::MatrixConstruction,
       rhs_construction::RHSConstruction,
       piso::{PISOSolver, PISOConfig},
   },
   // ...other Hydra modules
};

fn main() {
   // Step 1: Build and initialize a matrix
   let mut mat = MatrixConstruction::build_zero_matrix(10, 10);
   MatrixConstruction::initialize_matrix_with_value(&mut mat, 0.0);

   // Step 2: Build and initialize an RHS vector
   let mut rhs = RHSConstruction::build_zero_rhs(10);
   RHSConstruction::initialize_rhs_with_value(&mut rhs, 5.0);

   // Step 3: Create a PISO solver with a mesh, time stepper, and config
   let mesh = Mesh::new(); // Suppose we have a mesh
   let time_stepper = Box::new(...); // Provide a valid TimeStepper
   let config = PISOConfig { max_iterations: 5, tolerance: 1e-5, relaxation_factor: 0.7 };
   let mut piso_solver = PISOSolver::new(mesh, time_stepper, config);

   // Step 4: Solve with PISO for one step (requires a problem + state)
   // ...
}
```

---

## **7. Best Practices**

1. **Matrix & RHS**:  
   - Use `MatrixConstruction` and `RHSConstruction` for consistent creation/resizing.  
   - Initialize values carefully to avoid leftover data from prior simulations.

2. **PISO**:  
   - Ensure your mesh and boundary conditions are set properly before calling predictor/correction steps.  
   - Monitor the pressure Poisson solver’s **residual** to confirm convergence.  
   - Use the `nonlinear_loop` functionality for advanced iterative flows if needed.

3. **Integration**:  
   - The `use_cases` are building blocks. Combine them with Hydra’s domain, boundary, and solver modules for robust simulations.  
   - Keep track of the current simulation **time** and **time step** in PISO updates.

---

## **8. Conclusion**

The **`use_cases`** module in Hydra provides essential, higher-level building blocks for typical solver workflows:

- **Matrix construction** (`matrix_construction`) and **RHS building** (`rhs_construction`).
- A **PISO** solver suite that orchestrates the classic predictor, pressure correction, and velocity correction steps, with a nonlinear iteration loop and specialized boundary condition handling.

By combining these routines with Hydra’s domain, boundary, and solver infrastructure, users can **rapidly implement** advanced simulation pipelines for fluid dynamics, CFD, or other PDE-based problems.