# Hydra `Time Stepping` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Overview of the Time Stepping Module](#2-overview-of-the-time-stepping-module)  
3. [Core Components](#3-core-components)  
   - [TimeDependentProblem Trait](#timedependentproblem-trait)  
   - [TimeStepper Trait](#timestepper-trait)  
   - [FixedTimeStepper](#fixedtimestepper)  
4. [Implemented Time Stepping Methods](#4-implemented-time-stepping-methods)  
   - [Explicit Euler Method](#explicit-euler-method)  
   - [Backward Euler Method](#backward-euler-method)  
   - [Runge-Kutta (Partial)](#runge-kutta-partial)  
5. [Using the Time Stepping Module](#5-using-the-time-stepping-module)  
   - [Defining a Time-Dependent Problem](#defining-a-time-dependent-problem)  
   - [Selecting a Time Stepping Method](#selecting-a-time-stepping-method)  
   - [Performing Time Steps](#performing-time-steps)  
6. [Adaptivity and Planned Features](#6-adaptivity-and-planned-features)  
   - [Adaptive Time Stepping](#adaptive-time-stepping)  
   - [Crank-Nicolson Method](#crank-nicolson-method)  
   - [Step Size Control and Error Estimation](#step-size-control-and-error-estimation)  
7. [Best Practices](#7-best-practices)  
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the **`Time Stepping`** module in the Hydra computational framework. This module is designed for evolving **time-dependent** problems, such as ordinary differential equations (ODEs) and partial differential equations (PDEs). It aims to provide:

- **A unified interface** via the `TimeDependentProblem` and `TimeStepper` traits.
- **Multiple time integration methods** such as **explicit** and **implicit** Euler approaches, with partial or planned support for more advanced schemes (Runge-Kutta, Crank-Nicolson).
- **Optional adaptivity** to adjust the time step based on error estimates (in partial form).

The design emphasizes **modularity** and **extensibility**, allowing new methods or adaptivity techniques to be added cleanly.

---

## **2. Overview of the Time Stepping Module**

Below is a simplified file structure:

```
time_stepping/
├── mod.rs
├── ts.rs                 // Core traits: TimeDependentProblem, TimeStepper
├── methods/
│   ├── euler.rs          // Explicit Euler
│   ├── backward_euler.rs // Backward Euler
│   ├── runge_kutta.rs    // Partial Runge-Kutta
│   ├── crank_nicolson.rs // Planned
│   └── mod.rs
├── adaptivity/
│   ├── error_estimate.rs    // For local error estimation
│   ├── step_size_control.rs // For adjusting dt
│   └── mod.rs
└── tests.rs
```

Key submodules:

1. **`ts.rs`**:  
   - Defines `TimeDependentProblem` for problem specification.  
   - Defines `TimeStepper` trait for time-stepping logic.  
   - Offers a sample `FixedTimeStepper` that can step forward in fixed increments.  
   - Includes a custom error type `TimeSteppingError`.

2. **`methods/`**:  
   - `ExplicitEuler` (in `euler.rs`)  
   - `BackwardEuler` (in `backward_euler.rs`)  
   - `RungeKutta` (in `runge_kutta.rs`)  
   - `Crank-Nicolson` is **planned** but not yet implemented.  

3. **`adaptivity/`** (in partial form):  
   - `error_estimate.rs`: Demo function to estimate error by comparing single-step vs. multi-step approaches.  
   - `step_size_control.rs`: A function `adjust_step_size` that modifies dt based on the computed error.

---

## **3. Core Components**

### TimeDependentProblem Trait

Defines how a **time-dependent** system is specified:

```rust
pub trait TimeDependentProblem {
    type State: Clone + UpdateState;
    type Time: Copy + PartialOrd + Add<Output = Self::Time> + From<f64> + Into<f64>;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}
```

- **`compute_rhs(...)`**: Fills `derivative` with \(\frac{d}{dt}\) of the system at `time` and `state`.
- **`initial_state()`**: Returns the system’s initial condition.
- **`get_matrix()`** / `solve_linear_system(...)`: For **implicit** methods requiring matrix solves.

The `State` typically implements Hydra’s `UpdateState` trait (from `equation::fields`) to handle state updates like `state = state + alpha * derivative`.

### TimeStepper Trait

Specifies how to **advance** a system in time:

```rust
pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);
    fn set_time_step(&mut self, dt: P::Time);
    fn get_time_step(&self) -> P::Time;
    fn current_time(&self) -> P::Time;
    fn set_current_time(&mut self, time: P::Time);

    fn get_solver(&mut self) -> &mut dyn KSP; // Some methods might solve systems with a KSP solver
}
```

- **`step(...)`**: Does one time step of size `dt`.
- **`adaptive_step(...)`**: (Optional) for adaptive stepping. Currently partial in some methods.
- **`get_solver()`**: Access underlying solver if needed for matrix solves.

### FixedTimeStepper

An **example** implementation that maintains:

- `current_time`
- `start_time`
- `end_time`
- `time_step`
- A solver manager (though it may not be used in simple explicit steps).

This is one approach to iterating from `start_time` to `end_time` in increments of `time_step`.

---

## **4. Implemented Time Stepping Methods**

### Explicit Euler Method

**File**: `methods/euler.rs`  
**Struct**: `ExplicitEuler<P: TimeDependentProblem>`

- **Description**: Also known as “Forward Euler”, a first-order **explicit** scheme:
  \[
    y_{n+1} = y_n + dt \cdot f(t_n, y_n).
  \]
- **Implementation**:
  - `step(...)`: calls `compute_rhs(...)`, then updates the state with `state = state + dt * derivative`.
  - `adaptive_step(...)`: partial example using error estimates.  
- **Pros**: Very simple, cheap per-step.  
- **Cons**: Potentially unstable if the problem is stiff or if dt is too large.

### Backward Euler Method

**File**: `methods/backward_euler.rs`  
**Struct**: `BackwardEuler`

- **Description**: A **first-order implicit** scheme:
  \[
    y_{n+1} = y_n + dt \cdot f(t_{n+1}, y_{n+1}).
  \]
- **Implementation**:
  - **Requires** `get_matrix()` and `solve_linear_system(...)` from the problem.  
  - Calls `compute_rhs(...)`, then does an implicit solve to get the new state.  
- **Pros**: **Stable** for stiff problems.  
- **Cons**: Each step solves a linear system, so more expensive per-step.

### Runge-Kutta (Partial)

**File**: `methods/runge_kutta.rs`  
**Struct**: `RungeKutta<P>`

- **Current**: The code sets up `stages` but does not fully implement classical RK specifics.  
- **`step(...)`**: Loops over `stages` to compute intermediate states (`k` vectors).  
- **Pros**: Higher potential accuracy than Euler if completed.  
- **Status**: Partially functional. Not yet integrated with adaptivity or advanced Butcher tables, etc.

**Note**: Another partially planned method is `Crank-Nicolson` (in `methods/crank_nicolson.rs`), not currently implemented.

---

## **5. Using the Time Stepping Module**

### Defining a Time-Dependent Problem

Create a struct that implements **`TimeDependentProblem`**:

```rust
use hydra::time_stepping::{TimeDependentProblem, TimeSteppingError};
use hydra::linalg::Matrix;

#[derive(Clone)]
struct MyState {
    data: Vec<f64>,
    // possibly more fields
}

struct MyTimeDependentSystem;

impl TimeDependentProblem for MyTimeDependentSystem {
    type State = MyState;
    type Time = f64;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Fill derivative based on the ODE or PDE
        Ok(())
    }

    fn initial_state(&self) -> Self::State {
        MyState { data: vec![1.0, 2.0, 3.0] }
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar=f64>>> {
        None // or Some(...) if implicit methods need a matrix
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar=f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Solve the linear system if needed (e.g., for Backward Euler).
        Ok(())
    }
}
```

### Selecting a Time Stepping Method

Choose from:

- **`ExplicitEuler`**: `ExplicitEuler::new(dt, start, end)`
- **`BackwardEuler`**: `BackwardEuler::new(start_time, dt)`
- **`FixedTimeStepper`**: A generic approach that calls `compute_rhs`.

**Example**:
```rust
use hydra::time_stepping::methods::euler::ExplicitEuler;
let mut stepper = ExplicitEuler::new(0.01, 0.0, 1.0);
```

### Performing Time Steps

1. **Initialize**:
   ```rust
   let system = MyTimeDependentSystem;
   let mut state = system.initial_state();
   let mut time = 0.0;
   let end_time = 1.0;
   let dt = 0.01;
   ```
2. **Loop**:
   ```rust
   while time < end_time {
       stepper.step(&system, dt, time, &mut state)?;
       time += dt;
   }
   ```
3. **Error Handling**:
   ```rust
   if let Err(e) = stepper.step(&system, dt, time, &mut state) {
       eprintln!("Time stepping error: {:?}", e);
       break;
   }
   ```
4. For **adaptive** steps, call `stepper.adaptive_step(...)` if available.

---

## **6. Adaptivity and Planned Features**

The module includes placeholders for **error estimation** and **step-size control**:

### Adaptive Time Stepping

- **`adaptivity/error_estimate.rs`**: A sample function that compares single-step vs. multi-step solutions to estimate local error.  
- **`adaptivity/step_size_control.rs`**: Adjusts dt based on the ratio \(\sqrt{tol / error}\).  
- **`TimeStepper`** has an `adaptive_step(...)` method. 
  - Implementations in `ExplicitEuler` or `RungeKutta` are partially complete.

### Crank-Nicolson Method

- **File**: `methods/crank_nicolson.rs`  
- **Status**: *Not yet implemented.*  
- **Plan**: A second-order implicit scheme that averages explicit/implicit Euler steps.

### Step Size Control and Error Estimation

- **`estimate_error(...)`**: Inside `error_estimate.rs`, returns a numerical error measure.
- **`adjust_step_size(...)`**: In `step_size_control.rs`, modifies dt based on the computed error.

**Current**: Basic logic is provided, but a fully robust adaptive loop is still under development.

---

## **7. Best Practices**

1. **Method Selection**: 
   - Use **ExplicitEuler** for simpler or non-stiff equations. 
   - Prefer **BackwardEuler** for stiff systems or stability concerns.
2. **Implement Required Trait Methods**: Ensure your `TimeDependentProblem` provides everything needed by your chosen scheme (e.g., a matrix for implicit methods).
3. **Monitor Stability**: For stiff problems, explicit methods can fail unless dt is very small. 
4. **Error Checking**: 
   - The `step(...)` method returns a `Result`— handle `TimeSteppingError` carefully.
5. **Adaptive Steps**: 
   - If the problem changes rapidly, consider partial adaptivity in `ExplicitEuler` or `RungeKutta`.

---

## **8. Conclusion**

Hydra’s **`Time Stepping`** module supplies a flexible, trait-based framework for **time integration**. Users can define custom time-dependent problems, pick or implement a **TimeStepper** method (e.g., **Explicit Euler**, **Backward Euler**), and optionally incorporate **adaptivity**. The partial code for **RungeKutta** and future **Crank-Nicolson** expansions underscores ongoing improvements.

By following the guidelines and employing the module’s abstractions (particularly `TimeDependentProblem` + `TimeStepper`), it’s straightforward to integrate time-stepping logic into Hydra-based simulations—and to extend or refine these methods to meet advanced simulation requirements.