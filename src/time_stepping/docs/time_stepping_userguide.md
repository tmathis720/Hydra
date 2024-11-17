# Hydra `Time Stepping` Module User Guide

---

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Overview of the Time Stepping Module](#2-overview-of-the-time-stepping-module)
3. [Core Components](#3-core-components)
   - [TimeDependentProblem Trait](#timedependentproblem-trait)
   - [TimeStepper Trait](#timestepper-trait)
4. [Implemented Time Stepping Methods](#4-implemented-time-stepping-methods)
   - [Forward Euler Method](#forward-euler-method)
   - [Backward Euler Method](#backward-euler-method)
5. [Using the Time Stepping Module](#5-using-the-time-stepping-module)
   - [Defining a Time-Dependent Problem](#defining-a-time-dependent-problem)
   - [Selecting a Time Stepping Method](#selecting-a-time-stepping-method)
   - [Performing Time Steps](#performing-time-steps)
6. [Planned Features and Not Yet Implemented Components](#6-planned-features-and-not-yet-implemented-components)
   - [Adaptive Time Stepping](#adaptive-time-stepping)
   - [Higher-Order Methods](#higher-order-methods)
   - [Step Size Control and Error Estimation](#step-size-control-and-error-estimation)
7. [Best Practices](#7-best-practices)
8. [Conclusion](#8-conclusion)

---

## **1. Introduction**

Welcome to the user's guide for the `Time Stepping` module of the Hydra computational framework. This module provides tools and interfaces for numerically solving time-dependent problems, such as ordinary differential equations (ODEs) and partial differential equations (PDEs). Time stepping is a critical component in simulations that evolve over time, and Hydra's module is designed to offer flexibility and extensibility for various time integration methods.

**Note**: As of the current version, some parts of the module are not yet implemented. This guide will point out the implemented features and those planned for future development.

---

## **2. Overview of the Time Stepping Module**

The `Time Stepping` module is structured to facilitate the integration of different time-stepping methods and to provide a unified interface for time-dependent problems. The key components include:

- **Traits**: Define interfaces for time-dependent problems and time-stepping methods.
- **Time Stepping Methods**: Implementations of specific algorithms like Forward Euler and Backward Euler.
- **Adaptivity Components** (Planned): Modules for error estimation and adaptive step size control.

**Module Structure**:

```bash
time_stepping/
├── adaptivity/
│   ├── error_estimate.rs      # Not yet implemented
│   ├── step_size_control.rs   # Not yet implemented
│   └── mod.rs                 # Not yet implemented
├── methods/
│   ├── backward_euler.rs      # Implemented
│   ├── euler.rs               # Implemented
│   ├── crank_nicolson.rs      # Not yet implemented
│   ├── runge_kutta.rs         # Not yet implemented
│   └── mod.rs
├── ts.rs                      # Core traits and structures
└── mod.rs                     # Module exports
```

---

## **3. Core Components**

### TimeDependentProblem Trait

The `TimeDependentProblem` trait defines the interface for any time-dependent problem that can be solved using the time-stepping methods provided. Implementing this trait requires specifying:

- **State Type**: The type representing the system's state, which must implement the `Vector` trait.
- **Time Type**: The type representing time, typically `f64`.
- **Methods**:
  - `compute_rhs`: Computes the right-hand side (RHS) of the system.
  - `initial_state`: Provides the initial state of the system.
  - `time_to_scalar`: Converts time values to the scalar type used in vectors.
  - `get_matrix`: Returns a matrix representation if applicable (used in implicit methods).
  - `solve_linear_system`: Solves linear systems for implicit methods.

**Trait Definition**:

```rust
pub trait TimeDependentProblem {
    type State: Vector;
    type Time;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar;

    fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}
```

### TimeStepper Trait

The `TimeStepper` trait defines the interface for time-stepping methods. It requires the implementation of:

- `step`: Advances the solution by one time step.
- `adaptive_step`: Performs an adaptive time step (not fully implemented yet).
- `set_time_interval`: Sets the start and end times for the simulation.
- `set_time_step`: Sets the fixed time step size.

**Trait Definition**:

```rust
pub trait TimeStepper<P: TimeDependentProblem> {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    fn set_time_step(&mut self, dt: P::Time);
}
```

---

## **4. Implemented Time Stepping Methods**

As of the current version, the following time-stepping methods are implemented:

### Forward Euler Method

The Forward Euler method is an explicit first-order method for numerically integrating ordinary differential equations.

**Implementation Highlights**:

- **Module**: `euler.rs`
- **Struct**: `ForwardEuler`
- **Key Characteristics**:
  - Simple and easy to implement.
  - Suitable for problems where accuracy and stability are not critical.

**Usage**:

Implementing the `TimeStepper` trait for `ForwardEuler`:

```rust
pub struct ForwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for ForwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut rhs = problem.initial_state();
        problem.compute_rhs(time, state, &mut rhs)?;
        let scalar_dt = problem.time_to_scalar(dt);
        state.axpy(scalar_dt, &rhs);
        Ok(())
    }
    
    // Other methods...
}
```

**Key Methods**:

- **`step`**: Performs the explicit update `state = state + dt * rhs`.

### Backward Euler Method

The Backward Euler method is an implicit first-order method, offering better stability properties compared to the Forward Euler method.

**Implementation Highlights**:

- **Module**: `backward_euler.rs`
- **Struct**: `BackwardEuler`
- **Key Characteristics**:
  - Implicit method requiring the solution of a linear system at each time step.
  - More stable for stiff problems.

**Usage**:

Implementing the `TimeStepper` trait for `BackwardEuler`:

```rust
pub struct BackwardEuler;

impl<P: TimeDependentProblem> TimeStepper<P> for BackwardEuler {
    fn step(
        &mut self,
        problem: &P,
        time: P::Time,
        dt: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut matrix = problem.get_matrix();
        let mut rhs = problem.initial_state();
        problem.compute_rhs(time, state, &mut rhs)?;
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;
        Ok(())
    }
    
    // Other methods...
}
```

**Key Methods**:

- **`step`**: Involves computing the RHS and solving the linear system `A * state = rhs`.

---

## **5. Using the Time Stepping Module**

### Defining a Time-Dependent Problem

To use the time-stepping methods, you need to define a struct that implements the `TimeDependentProblem` trait.

**Example**:

```rust
struct MyProblem {
    // Problem-specific fields
}

impl TimeDependentProblem for MyProblem {
    type State = Vec<f64>;
    type Time = f64;

    fn initial_state(&self) -> Self::State {
        // Return the initial state vector
    }

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Compute the RHS based on the current state and time
    }

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
        time
    }

    fn get_matrix(&self) -> Box<dyn Matrix<Scalar = f64>> {
        // Return the system matrix if needed (for implicit methods)
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Solve the linear system for implicit methods
    }
}
```

### Selecting a Time Stepping Method

Choose a time-stepping method based on the problem's requirements:

- **Forward Euler**: For simple, non-stiff problems where computational efficiency is important.
- **Backward Euler**: For stiff problems requiring stability.

**Example**:

```rust
let mut stepper = ForwardEuler;
// or
let mut stepper = BackwardEuler;
```

### Performing Time Steps

Set up the time interval and time step size (if applicable):

```rust
stepper.set_time_interval(0.0, 10.0);
stepper.set_time_step(0.1);
```

Perform the time-stepping loop:

```rust
let mut state = problem.initial_state();
let mut time = 0.0;
let end_time = 10.0;
let dt = 0.1;

while time < end_time {
    stepper.step(&problem, time, dt, &mut state)?;
    time += dt;
}
```

**Error Handling**:

- Each `step` method returns a `Result`. Handle errors appropriately.
- Use `?` operator or match statements to manage `TimeSteppingError`.

---

## **6. Planned Features and Not Yet Implemented Components**

The `Time Stepping` module has several components and features planned for future implementation:

### Adaptive Time Stepping

- **Description**: Adjusting the time step size dynamically based on error estimates to improve efficiency and accuracy.
- **Current Status**: The `adaptive_step` method is defined in the `TimeStepper` trait but not fully implemented in existing methods.
- **Planned Components**:
  - **Error Estimation**: Modules to estimate the local truncation error.
  - **Step Size Control**: Algorithms to adjust `dt` based on error estimates.

### Higher-Order Methods

- **Crank-Nicolson Method**: A second-order implicit method combining Forward and Backward Euler.
- **Runge-Kutta Methods**: Higher-order explicit methods for improved accuracy.
- **Current Status**: These methods are listed in the module structure but not yet implemented.

### Step Size Control and Error Estimation

- **Modules**:
  - **`error_estimate.rs`**: Will provide functionalities for error estimation.
  - **`step_size_control.rs`**: Will implement algorithms for adjusting the time step size.
- **Adaptivity Module**: The `adaptivity` folder contains placeholders for these components.

**Note**: Users interested in these features should keep an eye on future releases of Hydra for updates.

---

## **7. Best Practices**

- **Choose the Right Method**: Select a time-stepping method appropriate for your problem's stiffness and accuracy requirements.
- **Implement Required Traits**: Ensure that your problem struct correctly implements all methods of the `TimeDependentProblem` trait.
- **Handle Errors**: Always handle potential errors returned by the `step` methods to avoid unexpected crashes.
- **Monitor Stability**: Be cautious with explicit methods for stiff problems; consider using implicit methods instead.
- **Stay Updated**: Keep track of updates to the module for new features and methods as they are implemented.

---

## **8. Conclusion**

The `Time Stepping` module in Hydra provides a flexible framework for integrating time-dependent problems using various numerical methods. While the current implementation includes fundamental methods like Forward and Backward Euler, the framework is designed to accommodate more advanced techniques in the future.

By defining clear interfaces through the `TimeDependentProblem` and `TimeStepper` traits, users can implement custom problems and apply different time-stepping strategies with ease. The planned features, such as adaptive time stepping and higher-order methods, will further enhance the module's capabilities.

---

**Note**: As the module is still under development, some features are not yet available. Users are encouraged to contribute to the project or check back for updates in future releases.