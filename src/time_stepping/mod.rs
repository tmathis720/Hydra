/// Time-stepping module.
///
/// This module provides functionality for implementing time-stepping
/// schemes used in numerical simulations. It is structured into
/// submodules for defining general time-stepping interfaces, specific
/// methods, and adaptivity strategies.

/// Submodule for defining core time-stepping interfaces and structures.
pub mod ts;

/// Submodule for implementing specific time-stepping methods, such as
/// Euler and Backward Euler schemes.
pub mod methods;

/// Submodule for implementing time-step adaptivity strategies, which
/// allow dynamic adjustment of time-step sizes based on solution requirements.
pub mod adaptivity;

// Re-exports for ease of use
pub use ts::{
    TimeStepper, // The main trait defining the interface for a time-stepper.
    TimeSteppingError, // Error type for handling issues during time-stepping.
    TimeDependentProblem // Trait for problems that evolve over time and can be solved using a time-stepper.
};

/// Backward Euler time-stepping method, a first-order implicit scheme.
pub use methods::backward_euler::BackwardEuler;

/// Explicit Euler time-stepping method, a first-order explicit scheme.
pub use methods::euler::ExplicitEuler;

#[cfg(test)]
mod tests;