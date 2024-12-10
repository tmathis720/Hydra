// Module imports for fields and fluxes, which represent the state and flux data of the system.
use fields::{Fields, Fluxes};

use crate::{
    boundary::bc_handler::BoundaryConditionHandler, // Handles boundary conditions for the domain.
    Mesh, // Represents the computational mesh.
};

// Submodules related to different aspects of equation solving and field computation.
pub mod equation;        // Core definitions and logic for equations.
pub mod reconstruction;  // Methods for reconstructing values (e.g., higher-order reconstruction).
pub mod gradient;        // Gradient computation methods for fields.
pub mod flux_limiter;    // Flux limiting techniques for stability in numerical solvers.

// Field-related modules for managing physical fields and flux computations.
pub mod fields;          // Management of scalar, vector, and tensor fields.
pub mod manager;         // Orchestrates equations and fields over a domain.
pub mod energy_equation; // Defines the energy equation and related computations.
pub mod momentum_equation; // Defines the momentum equation and related computations.
pub mod turbulence_models; // Defines the turbulence closure model and handles related computations.

/// A trait representing a physical equation in the simulation.
///
/// Implementors of this trait define the behavior for assembling fluxes and updating fields
/// based on the governing equation. The `assemble` method integrates the equation over the
/// domain mesh, using the current state of fields and boundary conditions.
///
/// # Arguments
/// - `domain`: The computational mesh defining the domain.
/// - `fields`: The state of scalar, vector, and tensor fields in the domain.
/// - `fluxes`: The fluxes to be updated during the assembly process.
/// - `boundary_handler`: The handler for enforcing boundary conditions.
/// - `current_time`: The current simulation time.
pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

/// Module for testing functionality related to equations and fields.
///
/// Contains unit tests and integration tests for validating module behavior.
#[cfg(test)]
pub mod tests;
