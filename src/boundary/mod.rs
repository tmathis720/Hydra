//! Boundary Module
//!
//! This module provides functionality for defining and applying various types of boundary
//! conditions (Dirichlet, Neumann, Robin, etc.) to mesh entities in a computational fluid
//! dynamics (CFD) simulation.
//!
//! # Overview
//! - `BoundaryCondition`: Enum representing supported boundary condition types.
//! - `BoundaryConditionHandler`: Manages boundary conditions for mesh entities.
//! - `BoundaryConditionApply`: Trait for applying boundary conditions to system matrices.
//!
//! # Computational Context
//! Boundary conditions play a crucial role in solving partial differential equations (PDEs)
//! in CFD. This module ensures compatibility with Hydra's unstructured grid and time-stepping framework.

pub mod bc_handler;
pub mod dirichlet;
pub mod neumann;
pub mod robin;
pub mod cauchy;
pub mod mixed;

pub mod solid_wall;
pub mod inlet_outlet;
pub mod injection;
pub mod symmetry;
pub mod far_field;
pub mod periodic;

use log::{error, info, warn};
use thiserror::Error;

/// Custom error type for boundary condition operations.
#[derive(Debug, Error)]
pub enum BoundaryError {
    #[error("Failed to set boundary condition for entity {0}: Entity not found.")]
    EntityNotFound(String),

    #[error("Invalid boundary condition type for entity {0}.")]
    InvalidBoundaryType(String),

    #[error("Conflicting boundary conditions applied to entity {0}.")]
    ConflictingBoundary(String),

    #[error("Unknown error encountered: {0}")]
    Unknown(String),

    #[error("Invalid index: {0}")]
    InvalidIndex(String),

    #[error("Invalid boundary condition: {0}")]
    InvalidCondition(String),
}

/// Logs an error message.
pub fn log_boundary_error(err: &BoundaryError) {
    error!("Boundary Error: {}", err);
}

/// Logs a warning for potential issues in boundary condition assignments.
pub fn log_boundary_warning(message: &str) {
    warn!("Boundary Warning: {}", message);
}

/// Logs informational messages for debugging and tracking boundary applications.
pub fn log_boundary_info(message: &str) {
    info!("Boundary Info: {}", message);
}
