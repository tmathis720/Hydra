//! Module for gradient calculation in finite element and finite volume methods.
//!
//! This module provides a flexible framework for computing gradients using
//! different numerical methods. It defines the `Gradient` struct, which serves
//! as the main interface for gradient computation, and supports multiple
//! gradient calculation methods via the `GradientCalculationMethod` enum and
//! `GradientMethod` trait.

use crate::boundary::bc_handler::BoundaryConditionHandler;
use crate::domain::section::{scalar::Scalar, vector::Vector3};
use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::geometry::Geometry;
use thiserror::Error;

pub mod gradient_calc;
pub mod least_squares;
pub mod tests;

use gradient_calc::FiniteVolumeGradient;
use least_squares::LeastSquaresGradient;

/// Enum representing the available gradient calculation methods.
pub enum GradientCalculationMethod {
    FiniteVolume,
    LeastSquares,
    // Additional methods can be added here as needed
}

impl GradientCalculationMethod {
    /// Factory function to create a specific gradient calculation method based on the enum variant.
    pub fn create_method(&self) -> Box<dyn GradientMethod> {
        match self {
            GradientCalculationMethod::FiniteVolume => Box::new(FiniteVolumeGradient {}),
            GradientCalculationMethod::LeastSquares => Box::new(LeastSquaresGradient {}),
            // Extend here with other methods as needed
        }
    }
}

/// Custom error type for gradient computation errors.
#[derive(Debug, Error)]
pub enum GradientError {
    #[error("Cell {0:?} is missing from the mesh or invalid.")]
    InvalidCell(MeshEntity),
    #[error("Gradient calculation for cell {0:?} failed with reason: {1}")]
    CalculationError(MeshEntity, String),
    #[error("Unknown error during gradient computation.")]
    Unknown,
}

/// Trait defining the interface for gradient calculation methods.
///
/// Each gradient calculation method must implement this trait, which includes
/// the `calculate_gradient` function for computing the gradient at a given cell.
pub trait GradientMethod {
    /// Computes the gradient for a given cell.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure containing cells and faces.
    /// - `boundary_handler`: Reference to the boundary condition handler.
    /// - `geometry`: Geometry utilities for computing areas, volumes, etc.
    /// - `field`: Scalar field values for each cell.
    /// - `cell`: The current cell for which the gradient is computed.
    /// - `time`: Current simulation time.
    ///
    /// # Returns
    /// - `Ok([f64; 3])`: Computed gradient vector.
    /// - `Err(Box<dyn Error>)`: If any error occurs during computation.
    fn calculate_gradient(
        &self,
        mesh: &Mesh,
        boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        field: &Section<Scalar>,
        cell: &MeshEntity,
        time: f64,
    ) -> Result<[f64; 3], GradientError>;
}

/// Gradient calculator that accepts a gradient method for flexible computation.
///
/// This struct serves as the main interface for computing gradients across the mesh.
/// It delegates the actual gradient computation to the specified `GradientMethod`.
pub struct Gradient<'a> {
    mesh: &'a Mesh,
    boundary_handler: &'a BoundaryConditionHandler,
    geometry: Geometry,
    method: Box<dyn GradientMethod>,
}

impl<'a> Gradient<'a> {
    /// Constructs a new `Gradient` calculator with the specified calculation method.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure.
    /// - `boundary_handler`: Reference to the boundary condition handler.
    /// - `method`: The gradient calculation method to use.
    pub fn new(
        mesh: &'a Mesh,
        boundary_handler: &'a BoundaryConditionHandler,
        method: GradientCalculationMethod,
    ) -> Self {
        Self {
            mesh,
            boundary_handler,
            geometry: Geometry::new(),
            method: method.create_method(),
        }
    }

    /// Computes the gradient of a scalar field across each cell in the mesh.
    ///
    /// # Parameters
    /// - `field`: Scalar field values for each cell.
    /// - `gradient`: Mutable section to store the computed gradient vectors.
    /// - `time`: Current simulation time.
    ///
    /// # Returns
    /// - `Ok(())`: If gradients are successfully computed for all cells.
    /// - `Err(Box<dyn Error>)`: If any error occurs during computation.
    pub fn compute_gradient(
        &mut self,  // Changed to mutable reference
        field: &Section<Scalar>,
        gradient: &mut Section<Vector3>,
        time: f64,
    ) -> Result<(), GradientError> {
        for cell in self.mesh.get_cells() {
            // Check if cell is valid in the mesh
            if !self.mesh.entity_exists(&cell) {
                return Err(GradientError::InvalidCell(cell.clone()));
            }

            // Attempt to compute the gradient
            let grad_phi = self
                .method
                .calculate_gradient(
                    self.mesh,
                    self.boundary_handler,
                    &mut self.geometry,
                    field,
                    &cell,
                    time,
                )
                .map_err(|e| GradientError::CalculationError(cell.clone(), e.to_string()))?;

            // Store the computed gradient
            gradient.set_data(cell, Vector3(grad_phi));
        }
        Ok(())
    }
}
