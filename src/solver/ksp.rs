//! Defines the Krylov Subspace Method (KSP) trait for solver implementation in Hydra.
//!
//! This trait standardizes methods for solving linear systems across different Krylov methods.

use crate::linalg::{Matrix, Vector};

#[derive(Debug)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// KSP trait for Krylov solvers, encompassing solvers like CG and GMRES.
pub trait KSP {
    fn solve(
        &mut self, 
        a: &dyn Matrix<Scalar = f64>, 
        b: &dyn Vector<Scalar = f64>, 
        x: &mut dyn Vector<Scalar = f64>
    ) -> SolverResult;
}

