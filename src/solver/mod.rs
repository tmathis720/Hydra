//! Main module for the solver interface in Hydra.
//!
//! This module houses the Krylov solvers and preconditioners,
//! facilitating flexible solver selection.
//! 
pub mod ksp;
pub mod cg;
pub mod preconditioner;
pub mod gmres;
pub mod piso;

pub use ksp::KSP;
pub use cg::ConjugateGradient;
pub use gmres::GMRES;

#[cfg(test)]
mod tests;