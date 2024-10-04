// src/linalg/matrix/mod.rs

pub mod traits;
pub mod mat_impl;

pub use traits::Matrix;

#[cfg(test)]
mod tests;