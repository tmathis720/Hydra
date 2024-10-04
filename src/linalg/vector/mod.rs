// src/vector/mod.rs

pub mod traits;
pub mod vec_impl;
pub mod mat_impl;

pub use traits::Vector;

#[cfg(test)]
mod tests;
