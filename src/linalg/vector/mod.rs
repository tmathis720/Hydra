// src/vector/mod.rs

pub mod traits;
pub mod vec_impl;
pub mod mat_impl;
pub mod vector_builder;

pub use traits::Vector;

#[cfg(test)]
mod tests;
