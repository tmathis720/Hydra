// src/boundary/mod.rs
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
