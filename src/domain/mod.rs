//! This module defines the core components of the computational domain, including:

pub mod dm_field;
pub mod dm_mesh;
pub mod dm_point;
pub mod dm_section;

pub use dm_field::Field;
pub use dm_mesh::Mesh;
pub use dm_point::{DPoint, DPointType};
pub use dm_section::Section;
