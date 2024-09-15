//! This module defines the core components of the computational domain, including:
//! - `Element`: Represents a finite element in the simulation domain.
//! - `Node`: Represents a point in space, typically shared between elements.
//! - `Mesh`: The collection of elements and nodes, along with the relationships between them.
//! - `Face`: Represents the boundary between elements.
//! - `Neighbor`: Handles neighboring relationships between elements.


// Core domain components
pub mod element;
pub mod node;

// Mesh-related components
pub mod mesh;
pub mod neighbor;
pub mod face;

// Flow field
pub mod flow_field;
pub use flow_field::FlowField;

// Re-export core domain components
pub use element::Element;
pub use node::Node;

// Re-export mesh-related components
pub use mesh::Mesh;
pub use neighbor::Neighbor;
pub use face::Face;
pub use mesh::FaceElementRelation;
