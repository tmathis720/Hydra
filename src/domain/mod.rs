pub mod mesh_entity;
pub mod sieve;
pub mod section;
pub mod overlap;
pub mod stratify;
pub mod entity_fill;
pub mod mesh;

/// Re-exports key components from the `mesh_entity`, `sieve`, and `section` modules.  
/// 
/// This allows the user to access the `MeshEntity`, `Arrow`, `Sieve`, and `Section`  
/// structs directly when importing this module.  
///
/// Example usage:
///    ```rust
///    use hydra::domain::{MeshEntity, Arrow, Sieve, Section};  
///    let entity = MeshEntity::Vertex(1);  
///    let sieve = Sieve::new();  
///    let section: Section<f64> = Section::new();  
///    ```
/// 
pub use mesh_entity::{MeshEntity, Arrow};
pub use sieve::Sieve;
pub use section::Section;
