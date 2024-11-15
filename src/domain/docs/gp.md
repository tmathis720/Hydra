Generate a detailed outline for a users guide for the `Domain` module for Hydra. The outline should provide high level details which can later be fleshed out. I am going to provide the code for all of the parts of the `Domain` module below, and you can analyze and build the detailed outline based on this version of the source code. Retain the details of your analysis for later as we will be going through the outline in detail throughout this conversation and refining it.

Here is the source tree for the `Domain` module:
```bash
C:.
│   entity_fill.rs
│   mesh_entity.rs
│   mod.rs
│   overlap.rs
│   section.rs
│   sieve.rs
│   stratify.rs
│
└───mesh
        boundary.rs
        boundary_validation.rs
        entities.rs
        geometry.rs
        geometry_validation.rs
        hierarchical.rs
        mod.rs
        reordering.rs
        tests.rs
        topology.rs
```

I will provide the source code in the order provided above, but the organization of the detailed outline should follow logically from overall structure of the source code firstly.

Here is `src/domain/entity_fill.rs` :

```rust
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use dashmap::DashMap;

impl Sieve {
    /// Infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.  
    /// 
    /// For 2D meshes, this method generates edges by connecting vertices of a cell.  
    /// These edges are then associated with the corresponding vertices in the sieve.  
    ///
    /// Example usage:
    /// 
    ///    sieve.fill_missing_entities();  
    ///
    pub fn fill_missing_entities(&self) {
        // Use DashMap instead of FxHashSet for concurrent access.
        let edge_set: DashMap<(MeshEntity, MeshEntity), ()> = DashMap::new();

        // Loop through each cell and infer its edges (for 2D meshes)
        self.adjacency.iter().for_each(|entry| {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                // Connect each vertex with its neighboring vertex to form edges.
                for i in 0..vertices.len() {
                    let v1 = vertices[i].clone();
                    let v2 = vertices[(i + 1) % vertices.len()].clone();
                    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    edge_set.insert(edge, ());
                }
            }
        });

        // Add the deduced edges to the sieve.
        let edge_count = self.adjacency.len();
        edge_set.into_iter().enumerate().for_each(|(index, ((v1, v2), _))| {
            // Generate a unique ID for the new edge.
            let edge = MeshEntity::Edge(edge_count + index);
            self.add_arrow(v1, edge.clone());
            self.add_arrow(v2, edge);
        });
    }
}
```

---

`src/domain/mesh_entity.rs`

```rust
// src/domain/mesh_entity.rs

/// Represents an entity in a mesh, such as a vertex, edge, face, or cell, 
/// using a unique identifier for each entity type.  
/// 
/// The `MeshEntity` enum defines four types of mesh entities:
/// - Vertex: Represents a point in the mesh.
/// - Edge: Represents a connection between two vertices.
/// - Face: Represents a polygonal area bounded by edges.
/// - Cell: Represents a volumetric region of the mesh.
///
/// Example usage:
///
///    let vertex = MeshEntity::Vertex(1);  
///    let edge = MeshEntity::Edge(2);  
///    assert_eq!(vertex.get_id(), 1);  
///    assert_eq!(vertex..get_entity_type(), "Vertex");  
///    assert_eq!(edge.get_id(), 2);  
/// 
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
pub enum MeshEntity {
    Vertex(usize),  // Vertex id 
    Edge(usize),    // Edge id 
    Face(usize),    // Face id 
    Cell(usize),    // Cell id 
}

impl MeshEntity {
    /// Returns the unique identifier associated with the `MeshEntity`.  
    ///
    /// This function matches the enum variant and returns the id for that 
    /// particular entity (e.g., for a `Vertex`, it will return the vertex id).  
    ///
    /// Example usage:
    /// 
    ///    let vertex = MeshEntity::Vertex(3);  
    ///    assert_eq!(vertex.get_id(), 3);  
    ///
    pub fn get_id(&self) -> usize {
        match *self {
            MeshEntity::Vertex(id) => id,
            MeshEntity::Edge(id) => id,
            MeshEntity::Face(id) => id,
            MeshEntity::Cell(id) => id,
        }
    }

    /// Returns the type of the `MeshEntity` as a string, indicating whether  
    /// the entity is a Vertex, Edge, Face, or Cell.  
    ///
    /// Example usage:
    /// 
    ///    let face = MeshEntity::Face(1);  
    ///    assert_eq!(face..get_entity_type(), "Face");  
    ///
    pub fn get_entity_type(&self) -> &str {
        match *self {
            MeshEntity::Vertex(_) => "Vertex",
            MeshEntity::Edge(_) => "Edge",
            MeshEntity::Face(_) => "Face",
            MeshEntity::Cell(_) => "Cell",
        }
    }

    /// Creates a new `MeshEntity` with a specified id for each variant type.
    /// This is a pseudo-set function since enums require construction with
    /// new data to update the id.
    pub fn with_id(&self, new_id: usize) -> Self {
        match *self {
            MeshEntity::Vertex(_) => MeshEntity::Vertex(new_id),
            MeshEntity::Edge(_) => MeshEntity::Edge(new_id),
            MeshEntity::Face(_) => MeshEntity::Face(new_id),
            MeshEntity::Cell(_) => MeshEntity::Cell(new_id),
        }
    }
}

/// A struct representing a directed relationship between two mesh entities,  
/// known as an `Arrow`. It holds the "from" and "to" entities, representing  
/// a connection from one entity to another.  
///
/// Example usage:
/// 
///    let from = MeshEntity::Vertex(1);  
///    let to = MeshEntity::Edge(2);  
///    let arrow = Arrow::new(from, to);  
///    let (start, end) = arrow.get_relation();  
///    assert_eq!(*start, MeshEntity::Vertex(1));  
///    assert_eq!(*end, MeshEntity::Edge(2));  
/// 
pub struct Arrow {
    pub from: MeshEntity,  // The starting entity of the relation 
    pub to: MeshEntity,    // The ending entity of the relation 
}

impl Arrow {
    /// Creates a new `Arrow` between two mesh entities.  
    ///
    /// Example usage:
    /// 
    ///    let from = MeshEntity::Cell(1);  
    ///    let to = MeshEntity::Face(3);  
    ///    let arrow = Arrow::new(from, to);  
    ///
    pub fn new(from: MeshEntity, to: MeshEntity) -> Self {
        Arrow { from, to }
    }

    /// Converts a generic entity type that implements `Into<MeshEntity>` into  
    /// a `MeshEntity`.  
    ///
    /// Example usage:
    /// 
    ///    let vertex = MeshEntity::Vertex(5);  
    ///    let entity = Arrow::add_entity(vertex);  
    ///    assert_eq!(entity.get_id(), 5);  
    ///
    pub fn add_entity<T: Into<MeshEntity>>(entity: T) -> MeshEntity {
        entity.into()
    }

    /// Returns a tuple reference of the "from" and "to" entities of the `Arrow`.  
    ///
    /// Example usage:
    /// 
    ///    let from = MeshEntity::Edge(1);  
    ///    let to = MeshEntity::Face(2);  
    ///    let arrow = Arrow::new(from, to);  
    ///    let (start, end) = arrow.get_relation();  
    ///    assert_eq!(*start, MeshEntity::Edge(1));  
    ///    assert_eq!(*end, MeshEntity::Face(2));  
    ///
    pub fn get_relation(&self) -> (&MeshEntity, &MeshEntity) {
        (&self.from, &self.to)
    }

    /// Sets the `from` entity in the `Arrow`.
    pub fn set_from(&mut self, from: MeshEntity) {
        self.from = from;
    }

    /// Sets the `to` entity in the `Arrow`.
    pub fn set_to(&mut self, to: MeshEntity) {
        self.to = to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test that verifies the id and type of a `MeshEntity` are correctly returned.  
    fn test_entity_id_and_type() {
        let vertex = MeshEntity::Vertex(1);
        assert_eq!(vertex.get_id(), 1);
        assert_eq!(vertex.get_entity_type(), "Vertex");
    }

    #[test]
    /// Test that verifies the creation of an `Arrow` and the correctness of  
    /// the `get_relation` function.  
    fn test_arrow_creation_and_relation() {
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(2);
        let arrow = Arrow::new(vertex, edge);
        let (from, to) = arrow.get_relation();
        assert_eq!(*from, MeshEntity::Vertex(1));
        assert_eq!(*to, MeshEntity::Edge(2));
    }

    #[test]
    /// Test that verifies the addition of an entity using the `add_entity` function.  
    fn test_add_entity() {
        let vertex = MeshEntity::Vertex(5);
        let added_entity = Arrow::add_entity(vertex);

        assert_eq!(added_entity.get_id(), 5);
        assert_eq!(added_entity.get_entity_type(), "Vertex");
    }

    #[test]
    fn test_with_id() {
        let edge = MeshEntity::Edge(5);
        let new_edge = edge.with_id(10);
        assert_eq!(new_edge.get_id(), 10);
        assert_eq!(new_edge.get_entity_type(), "Edge");
    }
}
```

---

`src/domain/mod.rs`

```rust
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
```

---

`src/domain/overlap.rs`

```rust
use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;

/// The `Overlap` struct manages two sets of `MeshEntity` elements:  
/// - `local_entities`: Entities that are local to the current partition.
/// - `ghost_entities`: Entities that are shared with other partitions.
pub struct Overlap {
    /// A thread-safe set of local entities.  
    pub local_entities: Arc<DashMap<MeshEntity, ()>>,
    /// A thread-safe set of ghost entities.  
    pub ghost_entities: Arc<DashMap<MeshEntity, ()>>,
}

impl Overlap {
    /// Creates a new `Overlap` with empty sets for local and ghost entities.
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(DashMap::new()),
            ghost_entities: Arc::new(DashMap::new()),
        }
    }

    /// Adds a `MeshEntity` to the set of local entities.
    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.insert(entity, ());
    }

    /// Adds a `MeshEntity` to the set of ghost entities.
    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.insert(entity, ());
    }

    /// Checks if a `MeshEntity` is a local entity.
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        self.local_entities.contains_key(entity)
    }

    /// Checks if a `MeshEntity` is a ghost entity.
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        self.ghost_entities.contains_key(entity)
    }

    /// Retrieves a clone of all local entities.
    pub fn local_entities(&self) -> Vec<MeshEntity> {
        self.local_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves a clone of all ghost entities.
    pub fn ghost_entities(&self) -> Vec<MeshEntity> {
        self.ghost_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Merges another `Overlap` instance into this one, combining local  
    /// and ghost entities from both overlaps.
    pub fn merge(&self, other: &Overlap) {
        other.local_entities.iter().for_each(|entry| {
            self.local_entities.insert(entry.key().clone(), ());
        });

        other.ghost_entities.iter().for_each(|entry| {
            self.ghost_entities.insert(entry.key().clone(), ());
        });
    }
}

/// The `Delta` struct manages transformation data for `MeshEntity` elements  
/// in overlapping regions. It is used to store and apply data transformations  
/// across entities in distributed environments.
pub struct Delta<T> {
    /// A thread-safe map storing transformation data associated with `MeshEntity` objects.  
    pub data: Arc<DashMap<MeshEntity, T>>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty `Delta`.
    pub fn new() -> Self {
        Delta {
            data: Arc::new(DashMap::new()),
        }
    }

    /// Sets the transformation data for a specific `MeshEntity`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves the transformation data associated with a specific `MeshEntity`.
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|entry| entry.clone())
    }

    /// Removes the transformation data associated with a specific `MeshEntity`.
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        self.data.remove(entity).map(|(_, value)| value)
    }

    /// Checks if there is transformation data for a specific `MeshEntity`.
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        self.data.contains_key(entity)
    }

    /// Applies a function to all entities in the delta.
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        self.data.iter().for_each(|entry| func(entry.key(), entry.value()));
    }

    /// Merges another `Delta` instance into this one, combining data from both deltas.
    pub fn merge(&self, other: &Delta<T>)
    where
        T: Clone,
    {
        other.data.iter().for_each(|entry| {
            self.data.insert(entry.key().clone(), entry.value().clone());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_overlap_local_and_ghost_entities() {
        let overlap = Overlap::new();
        let vertex_local = MeshEntity::Vertex(1);
        let vertex_ghost = MeshEntity::Vertex(2);
        overlap.add_local_entity(vertex_local);
        overlap.add_ghost_entity(vertex_ghost);
        assert!(overlap.is_local(&vertex_local));
        assert!(overlap.is_ghost(&vertex_ghost));
    }

    #[test]
    fn test_overlap_merge() {
        let overlap1 = Overlap::new();
        let overlap2 = Overlap::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);

        overlap1.add_local_entity(vertex1);
        overlap1.add_ghost_entity(vertex2);

        overlap2.add_local_entity(vertex3);

        overlap1.merge(&overlap2);

        assert!(overlap1.is_local(&vertex1));
        assert!(overlap1.is_ghost(&vertex2));
        assert!(overlap1.is_local(&vertex3));
        assert_eq!(overlap1.local_entities().len(), 2);
    }

    #[test]
    fn test_delta_set_and_get_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 42);

        assert_eq!(delta.get_data(&vertex), Some(42));
        assert!(delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_remove_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 100);
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_merge() {
        let delta1 = Delta::new();
        let delta2 = Delta::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        delta1.set_data(vertex1, 10);
        delta2.set_data(vertex2, 20);

        delta1.merge(&delta2);

        assert_eq!(delta1.get_data(&vertex1), Some(10));
        assert_eq!(delta1.get_data(&vertex2), Some(20));
    }
}
```

---

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.  
/// It provides methods for setting, updating, and retrieving data, and supports  
/// parallel updates for performance improvements.  
///
/// Example usage:
///
///    let section = Section::new();  
///    let vertex = MeshEntity::Vertex(1);  
///    section.set_data(vertex, 42);  
///    assert_eq!(section.restrict(&vertex), Some(42));  
/// 
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.  
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T> {
    /// Creates a new `Section` with an empty data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Sets the data associated with a given `MeshEntity`.  
    /// This method inserts the `entity` and its corresponding `value` into the data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    section.set_data(MeshEntity::Vertex(1), 10);  
    ///
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Restricts the data for a given `MeshEntity` by returning an immutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 42);  
    ///    assert_eq!(section.restrict(&vertex), Some(42));  
    ///
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Applies the given function in parallel to update all data values in the section.
    ///
    /// Example usage:
    ///
    ///    section.parallel_update(|v| *v += 1);  
    ///
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
        T: Send + Sync,
    {
        // Clone the keys to ensure safe access to each mutable entry in parallel.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Apply the update function to each entry in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Restricts the data for a given `MeshEntity` by returning a mutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 5);  
    ///    let mut value = section.restrict_mut(&vertex).unwrap();  
    ///    value = 10;  
    ///    section.set_data(vertex, value);  
    ///
    pub fn restrict_data_mut(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates the data for a specific `MeshEntity` by replacing the existing value  
    /// with the new value.  
    ///
    /// Example usage:
    ///
    ///    section.update_data(&MeshEntity::Vertex(1), 15);  
    ///
    pub fn update_data(&self, entity: &MeshEntity, new_value: T) {
        self.data.insert(*entity, new_value);
    }

    /// Clears all data from the section, removing all entity associations.  
    ///
    /// Example usage:
    ///
    ///    section.clear();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Retrieves all `MeshEntity` objects associated with the section.  
    ///
    /// Returns a vector containing all mesh entities currently stored in the section.  
    ///
    /// Example usage:
    ///
    ///    let entities = section.entities();  
    ///
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all data stored in the section as immutable copies.  
    ///
    /// Returns a vector of data values.  
    ///
    /// Example usage:
    ///
    ///    let all_data = section.all_data();  
    ///
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Retrieves all data stored in the section with mutable access.  
    ///
    /// Returns a vector of data values that can be modified.  
    ///
    /// Example usage:
    ///
    ///    let all_data_mut = section.all_data_mut();  
    ///
    pub fn all_data_mut(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter_mut().map(|entry| entry.value().clone()).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies setting and restricting data for a `MeshEntity`  
    /// works as expected.  
    fn test_set_and_restrict_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        section.set_data(vertex, 42);
        assert_eq!(section.restrict(&vertex), Some(42));
    }

    #[test]
    /// Test that verifies updating the data for an entity works as expected,  
    /// including updating a non-existent entity.  
    fn test_update_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 10);
        assert_eq!(section.restrict(&vertex), Some(10));

        // Update the data
        section.update_data(&vertex, 15);
        assert_eq!(section.restrict(&vertex), Some(15));

        // Try updating data for a non-existent entity (should insert it)
        let non_existent_entity = MeshEntity::Vertex(2);
        section.update_data(&non_existent_entity, 30);
        assert_eq!(section.restrict(&non_existent_entity), Some(30));
    }

    #[test]
    /// Test that verifies the mutable restriction of data for a `MeshEntity`  
    /// works as expected.  
    fn test_restrict_mut() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 5);
        if let Some(mut value) = section.restrict_data_mut(&vertex) {
            value = 50;
            section.set_data(vertex, value);
        }
        assert_eq!(section.restrict(&vertex), Some(50));
    }

    #[test]
    /// Test that verifies retrieving all entities associated with the section  
    /// works as expected.  
    fn test_get_all_entities() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let entities = section.entities();
        assert!(entities.contains(&vertex));
        assert!(entities.contains(&edge));
        assert_eq!(entities.len(), 2);
    }

    #[test]
    /// Test that verifies retrieving all data stored in the section works  
    /// as expected.  
    fn test_get_all_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let all_data = section.all_data();
        assert_eq!(all_data.len(), 2);
        assert!(all_data.contains(&10));
        assert!(all_data.contains(&20));
    }

    #[test]
    /// Test that verifies parallel updates to data in the section are  
    /// applied correctly using Rayon for concurrency.  
    fn test_parallel_update() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        section.set_data(vertex, 10);
        section.parallel_update(|v| *v += 5);
        assert_eq!(section.restrict(&vertex), Some(15));
    }
}
```

---

`src/domain/sieve.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity`  
/// elements, organized in an adjacency map.
///
/// The adjacency map tracks directed relations between entities in the mesh.  
/// It supports operations such as adding relationships, querying direct  
/// relations (cones), and computing closure and star sets for entities.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// A thread-safe adjacency map where each key is a `MeshEntity`,  
    /// and the value is a set of `MeshEntity` objects related to the key.  
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new empty `Sieve` instance with an empty adjacency map.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.  
    /// The relationship is stored in the adjacency map from the `from` entity  
    /// to the `to` entity.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).  
    /// This operation is referred to as retrieving the cone of the entity.  
    /// Returns `None` if there are no related entities.
    pub fn cone(&self, point: &MeshEntity) -> Option<Vec<MeshEntity>> {
        self.adjacency.get(point).map(|cone| {
            cone.iter().map(|entry| entry.key().clone()).collect()
        })
    }

    /// Computes the closure of a given `MeshEntity`.  
    /// The closure includes the entity itself and all entities it covers (cones) recursively.
    pub fn closure(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());

        while !stack.is_empty() {
            let keys: Vec<MeshEntity> = stack.iter().map(|entry| entry.key().clone()).collect();
            for p in keys {
                if result.insert(p.clone(), ()).is_none() {
                    if let Some(cones) = self.cone(&p) {
                        for q in cones {
                            stack.insert(q, ());
                        }
                    }
                }
                stack.remove(&p);
            }
        }
        result
    }

    /// Computes the star of a given `MeshEntity`.  
    /// The star includes the entity itself and all entities that directly cover it (supports).
    pub fn star(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        result.insert(point.clone(), ());
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        result
    }

    /// Retrieves all entities that support the given entity (`point`).  
    /// These are the entities that have an arrow pointing to `point`.
    pub fn support(&self, point: &MeshEntity) -> Vec<MeshEntity> {
        let mut supports = Vec::new();
        self.adjacency.iter().for_each(|entry| {
            let from = entry.key();
            if entry.value().contains_key(point) {
                supports.push(from.clone());
            }
        });
        supports
    }

    /// Computes the meet operation for two entities, `p` and `q`.  
    /// This is the intersection of their closures.
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        let result = DashMap::new();

        closure_p.iter().for_each(|entry| {
            let key = entry.key();
            if closure_q.contains_key(key) {
                result.insert(key.clone(), ());
            }
        });

        result
    }

    /// Computes the join operation for two entities, `p` and `q`.  
    /// This is the union of their stars.
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let star_p = self.star(p);
        let star_q = self.star(q);
        let result = DashMap::new();

        star_p.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });
        star_q.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });

        result
    }

    /// Applies a given function in parallel to all adjacency map entries.  
    /// This function is executed concurrently over each entity and its  
    /// corresponding set of related entities.
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, Vec<MeshEntity>)) + Sync + Send,
    {
        // Collect entries from DashMap to avoid borrow conflicts
        let entries: Vec<_> = self.adjacency.iter().map(|entry| {
            let key = entry.key().clone();
            let values: Vec<MeshEntity> = entry.value().iter().map(|e| e.key().clone()).collect();
            (key, values)
        }).collect();

        // Execute in parallel over collected entries
        entries.par_iter().for_each(|entry| {
            func((&entry.0, entry.1.clone()));
        });
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies adding an arrow between two entities and querying  
    /// the cone of an entity works as expected.
    fn test_add_arrow_and_cone() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        sieve.add_arrow(vertex, edge);
        let cone_result = sieve.cone(&vertex).unwrap();
        assert!(cone_result.contains(&edge));
    }

    #[test]
    /// Test that verifies the closure of a vertex correctly includes  
    /// all transitive relationships and the entity itself.
    fn test_closure() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);
        sieve.add_arrow(vertex, edge);
        sieve.add_arrow(edge, face);
        let closure_result = sieve.closure(&vertex);
        assert!(closure_result.contains_key(&vertex));
        assert!(closure_result.contains_key(&edge));
        assert!(closure_result.contains_key(&face));
        assert_eq!(closure_result.len(), 3);
    }

    #[test]
    /// Test that verifies the support of an entity includes the  
    /// correct supporting entities.
    fn test_support() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let support_result = sieve.support(&edge);

        assert!(support_result.contains(&vertex));
        assert_eq!(support_result.len(), 1);
    }

    #[test]
    /// Test that verifies the star of an entity includes both the entity itself and  
    /// its immediate supports.
    fn test_star() {
        let sieve = Sieve::new();
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(edge, face);

        let star_result = sieve.star(&face);

        assert!(star_result.contains_key(&face));
        assert!(star_result.contains_key(&edge));
        assert_eq!(star_result.len(), 2);
    }

    #[test]
    /// Test that verifies the meet operation between two entities returns  
    /// the correct intersection of their closures.
    fn test_meet() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex1, edge);
        sieve.add_arrow(vertex2, edge);

        let meet_result = sieve.meet(&vertex1, &vertex2);

        assert!(meet_result.contains_key(&edge));
        assert_eq!(meet_result.len(), 1);
    }

    #[test]
    /// Test that verifies the join operation between two entities returns  
    /// the correct union of their stars.
    fn test_join() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        let join_result = sieve.join(&vertex1, &vertex2);

        assert!(join_result.contains_key(&vertex1), "Join result should contain vertex1");
        assert!(join_result.contains_key(&vertex2), "Join result should contain vertex2");
        assert_eq!(join_result.len(), 2);
    }
}

```

---

`src/domain/stratify.rs`

```rust
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use dashmap::DashMap;

/// Implements a stratification method for the `Sieve` structure.  
/// Stratification organizes the mesh entities into different strata based on  
/// their dimensions:  
/// - Stratum 0: Vertices  
/// - Stratum 1: Edges  
/// - Stratum 2: Faces  
/// - Stratum 3: Cells  
///
/// This method categorizes each `MeshEntity` into its corresponding stratum and  
/// returns a `DashMap` where the keys are the dimension (stratum) and the values  
/// are vectors of mesh entities in that stratum.  
///
/// Example usage:
/// 
///    let sieve = Sieve::new();  
///    sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));  
///    let strata = sieve.stratify();  
///    assert_eq!(strata.get(&0).unwrap().len(), 1);  // Stratum for vertices  
/// 
impl Sieve {
    /// Organizes the mesh entities in the sieve into strata based on their dimension.  
    ///
    /// The method creates a map where each key is the dimension (0 for vertices,  
    /// 1 for edges, 2 for faces, 3 for cells), and the value is a vector of mesh  
    /// entities in that dimension.
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));  
    ///    let strata = sieve.stratify();  
    ///
    pub fn stratify(&self) -> DashMap<usize, Vec<MeshEntity>> {
        let strata: DashMap<usize, Vec<MeshEntity>> = DashMap::new();

        // Iterate over the adjacency map to classify entities by their dimension.
        self.adjacency.iter().for_each(|entry| {
            let entity = entry.key();
            // Determine the dimension of the current entity.
            let dimension = match entity {
                MeshEntity::Vertex(_) => 0,  // Stratum 0 for vertices
                MeshEntity::Edge(_) => 1,    // Stratum 1 for edges
                MeshEntity::Face(_) => 2,    // Stratum 2 for faces
                MeshEntity::Cell(_) => 3,    // Stratum 3 for cells
            };
            
            // Insert entity into the appropriate stratum in a thread-safe manner.
            strata.entry(dimension).or_insert_with(Vec::new).push(entity.clone());
        });

        strata
    }
}
```

---

`src/domain/mesh/boundary.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};

impl Mesh {
    /// Synchronizes the boundary data by first sending the local boundary data  
    /// and then receiving any updated boundary data from other sources.  
    ///
    /// This function ensures that boundary data, such as vertex coordinates,  
    /// is consistent across all mesh partitions.  
    ///
    /// Example usage:
    /// 
    ///    mesh.sync_boundary_data();  
    ///
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data();
        self.receive_boundary_data();
    }

    /// Sets the communication channels for boundary data transmission.  
    ///
    /// The sender channel is used to transmit the local boundary data, and  
    /// the receiver channel is used to receive boundary data from other  
    /// partitions or sources.  
    ///
    /// Example usage:
    /// 
    ///    mesh.set_boundary_channels(sender, receiver);  
    ///
    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender);
        self.boundary_data_receiver = Some(receiver);
    }

    /// Receives boundary data from the communication channel and updates the mesh.  
    ///
    /// This method listens for incoming boundary data (such as vertex coordinates)  
    /// from the receiver channel and updates the local mesh entities and coordinates.  
    ///
    /// Example usage:
    /// 
    ///    mesh.receive_boundary_data();  
    ///
    pub fn receive_boundary_data(&mut self) {
        if let Some(ref receiver) = self.boundary_data_receiver {
            if let Ok(boundary_data) = receiver.recv() {
                let mut entities = self.entities.write().unwrap();
                for (entity, coords) in boundary_data {
                    // Update vertex coordinates if the entity is a vertex.
                    if let MeshEntity::Vertex(id) = entity {
                        self.vertex_coordinates.insert(id, coords);
                    }
                    entities.insert(entity);
                }
            }
        }
    }

    /// Sends the local boundary data (such as vertex coordinates) through  
    /// the communication channel to other partitions or sources.  
    ///
    /// This method collects the vertex coordinates for all mesh entities  
    /// and sends them using the sender channel.  
    ///
    /// Example usage:
    /// 
    ///    mesh.send_boundary_data();  
    ///
    pub fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default();
            let entities = self.entities.read().unwrap();
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords);
                    }
                }
            }

            // Send the boundary data through the sender channel.
            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        }
    }
}
```

---

`src/domain/mesh/entities.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

impl Mesh {
    /// Adds a new `MeshEntity` to the mesh.  
    /// The entity will be inserted into the thread-safe `entities` set.  
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    mesh.add_entity(vertex);  
    /// 
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    /// Establishes a relationship (arrow) between two mesh entities.  
    /// This creates an arrow from the `from` entity to the `to` entity  
    /// in the sieve structure.  
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_relationship(vertex, edge);  
    /// 
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Adds an arrow from one mesh entity to another in the sieve structure.  
    /// This method is a simple delegate to the `Sieve`'s `add_arrow` method.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_arrow(vertex, edge);  
    /// 
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Sets the 3D coordinates for a vertex and adds the vertex entity  
    /// to the mesh if it's not already present.  
    /// 
    /// This method inserts the vertex's coordinates into the  
    /// `vertex_coordinates` map and adds the vertex to the `entities` set.
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    mesh.set_vertex_coordinates(1, [1.0, 2.0, 3.0]);  
    ///    assert_eq!(mesh.get_vertex_coordinates(1), Some([1.0, 2.0, 3.0]));  
    ///
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.  
    ///
    /// Returns `None` if the vertex does not exist in the `vertex_coordinates` map.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let coords = mesh.get_vertex_coordinates(1);  
    ///    assert!(coords.is_none());  
    ///
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    /// Counts the number of entities of a specified type (e.g., Vertex, Edge, Face, Cell)  
    /// within the mesh.  
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let count = mesh.count_entities(&MeshEntity::Vertex(1));  
    ///    assert_eq!(count, 0);  
    ///
    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count()
    }

    /// Applies a given function to each entity in the mesh in parallel.  
    ///
    /// The function `func` is applied to all mesh entities concurrently  
    /// using Rayon’s parallel iterator.
    ///
    /// Example usage:
    /// 
    ///    mesh.par_for_each_entity(|entity| {  
    ///        println!("{:?}", entity);  
    ///    });  
    ///
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Retrieves all the `Cell` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as cells.
    ///
    /// Example usage:
    /// 
    ///    let cells = mesh.get_cells();  
    ///    assert!(cells.is_empty());  
    ///
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all the `Face` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as faces.
    ///
    /// Example usage:
    /// 
    ///    let faces = mesh.get_faces();  
    ///    assert!(faces.is_empty());  
    ///
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Retrieves the vertices of the given face.
    pub fn get_vertices_of_face(&self, face: &MeshEntity) -> Vec<MeshEntity> {
        self.sieve.cone(face).unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .collect()
    }

    /// Computes properties for each entity in the mesh in parallel,  
    /// returning a map of `MeshEntity` to the computed property.  
    ///
    /// The `compute_fn` is a user-provided function that takes a reference  
    /// to a `MeshEntity` and returns a computed value of type `PropertyType`.  
    ///
    /// Example usage:
    /// 
    ///    let properties = mesh.compute_properties(|entity| {  
    ///        entity.get_id()  
    ///    });  
    ///
    pub fn compute_properties<F, PropertyType>(&self, compute_fn: F) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect()
    }

    /// Retrieves the ordered neighboring cells for each cell in the mesh.
    ///
    /// This method is designed for use in flux computations and gradient reconstruction,
    /// and returns the neighboring cells in a predetermined, consistent order.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which neighbors are retrieved.
    ///
    /// # Returns
    /// A vector of neighboring cells ordered for consistency in TVD calculations.
    pub fn get_ordered_neighbors(&self, cell: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = Vec::new();
        if let Some(faces) = self.get_faces_of_cell(cell) {
            for face in faces.iter() {
                let cells_sharing_face = self.get_cells_sharing_face(&face.key());
                for neighbor in cells_sharing_face.iter() {
                    if *neighbor.key() != *cell {
                        neighbors.push(*neighbor.key());
                    }
                }
            }
        }
        neighbors.sort_by(|a, b| a.get_id().cmp(&b.get_id())); // Ensures consistent ordering by ID
        neighbors
    }
}
```

---

`src/domain/mesh/geometry.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;

impl Mesh {
    /// Retrieves all the faces of a given cell, filtering only face entities.
    ///
    /// Returns a set of `MeshEntity` representing the faces of the cell, or
    /// `None` if the cell has no connected faces.
    ///
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<DashMap<MeshEntity, ()>> {
        self.sieve.cone(cell).map(|set| {
            let faces = DashMap::new();
            set.into_iter()
                .filter(|entity| matches!(entity, MeshEntity::Face(_)))
                .for_each(|face| {
                    faces.insert(face, ());
                });
            faces
        })
    }

    /// Retrieves all the cells that share the given face, filtering only cell entities that are present in the mesh.
    ///
    /// Returns a set of `MeshEntity` representing the neighboring cells.
    ///
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let cells = DashMap::new();
        let entities = self.entities.read().unwrap();
        self.sieve
            .support(face)
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
            .for_each(|cell| {
                cells.insert(cell, ());
            });
        cells
    }

    /// Computes the Euclidean distance between two cells based on their centroids.  
    ///
    /// This method calculates the centroids of both cells and then uses the `Geometry`  
    /// module to compute the distance between these centroids.  
    ///
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.  
    ///
    /// This method determines the face shape (triangle or quadrilateral) and  
    /// uses the `Geometry` module to compute the area.  
    ///
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        geometry.compute_face_area(face_id, face_shape, &face_vertices)
    }

    /// Computes the centroid of a cell based on its vertices.  
    ///
    /// This method determines the cell shape and uses the `Geometry` module to compute the centroid.  
    ///
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to the given vertex by shared cells.  
    ///
    /// This method uses the `support` function of the sieve to find cells that  
    /// contain the given vertex and then retrieves all other vertices in those cells.  
    ///
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let neighbors = DashMap::new();
        let connected_cells = self.sieve.support(vertex);

        connected_cells.into_iter().for_each(|cell| {
            if let Some(cell_vertices) = self.sieve.cone(&cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(v.clone(), ());
                    }
                }
            } else {
                panic!("Cell {:?} has no connected vertices", cell);
            }
        });
        neighbors.into_iter().map(|(vertex, _)| vertex).collect()
    }

    /// Returns an iterator over the IDs of all vertices in the mesh.  
    ///
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.  
    ///
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell);
        match cell_vertices.len() {
            4 => Ok(CellShape::Tetrahedron),
            5 => Ok(CellShape::Pyramid),
            6 => Ok(CellShape::Prism),
            8 => Ok(CellShape::Hexahedron),
            _ => Err(format!(
                "Unsupported cell shape with {} vertices. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len()
            )),
        }
    }

    /// Retrieves the vertices of a cell and their coordinates, sorted by vertex ID.
    ///
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
            // Sort the vertices based on their IDs
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Retrieves the vertices of a face and their coordinates, sorted by vertex ID.
    ///
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            // Sort the vertices based on their IDs
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }
}
```

---

`src/domain/mesh/hierarchical.rs`

```rust
use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)  
/// or a branch (refined into smaller child elements).  
/// 
/// In 2D, each branch contains 4 child elements (quadtree), while in 3D, each branch  
/// would contain 8 child elements (octree).
///
/// Example usage:
/// 
///    let mut node = MeshNode::Leaf(10);  
///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
///    assert!(matches!(node, MeshNode::Branch { .. }));  
/// 
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    /// A leaf node representing an unrefined element containing data of type `T`.  
    Leaf(T),
    
    /// A branch node representing a refined element with child elements.  
    /// The branch contains its own data and an array of 4 child nodes (for 2D).  
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>,  // 2D quadtree; change to `[MeshNode<T>; 8]` for 3D.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.  
    ///
    /// The `init_child_data` function is used to generate the data for each child  
    /// element based on the parent node's data.  
    ///
    /// Example usage:
    /// 
    ///    let mut node = MeshNode::Leaf(10);  
    ///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
    ///
    pub fn refine<F>(&mut self, init_child_data: F)
    where
        F: Fn(&T) -> [T; 4],  // Function to generate child data from the parent.
    {
        if let MeshNode::Leaf(data) = self {
            let children = init_child_data(data);
            *self = MeshNode::Branch {
                data: data.clone(),
                children: Box::new([
                    MeshNode::Leaf(children[0].clone()),
                    MeshNode::Leaf(children[1].clone()),
                    MeshNode::Leaf(children[2].clone()),
                    MeshNode::Leaf(children[3].clone()),
                ]),
            };
        }
    }

    /// Coarsens a branch back into a leaf node by collapsing its child elements.  
    ///
    /// This method turns a branch back into a leaf node, retaining the data of the  
    /// parent node but removing its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.coarsen();  
    ///
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Applies constraints at hanging nodes to ensure continuity between the parent  
    /// and its child elements.  
    ///
    /// This function adjusts the degrees of freedom (DOFs) at the parent node by  
    /// averaging the DOFs from its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);  
    ///
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// Returns an iterator over all leaf nodes in the mesh hierarchy.  
    ///
    /// This iterator allows traversal of the entire hierarchical mesh,  
    /// returning only the leaf nodes.  
    ///
    /// Example usage:
    /// 
    ///    let leaves: Vec<_> = node.leaf_iter().collect();  
    ///
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// An iterator for traversing through leaf nodes in the hierarchical mesh.  
/// 
/// This iterator traverses all nodes in the hierarchy but only returns  
/// the leaf nodes.
pub struct LeafIterator<'a, T> {
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    /// Returns the next leaf node in the traversal.  
    /// If the current node is a branch, its children are pushed onto the stack  
    /// for traversal in depth-first order.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    // Push children onto the stack in reverse order to get them in the desired order.
                    for child in children.iter().rev() {
                        self.stack.push(child);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::hierarchical::MeshNode;

    #[test]
    /// Test that verifies refining a leaf node into a branch works as expected.  
    fn test_refine_leaf() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        if let MeshNode::Branch { children, .. } = node {
            assert_eq!(children[0], MeshNode::Leaf(11));
            assert_eq!(children[1], MeshNode::Leaf(12));
            assert_eq!(children[2], MeshNode::Leaf(13));
            assert_eq!(children[3], MeshNode::Leaf(14));
        } else {
            panic!("Node should have been refined to a branch.");
        }
    }

    #[test]
    /// Test that verifies coarsening a branch node back into a leaf works as expected.  
    fn test_coarsen_branch() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
        node.coarsen();

        assert_eq!(node, MeshNode::Leaf(10));
    }

    #[test]
    /// Test that verifies applying hanging node constraints works correctly by  
    /// averaging the degrees of freedom from the child elements to the parent element.  
    fn test_apply_hanging_node_constraints() {
        let node = MeshNode::Branch {
            data: 0,
            children: Box::new([
                MeshNode::Leaf(1),
                MeshNode::Leaf(2),
                MeshNode::Leaf(3),
                MeshNode::Leaf(4),
            ]),
        };

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ];

        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        assert_eq!(parent_dofs, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    /// Test that verifies the leaf iterator correctly traverses all leaf nodes in  
    /// the mesh hierarchy and returns them in the expected order.  
    fn test_leaf_iterator() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        let leaves: Vec<_> = node.leaf_iter().collect();
        assert_eq!(leaves, [&11, &12, &13, &14]);
    }
}
```

---

`src/domain/mesh/mod.rs`

```rust
pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;
pub mod topology;
/* pub mod geometry_validation;
pub mod boundary_validation; */

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};

// Delegate methods to corresponding modules

/// Represents the mesh structure, which is composed of a sieve for entity management,  
/// a set of mesh entities, vertex coordinates, and channels for boundary data.  
/// 
/// The `Mesh` struct is the central component for managing mesh entities and  
/// their relationships. It stores entities such as vertices, edges, faces,  
/// and cells, along with their geometric data and boundary-related information.  
/// 
/// Example usage:
/// 
///    let mesh = Mesh::new();  
///    let entity = MeshEntity::Vertex(1);  
///    mesh.entities.write().unwrap().insert(entity);  
/// 
#[derive(Clone, Debug)]
pub struct Mesh {
    /// The sieve structure used for organizing the mesh entities' relationships.  
    pub sieve: Arc<Sieve>,  
    /// A thread-safe, read-write lock for managing mesh entities.  
    /// This set contains all `MeshEntity` objects in the mesh.  
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,  
    /// A map from vertex indices to their 3D coordinates.  
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,  
    /// An optional channel sender for transmitting boundary data related to mesh entities.  
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// An optional channel receiver for receiving boundary data related to mesh entities.  
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,  
}

impl Mesh {
    /// Creates a new instance of the `Mesh` struct with initialized components.  
    /// 
    /// This method sets up the sieve, entity set, vertex coordinate map,  
    /// and a channel for boundary data communication between mesh components.  
    ///
    /// The `Sender` and `Receiver` are unbounded channels used to pass boundary  
    /// data between mesh modules asynchronously.
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    assert!(mesh.entities.read().unwrap().is_empty());  
    /// 
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }
}

#[cfg(test)]
pub mod tests;
```

---

`src/domain/mesh/reordering.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use rayon::prelude::*;

/// Reorders mesh entities using the Cuthill-McKee algorithm.  
/// This algorithm improves memory locality by reducing the bandwidth of sparse matrices,  
/// which is beneficial for solver optimizations.  
///
/// The algorithm starts from the node with the smallest degree and visits its neighbors  
/// in increasing order of their degree.
///
/// Example usage:
/// 
///    let ordered_entities = cuthill_mckee(&entities, &adjacency);  
///
pub fn cuthill_mckee(
    entities: &[MeshEntity], 
    adjacency: &FxHashMap<MeshEntity, Vec<MeshEntity>>
) -> Vec<MeshEntity> {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::new();

    // Find the starting entity (node) with the smallest degree.
    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);
        visited.insert(*start);
    }

    // Perform the Cuthill-McKee reordering.
    while let Some(entity) = queue.pop_front() {
        ordered.push(entity);
        if let Some(neighbors) = adjacency.get(&entity) {
            let mut sorted_neighbors: Vec<_> = neighbors.iter()
                .filter(|&&n| !visited.contains(&n))
                .cloned()
                .collect();
            sorted_neighbors.sort_by_key(|n| adjacency.get(n).map_or(0, |neighbors| neighbors.len()));
            for neighbor in sorted_neighbors {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    ordered
}

impl Mesh {
    /// Applies a reordering to the mesh entities based on the given new order.  
    ///
    /// This method can be used to reorder entities or update a sparse matrix  
    /// structure based on the new ordering.
    ///
    /// Example usage:
    /// 
    ///    mesh.apply_reordering(&new_order);  
    ///
    pub fn apply_reordering(&mut self, _new_order: &[usize]) {
        // Implement the application of reordering to mesh entities or sparse matrix structure.
    }

    /// Computes the reverse Cuthill-McKee (RCM) ordering starting from a given node.  
    ///
    /// This method performs the RCM algorithm to minimize the bandwidth of sparse matrices  
    /// by reordering mesh entities in reverse order of their Cuthill-McKee ordering.  
    ///
    /// Example usage:
    /// 
    ///    let rcm_order = mesh.rcm_ordering(start_node);  
    ///
    pub fn rcm_ordering(&self, start_node: MeshEntity) -> Vec<MeshEntity> {
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        let mut ordering = Vec::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        // Perform breadth-first traversal and order nodes by degree.
        while let Some(node) = queue.pop_front() {
            ordering.push(node);
            if let Some(neighbors) = self.sieve.cone(&node) {
                let mut sorted_neighbors: Vec<_> = neighbors
                    .into_iter()
                    .filter(|n| !visited.contains(n))
                    .collect();
                sorted_neighbors.sort_by_key(|n| self.sieve.cone(n).map_or(0, |set| set.len()));
                for neighbor in sorted_neighbors {
                    queue.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        // Reverse the ordering to get the RCM order.
        ordering.reverse();
        ordering
    }

    /// Reorders elements in the mesh using Morton order (Z-order curve) for better memory locality.  
    ///
    /// This method applies the Morton order to the given set of 2D elements (with x and y coordinates).  
    /// Morton ordering is a space-filling curve that helps improve memory access patterns  
    /// in 2D meshes or grids.
    ///
    /// Example usage:
    /// 
    ///    mesh.reorder_by_morton_order(&mut elements);  
    ///
    pub fn reorder_by_morton_order(&mut self, elements: &mut [(u32, u32)]) {
        elements.par_sort_by_key(|&(x, y)| Self::morton_order_2d(x, y));
    }

    /// Computes the Morton order (Z-order curve) for a 2D point with coordinates (x, y).  
    ///
    /// This function interleaves the bits of the x and y coordinates to generate  
    /// a single value that represents the Morton order.  
    ///
    /// Example usage:
    /// 
    ///    let morton_order = Mesh::morton_order_2d(10, 20);  
    ///
    pub fn morton_order_2d(x: u32, y: u32) -> u64 {
        // Helper function to interleave the bits of a 32-bit integer.
        fn part1by1(n: u32) -> u64 {
            let mut n = n as u64;
            n = (n | (n << 16)) & 0x0000_0000_ffff_0000;
            n = (n | (n << 8)) & 0x0000_ff00_00ff_0000;
            n = (n | (n << 4)) & 0x00f0_00f0_00f0_00f0;
            n = (n | (n << 2)) & 0x0c30_0c30_0c30_0c30;
            n = (n | (n << 1)) & 0x2222_2222_2222_2222;
            n
        }

        // Interleave the bits of x and y to compute the Morton order.
        part1by1(x) | (part1by1(y) << 1)
    }
}
```

---

`src/domain/mesh/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    /// Tests that boundary data can be sent from one mesh and received by another.  
    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        mesh.send_boundary_data();

        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(
            mesh.boundary_data_sender.clone().unwrap(),
            mesh.boundary_data_receiver.clone().unwrap(),
        );

        mesh_receiver.receive_boundary_data();
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /// Tests sending boundary data without a receiver does not cause a failure.
    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        mesh.send_boundary_data();
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    /// Tests the addition of a new entity to the mesh.  
    /// Tests the addition of a new entity to the mesh.  
    /// Verifies that the entity is successfully added to the mesh's entity set.  
    /// Tests the addition of a new entity to the mesh.
    /// Verifies that the entity is successfully added to the mesh's entity set.  
    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    /// Tests the iterator over the mesh's vertex coordinates.  
    /// Tests the iterator over the mesh's vertex coordinates.  
    /// Verifies that the iterator returns the correct vertex IDs.  
    /// Tests the iterator over the mesh's vertex coordinates.
    /// Verifies that the iterator returns the correct vertex IDs.  
    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::domain::mesh::hierarchical::MeshNode;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;
    use crossbeam::channel::unbounded;

    /// Full integration test for mesh operations including entity addition,  
    /// boundary data synchronization, and applying constraints at hanging nodes.
    #[test]
    fn test_full_mesh_integration() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let cell1 = MeshEntity::Cell(1);

        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(cell1);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);

        let (sender, receiver) = unbounded();
        mesh.set_boundary_channels(sender, receiver);
        mesh.send_boundary_data();

        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(
            mesh.boundary_data_sender.clone().unwrap(),
            mesh.boundary_data_receiver.clone().unwrap(),
        );
        mesh_receiver.receive_boundary_data();

        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[0.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[1.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&3), Some(&[0.0, 1.0, 0.0]));

        let mut node = MeshNode::Leaf(cell1);
        node.refine(|&_cell| [
            MeshEntity::Cell(2), MeshEntity::Cell(3), MeshEntity::Cell(4), MeshEntity::Cell(5)
        ]);

        if let MeshNode::Branch { ref children, .. } = node {
            assert_eq!(children.len(), 4);
            assert_eq!(children[0], MeshNode::Leaf(MeshEntity::Cell(2)));
            assert_eq!(children[1], MeshNode::Leaf(MeshEntity::Cell(3)));
            assert_eq!(children[2], MeshNode::Leaf(MeshEntity::Cell(4)));
            assert_eq!(children[3], MeshNode::Leaf(MeshEntity::Cell(5)));
        } else {
            panic!("Expected the node to be refined into a branch.");
        }

        let rcm_order = mesh.rcm_ordering(vertex1);
        assert!(rcm_order.len() > 0);

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
        ];
        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);
        assert_eq!(parent_dofs, [1.0, 1.5, 2.0, 2.5]);
    }
}
```

---

`src/domain/mesh/topology.rs`

```rust
// src/boundary/mesh/topology.rs

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use crate::domain::mesh::Mesh;
use rustc_hash::FxHashSet;
use std::sync::{Arc, RwLock};

/// `TopologyValidation` struct responsible for checking mesh entity connectivity and uniqueness.
pub struct TopologyValidation<'a> {
    sieve: &'a Sieve,
    entities: &'a Arc<RwLock<FxHashSet<MeshEntity>>>,
}

impl<'a> TopologyValidation<'a> {
    /// Creates a new `TopologyValidation` instance for a given mesh.
    pub fn new(mesh: &'a Mesh) -> Self {
        TopologyValidation {
            sieve: &mesh.sieve,
            entities: &mesh.entities,
        }
    }

    /// Validates that each `Cell` in the mesh has the correct connections to `Faces` and `Vertices`.
    /// Returns `true` if all cells are correctly connected, `false` otherwise.
    pub fn validate_connectivity(&self) -> bool {
        for cell in self.get_cells() {
            if !self.validate_cell_connectivity(&cell) {
                return false;
            }
        }
        true
    }

    /// Validates that `Edges` in the mesh are unique and not duplicated within any `Cell`.
    /// Returns `true` if all edges are unique, `false` otherwise.
    pub fn validate_unique_relationships(&self) -> bool {
        for cell in self.get_cells() {
            println!("Validating edges for cell: {:?}", cell); // Debugging statement
            let mut edge_set = FxHashSet::default(); // Reset edge_set for each cell
    
            if !self.validate_unique_edges_in_cell(&cell, &mut edge_set) {
                println!("Duplicate edge detected in cell: {:?}", cell); // Debugging statement
                return false;
            }
        }
        true
    }

    /// Retrieves all `Cell` entities from the mesh.
    fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Checks if a `Cell` is connected to valid `Faces` and `Vertices`.
    fn validate_cell_connectivity(&self, cell: &MeshEntity) -> bool {
        if let Some(connected_faces) = self.sieve.cone(cell) {
            for face in connected_faces {
                if !matches!(face, MeshEntity::Face(_)) {
                    return false;
                }
                // Check each face is connected to valid vertices
                if let Some(vertices) = self.sieve.cone(&face) {
                    if !vertices.iter().all(|v| matches!(v, MeshEntity::Vertex(_))) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Checks if `Edges` within a `Cell` are unique.
    fn validate_unique_edges_in_cell(&self, cell: &MeshEntity, edge_set: &mut FxHashSet<MeshEntity>) -> bool {
        if let Some(edges) = self.sieve.cone(cell) {
            for edge in edges {
                if !matches!(edge, MeshEntity::Edge(_)) {
                    return false;
                }
                // Debugging: Print edge and current edge set
                println!("Checking edge {:?} in cell {:?}. Current edge set: {:?}", edge, cell, edge_set);
                
                // Check for duplication in `edge_set`
                if !edge_set.insert(edge) {
                    println!("Duplicate edge {:?} found in cell {:?}", edge, cell); // Debugging statement
                    return false; // Duplicate edge found
                }
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::Mesh;

    #[test]
    fn test_valid_connectivity() {
        let mesh = Mesh::new();
        // Adding sample entities and relationships to the mesh for testing
        let cell = MeshEntity::Cell(1);
        let face1 = MeshEntity::Face(1);
        let face2 = MeshEntity::Face(2);
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);

        mesh.add_entity(cell);
        mesh.add_entity(face1);
        mesh.add_entity(face2);
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);

        mesh.add_arrow(cell, face1);
        mesh.add_arrow(cell, face2);
        mesh.add_arrow(face1, vertex1);
        mesh.add_arrow(face1, vertex2);
        mesh.add_arrow(face2, vertex2);
        mesh.add_arrow(face2, vertex3);

        let topology_validation = TopologyValidation::new(&mesh);
        assert!(topology_validation.validate_connectivity(), "Connectivity validation failed");
    }

    #[test]
    fn test_unique_relationships() {
        let mesh = Mesh::new();
        // Adding sample entities and relationships to the mesh for testing
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let edge1 = MeshEntity::Edge(1);
        let edge2 = MeshEntity::Edge(2);
        let edge3 = MeshEntity::Edge(3);

        mesh.add_entity(cell1);
        mesh.add_entity(cell2);
        mesh.add_entity(edge1);
        mesh.add_entity(edge2);
        mesh.add_entity(edge3);

        // Establish valid relationships between cells and edges
        mesh.add_arrow(cell1, edge1);
        mesh.add_arrow(cell1, edge2);
        mesh.add_arrow(cell2, edge2); // Edge2 is reused here, valid as it's unique within each cell.
        mesh.add_arrow(cell2, edge3);

        // Initialize TopologyValidation to verify unique relationships per cell
        let topology_validation = TopologyValidation::new(&mesh);

        // Check that relationships are valid and unique within the constraints of the current design
        assert!(topology_validation.validate_unique_relationships(), "Unique relationships validation failed");

        // Additional checks for edge-sharing across cells can be done if necessary
    }


}
```