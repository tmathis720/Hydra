Here is some background code to help address the following test failures:

```bash
failures:

---- domain::stratify::tests::test_stratify_multiple_entities_per_dimension stdout ----
thread 'domain::stratify::tests::test_stratify_multiple_entities_per_dimension' panicked at src\domain\stratify.rs:114:35:
called `Option::unwrap()` on a `None` value
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- domain::stratify::tests::test_stratify_single_entity_per_dimension stdout ----
thread 'domain::stratify::tests::test_stratify_single_entity_per_dimension' panicked at src\domain\stratify.rs:88:35:
called `Option::unwrap()` on a `None` value

---- domain::stratify::tests::test_stratify_large_mesh stdout ----
thread 'domain::stratify::tests::test_stratify_large_mesh' panicked at src\domain\stratify.rs:164:35:
called `Option::unwrap()` on a `None` value


failures:
    domain::stratify::tests::test_stratify_large_mesh
    domain::stratify::tests::test_stratify_multiple_entities_per_dimension
    domain::stratify::tests::test_stratify_single_entity_per_dimension
```

Here is `src/domain/mesh_entity.rs`

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
use lazy_static::lazy_static;

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

lazy_static! {
    static ref GLOBAL_MESH: Arc<RwLock<Mesh>> = Arc::new(RwLock::new(Mesh::new()));
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

    pub fn global() -> Arc<RwLock<Mesh>> {
        GLOBAL_MESH.clone()
    }
}

#[cfg(test)]
pub mod tests;
```

---

`src/domain/mesh/entities.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use dashmap::DashMap;
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

    /// Maps each `MeshEntity` in the mesh to a unique index.
    pub fn get_entity_to_index(&self) -> DashMap<MeshEntity, usize> {
        let entity_to_index = DashMap::new();
        let entities = self.entities.read().unwrap();
        entities.iter().enumerate().for_each(|(index, entity)| {
            entity_to_index.insert(entity.clone(), index);
        });

        entity_to_index
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

---

`src/input_output/mesh_generation.rs`

```rust
use crate::domain::{mesh::Mesh, MeshEntity};

pub struct MeshGenerator;

impl MeshGenerator {
    /// Generates a 2D rectangular mesh with a specified width, height, and resolution (nx, ny).
    pub fn generate_rectangle_2d(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_2d(width, height, nx, ny);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_quadrilateral_cells(&mut mesh, nx, ny);
        mesh
    }

    /// Generates a 3D rectangular mesh with a specified width, height, depth, and resolution (nx, ny, nz).
    pub fn generate_rectangle_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_grid_nodes_3d(width, height, depth, nx, ny, nz);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_hexahedral_cells(&mut mesh, nx, ny, nz);
        Self::_generate_faces_3d(&mut mesh, nx, ny, nz);
        mesh
    }

    /// Generates a circular 2D mesh with a given radius and number of divisions.
    pub fn generate_circle(radius: f64, num_divisions: usize) -> Mesh {
        let mut mesh = Mesh::new();
        let nodes = Self::generate_circle_nodes(radius, num_divisions);
        for (id, position) in nodes.into_iter().enumerate() {
            mesh.set_vertex_coordinates(id, position);
        }
        Self::generate_triangular_cells(&mut mesh, num_divisions);
        mesh
    }

    // --- Internal Helper Functions ---

    /// Generate 2D grid nodes for rectangular mesh
    fn generate_grid_nodes_2d(width: f64, height: f64, nx: usize, ny: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([i as f64 * dx, j as f64 * dy, 0.0]);
            }
        }
        nodes
    }

    /// Generate 3D grid nodes for rectangular mesh
    fn generate_grid_nodes_3d(width: f64, height: f64, depth: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push([i as f64 * dx, j as f64 * dy, k as f64 * dz]);
                }
            }
        }
        nodes
    }

    /// Generate circle nodes for circular mesh
    fn generate_circle_nodes(radius: f64, num_divisions: usize) -> Vec<[f64; 3]> {
        let mut nodes = Vec::new();
        nodes.push([0.0, 0.0, 0.0]);
        for i in 0..num_divisions {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_divisions as f64);
            nodes.push([radius * theta.cos(), radius * theta.sin(), 0.0]);
        }
        nodes
    }

    /// Generate quadrilateral cells for a 2D rectangular mesh
    fn generate_quadrilateral_cells(mesh: &mut Mesh, nx: usize, ny: usize) {
        let mut cell_id = 0;
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = n1 + (nx + 1) + 1;
                let n4 = n1 + (nx + 1);
                let cell = MeshEntity::Cell(cell_id);
                cell_id += 1;
                mesh.add_entity(cell.clone());
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3));
                mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4));
            }
        }
    }

    /// Generate hexahedral cells for a 3D rectangular mesh
    fn generate_hexahedral_cells(mesh: &mut Mesh, nx: usize, ny: usize, nz: usize) {
        let mut cell_id = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n1 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    let n2 = n1 + 1;
                    let n3 = n1 + (nx + 1);
                    let n4 = n3 + 1;
                    let n5 = n1 + (ny + 1) * (nx + 1);
                    let n6 = n5 + 1;
                    let n7 = n5 + (nx + 1);
                    let n8 = n7 + 1;
                    let cell = MeshEntity::Cell(cell_id);
                    cell_id += 1;
                    mesh.add_entity(cell.clone());
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n1));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n2));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n3));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n4));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n5));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n6));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n7));
                    mesh.add_relationship(cell.clone(), MeshEntity::Vertex(n8));
                }
            }
        }
    }

    /// Generate triangular cells for a circular mesh
    fn generate_triangular_cells(mesh: &mut Mesh, num_divisions: usize) {
        let mut cell_id = 0;
        for i in 0..num_divisions {
            let next = (i + 1) % num_divisions;
            let cell = MeshEntity::Cell(cell_id);
            cell_id += 1;
            mesh.add_entity(cell.clone());
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(0));
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(i + 1));
            mesh.add_relationship(cell.clone(), MeshEntity::Vertex(next + 1));
        }
    }

    /// Generate faces for a 3D rectangular mesh.
    fn _generate_faces_3d(mesh: &mut Mesh, nx: usize, ny: usize, nz: usize) {
        let mut face_id = 0;
        
        // Loop over all cells to add faces
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n1 = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    let n2 = n1 + 1;
                    let n3 = n1 + (nx + 1);
                    let n4 = n3 + 1;
                    let n5 = n1 + (ny + 1) * (nx + 1);
                    let n6 = n5 + 1;
                    let n7 = n5 + (nx + 1);
                    let n8 = n7 + 1;

                    // Define the vertices for each face of a hexahedron
                    let faces = [
                        (n1, n2, n4, n3), // front face
                        (n5, n6, n8, n7), // back face
                        (n1, n5, n7, n3), // left face
                        (n2, n6, n8, n4), // right face
                        (n3, n4, n8, n7), // top face
                        (n1, n2, n6, n5), // bottom face
                    ];

                    // Add each face to the mesh
                    for &(v1, v2, v3, v4) in &faces {
                        let face = MeshEntity::Face(face_id);
                        face_id += 1;
                        mesh.add_entity(face.clone());
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v1));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v2));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v3));
                        mesh.add_relationship(face.clone(), MeshEntity::Vertex(v4));
                    }
                }
            }
        }
    }

}
```