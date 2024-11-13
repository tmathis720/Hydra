To add robustness to the `Mesh` structure by implementing a topology validation framework, we should focus on verifying various properties before the solution process. Let's follow a test-driven development (TDD) approach and ensure clean architecture principles are in place for easy testing and future maintainability. Here’s a roadmap based on these principles:

### 1. **Outline of Validation Checks**
   - **Boundary Condition Validation**: Ensure boundary conditions are consistently applied across the mesh.
   - **Overlap and Delta Validations**: For distributed/partitioned meshes, validate overlaps and transformation data (`Delta`) across partitions.

### 2. **Modular Test Structure**
   Create test modules within `Mesh`, `Sieve`, and boundary condition components. This test-driven approach will confirm each function independently and collaboratively maintains expected outcomes. Here’s an outline for each module:

#### `BoundaryValidation` Module
   - **Responsibilities**:
     - Confirm that boundary conditions are only applied to entities tagged as boundaries.
     - Ensure compatibility of boundary data across partitioned entities.
   
   - **Key Tests**:
     - `test_boundary_condition_application`: Check boundary condition application on each boundary entity.
     - `test_boundary_data_synchronization`: Verify that boundary data sent via channels is consistent across partitions.

#### `OverlapValidation` and `DeltaValidation` Modules
   - **Responsibilities**:
     - Validate that `local_entities` and `ghost_entities` are correctly partitioned and synchronized across overlaps.
     - Ensure `Delta` values for transformations are correctly assigned and consistent across all `MeshEntity` instances.

   - **Key Tests**:
     - `test_overlap_partitioning`: Check that each entity belongs to the correct partition.
     - `test_delta_synchronization`: Validate the synchronization of transformation data (`Delta`) for overlapping entities.

### 3. **Implementation Approach with Clean Architecture Principles**

Each module should expose a minimal public interface:
   - **Encapsulation**: Keep utility functions within modules private unless they’re widely reusable.
   - **Dependency Injection**: Use dependency injection to facilitate testing by injecting mock data and functions where applicable.
   - **Test Isolation**: Design tests to isolate specific functionalities (e.g., connectivity, geometry) so they can be independently verified and debugged.

### 4. **Matrix and Solver Integrations**
   - **Matrix Operations**: Use `Faer` for matrix manipulations, ensuring efficient sparse matrix storage and solver performance for geometric and boundary computations.
   - **Solver Approaches**: For more complex geometries and connectivity checks, leverage iterative solvers like GMRES, which can effectively handle large sparse systems that arise from validating the mesh structure.

### 5. **Test-Driven Development Process**

For each validation aspect:
   - Write unit tests covering expected and edge cases.
   - Implement the corresponding functions to pass the tests.
   - Refactor as needed, ensuring clean architecture adherence.

Each test will serve as a continuous validation to reinforce mesh integrity across any future modifications or enhancements to the mesh structure.

---

# Present Instructions

We will develop the BoundaryValidation struct similar to the GeometryValidation struct we just successfully developed and tested.

Below is the relevant source code to provide details and appropriate context to facilitate you can generate accurate code in response to this prompt.

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

`src/domain/mesh/mod.rs`

```rust
pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;
pub mod topology;

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

Note that the following file is highly relevant to the development of `src/domain/mesh/geometry_validation.rs` and generation of accurate code.

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

`src/geometry/mod.rs`

```rust
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use std::sync::Mutex;

// Module for handling geometric data and computations
// 2D Shape Modules
pub mod quadrilateral;
pub mod triangle;
// 3D Shape Modules
pub mod tetrahedron;
pub mod hexahedron;
pub mod prism;
pub mod pyramid;

/// The `Geometry` struct stores geometric data for a mesh, including vertex coordinates, 
/// cell centroids, and volumes. It also maintains a cache of computed properties such as 
/// volume and centroid for reuse, optimizing performance by avoiding redundant calculations.
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>,  // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,         // Volumes of each cell
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>, // Cache for computed properties, with thread safety
}

/// The `GeometryCache` struct stores computed properties of geometric entities, 
/// including volume, centroid, and area, with an optional "dirty" flag for lazy evaluation.
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,  // Stores a precomputed normal vector for a face
}

/// `CellShape` enumerates the different cell shapes in a mesh, including:
/// * Tetrahedron
/// * Hexahedron
/// * Prism
/// * Pyramid
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// `FaceShape` enumerates the different face shapes in a mesh, including:
/// * Triangle
/// * Quadrilateral
#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}

impl Geometry {
    /// Initializes a new `Geometry` instance with empty data.
    pub fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            cell_centroids: Vec::new(),
            cell_volumes: Vec::new(),
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// Adds or updates a vertex in the geometry. If the vertex already exists,
    /// it updates its coordinates; otherwise, it adds a new vertex.
    ///
    /// # Arguments
    /// * `vertex_index` - The index of the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) {
        if vertex_index >= self.vertices.len() {
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }
        self.vertices[vertex_index] = coords;
        self.invalidate_cache();
    }

    /// Computes and returns the centroid of a specified cell using the cell's shape and vertices.
    /// Caches the result for reuse.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3] {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let centroid = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(&cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().centroid = Some(centroid);
        centroid
    }

    /// Computes the volume of a given cell using its shape and vertex coordinates.
    /// The computed volume is cached for efficiency.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64 {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let volume = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(&cell_vertices),
            CellShape::Prism => self.compute_prism_volume(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().volume = Some(volume);
        volume
    }

    /// Calculates Euclidean distance between two points in 3D space.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt()
    }

    /// Computes the area of a 2D face based on its shape, caching the result.
    pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.area) {
            return cached;
        }

        let area = match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        };

        self.cache.lock().unwrap().entry(face_id).or_default().area = Some(area);
        area
    }

    /// Computes the centroid of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the face centroid.
    pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        match face_shape {
            FaceShape::Triangle => self.compute_triangle_centroid(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_centroid(face_vertices),
        }
    }

    /// Computes and caches the normal vector for a face based on its shape.
    ///
    /// This function determines the face shape and calls the appropriate 
    /// function to compute the normal vector.
    ///
    /// # Arguments
    /// * `mesh` - A reference to the mesh.
    /// * `face` - The face entity for which to compute the normal.
    /// * `cell` - The cell associated with the face, used to determine the orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector, or `None` if it could not be computed.
    pub fn compute_face_normal(
        &mut self,
        mesh: &Mesh,
        face: &MeshEntity,
        _cell: &MeshEntity,
    ) -> Option<[f64; 3]> {
        let face_id = face.get_id();

        // Check if the normal is already cached
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal) {
            return Some(cached);
        }

        let face_vertices = mesh.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let normal = match face_shape {
            FaceShape::Triangle => self.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_normal(&face_vertices),
        };

        // Cache the normal vector for future use
        self.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal);

        Some(normal)
    }

    /// Invalidate the cache when geometry changes (e.g., vertex updates).
    fn invalidate_cache(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    /// Computes the total volume of all cells.
    pub fn compute_total_volume(&self) -> f64 {
        self.cell_volumes.par_iter().sum()
    }

    /// Updates all cell volumes in parallel using mesh information.
    pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
        let new_volumes: Vec<f64> = mesh
            .get_cells()
            .par_iter()
            .map(|cell| {
                let mut temp_geometry = Geometry::new();
                temp_geometry.compute_cell_volume(mesh, cell)
            })
            .collect();

        self.cell_volumes = new_volumes;
    }

    /// Computes the total centroid of all cells.
    pub fn compute_total_centroid(&self) -> [f64; 3] {
        let total_centroid: [f64; 3] = self.cell_centroids
            .par_iter()
            .cloned()
            .reduce(
                || [0.0, 0.0, 0.0],
                |acc, centroid| [
                    acc[0] + centroid[0],
                    acc[1] + centroid[1],
                    acc[2] + centroid[2],
                ],
            );

        let num_centroids = self.cell_centroids.len() as f64;
        [
            total_centroid[0] / num_centroids,
            total_centroid[1] / num_centroids,
            total_centroid[2] / num_centroids,
        ]
    }
}


#[cfg(test)]
mod tests {
    use crate::geometry::{Geometry, CellShape, FaceShape};
    use crate::domain::{MeshEntity, mesh::Mesh, Sieve};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_set_vertex() {
        let mut geometry = Geometry::new();

        // Set vertex at index 0
        geometry.set_vertex(0, [1.0, 2.0, 3.0]);
        assert_eq!(geometry.vertices[0], [1.0, 2.0, 3.0]);

        // Update vertex at index 0
        geometry.set_vertex(0, [4.0, 5.0, 6.0]);
        assert_eq!(geometry.vertices[0], [4.0, 5.0, 6.0]);

        // Set vertex at a higher index
        geometry.set_vertex(3, [7.0, 8.0, 9.0]);
        assert_eq!(geometry.vertices[3], [7.0, 8.0, 9.0]);

        // Ensure the intermediate vertices are initialized to [0.0, 0.0, 0.0]
        assert_eq!(geometry.vertices[1], [0.0, 0.0, 0.0]);
        assert_eq!(geometry.vertices[2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_distance() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 4.0, 0.0];

        let distance = Geometry::compute_distance(&p1, &p2);

        // The expected distance is 5 (Pythagoras: sqrt(3^2 + 4^2))
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_compute_cell_centroid_tetrahedron() {
        // Create a new Sieve and Mesh.
        let sieve = Sieve::new();
        let mut mesh = Mesh {
            sieve: Arc::new(sieve),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: None,
            boundary_data_receiver: None,
        };

        // Define vertices and a cell.
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
        let cell = MeshEntity::Cell(1);

        // Set vertex coordinates.
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);

        // Add entities to the mesh.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.add_entity(cell);

        // Establish relationships between the cell and vertices.
        mesh.add_arrow(cell, vertex1);
        mesh.add_arrow(cell, vertex2);
        mesh.add_arrow(cell, vertex3);
        mesh.add_arrow(cell, vertex4);

        // Verify that `get_cell_vertices` retrieves the correct vertices.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        assert_eq!(cell_vertices.len(), 4, "Expected 4 vertices for a tetrahedron cell.");

        // Validate the shape before computing.
        assert_eq!(mesh.get_cell_shape(&cell), Ok(CellShape::Tetrahedron));

        // Create a Geometry instance and compute the centroid.
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(&mesh, &cell);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_compute_face_area_triangle() {
        let mut geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_face_area(1, FaceShape::Triangle, &triangle_vertices);

        // Expected area: 0.5 * base * height = 0.5 * 3.0 * 4.0 = 6.0
        assert_eq!(area, 6.0);
    }

    #[test]
    fn test_compute_face_centroid_quadrilateral() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let centroid = geometry.compute_face_centroid(FaceShape::Quadrilateral, &quad_vertices);

        // Expected centroid is the geometric center: (0.5, 0.5, 0.0)
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_compute_total_volume() {
        let mut geometry = Geometry::new();
        let _mesh = Mesh::new();

        // Example setup: Define cells with known volumes
        // Here, you would typically define several cells and their volumes for the test
        geometry.cell_volumes = vec![1.0, 2.0, 3.0];

        // Expected total volume is the sum of individual cell volumes: 1.0 + 2.0 + 3.0 = 6.0
        assert_eq!(geometry.compute_total_volume(), 6.0);
    }

    #[test]
    fn test_compute_face_normal_triangle() {
        let geometry = Geometry::new();
        
        // Define vertices for a triangular face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        // Define the face as a triangle
        let _face = MeshEntity::Face(1);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal without setting up mesh connectivity
        let normal = geometry.compute_triangle_normal(&vertices);

        // Expected normal for a triangle in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_quadrilateral() {
        let geometry = Geometry::new();

        // Define vertices for a quadrilateral face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        // Define the face as a quadrilateral
        let _face = MeshEntity::Face(2);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal for quadrilateral
        let normal = geometry.compute_quadrilateral_normal(&vertices);

        // Expected normal for a quadrilateral in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_caching() {
        let geometry = Geometry::new();

        // Define vertices for a triangular face
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        let face_id = 3; // Unique identifier for caching
        let _face = MeshEntity::Face(face_id);
        let _cell = MeshEntity::Cell(1);

        // First computation to populate the cache
        let normal_first = geometry.compute_triangle_normal(&vertices);

        // Manually retrieve from cache to verify caching behavior
        geometry.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal_first);
        let cached_normal = geometry.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal);

        // Verify that the cached value matches the first computed value
        assert_eq!(Some(normal_first), cached_normal);
    }

    #[test]
    fn test_compute_face_normal_unsupported_shape() {
        let geometry = Geometry::new();

        // Define vertices for a pentagon (unsupported)
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.5, 0.5, 0.0], // vertex 5
        ];

        let _face = MeshEntity::Face(4);
        let _cell = MeshEntity::Cell(1);

        // Since compute_face_normal expects only triangles or quadrilaterals, it should return None
        let face_shape = match vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return, // Unsupported shape, skip test
        };

        // Attempt to compute the normal for an unsupported shape
        let normal = match face_shape {
            FaceShape::Triangle => Some(geometry.compute_triangle_normal(&vertices)),
            FaceShape::Quadrilateral => Some(geometry.compute_quadrilateral_normal(&vertices)),
        };

        // Assert that the function correctly handles unsupported shapes by skipping normal computation
        assert!(normal.is_none());
    }
}
```

---

Now generate the source code for `src/domain/mesh/boundary_validation.rs` based on the following instructions:


#### `BoundaryValidation` Module
   - **Responsibilities**:
     - Confirm that boundary conditions are only applied to entities tagged as boundaries.
     - Ensure compatibility of boundary data across partitioned entities.
   
   - **Key Tests**:
     - `test_boundary_condition_application`: Check boundary condition application on each boundary entity.
     - `test_boundary_data_synchronization`: Verify that boundary data sent via channels is consistent across partitions.