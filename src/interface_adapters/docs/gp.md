I want to develop a interface to make building simple domains easier in Hydra. I am providing the current code for the `Domain` module below. Please provide a detailed outline of the implementation for the interface, which will tie together the construction of a domain with a relatively simple API. For now, assume that we will develop this adapter in the file `src/interface_adapters/domain_adapter.rs`. Here is the current source code tree of `Domain` followed by the source code for each file listed. 

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
Here is `src/domain/mesh_entity.rs` : 

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
use std::ops::{AddAssign, Mul};
use std::ops::{Add, Sub, Neg, Div};

/// Represents a 3D vector with three floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Vector3(pub [f64; 3]);

impl AddAssign for Vector3 {
    /// Implements addition assignment for `Vector3`.
    /// Adds the components of another `Vector3` to this vector component-wise.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    /// Multiplies each component of the vector by a scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector3([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Vector3 {
    /// Provides an iterator over the vector's components.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for Vector3 {
    type Output = f64;

    /// Indexes into the vector by position.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Vector3 {
    /// Provides mutable access to the indexed component of the vector.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Vector3 {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 3>;

    /// Converts the vector into an iterator of its components.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector3 {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    /// Converts a reference to the vector into an iterator of its components.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Represents a 3x3 tensor with floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    /// Implements addition assignment for `Tensor3x3`.
    /// Adds the components of another tensor to this tensor component-wise.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] += other.0[i][j];
            }
        }
    }
}

impl Mul<f64> for Tensor3x3 {
    type Output = Tensor3x3;

    /// Implements scalar multiplication for `Tensor3x3`.
    /// Multiplies each component of the tensor by a scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.0[i][j] * rhs;
            }
        }
        Tensor3x3(result)
    }
}

/// Represents a scalar value.
#[derive(Clone, Copy, Debug)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    /// Implements addition assignment for `Scalar`.
    /// Adds another scalar value to this scalar.
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    /// Implements scalar multiplication for `Scalar`.
    /// Multiplies this scalar by another scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

/// Represents a 2D vector with two floating-point components.
#[derive(Clone, Copy, Debug)]
pub struct Vector2(pub [f64; 2]);

impl AddAssign for Vector2 {
    /// Implements addition assignment for `Vector2`.
    /// Adds the components of another `Vector2` to this vector component-wise.
    fn add_assign(&mut self, other: Self) {
        for i in 0..2 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    /// Implements scalar multiplication for `Vector2`.
    /// Multiplies each component of the vector by a scalar `rhs`.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector2([self.0[0] * rhs, self.0[1] * rhs])
    }
}

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.
#[derive(Clone, Debug)]
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T>
where
    T: Clone + AddAssign + Mul<f64, Output = T> + Send + Sync,
{
    /// Creates a new `Section` with an empty data map.
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Associates a given `MeshEntity` with a value of type `T`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves a copy of the data associated with the specified `MeshEntity`, if it exists.
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates all data values in the section in parallel using the provided function.
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
    {
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Updates the section by adding the derivative multiplied by a time step `dt`.
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) {
        for entry in derivative.data.iter() {
            let entity = entry.key();
            let deriv_value = entry.value().clone() * dt;
            if let Some(mut state_value) = self.data.get_mut(entity) {
                *state_value.value_mut() += deriv_value;
            } else {
                self.data.insert(*entity, deriv_value);
            }
        }
    }

    /// Returns a list of all `MeshEntity` objects associated with this section.
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Returns all data stored in the section as a vector of immutable copies.
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Clears all data from the section.
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Scales all data values in the section by the specified factor.
    pub fn scale(&self, factor: f64) {
        self.parallel_update(|value| {
            *value = value.clone() * factor;
        });
    }
}

// Add for Section<Scalar>
impl Add for Section<Scalar> {
    type Output = Section<Scalar>;

    fn add(self, rhs: Self) -> Self::Output {
        let result = self.clone();
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 += value.0;
            } else {
                result.set_data(*key, *value);
            }
        }
        result
    }
}


// Sub for Section<Scalar>
impl Sub for Section<Scalar> {
    type Output = Section<Scalar>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result = self.clone();
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 -= value.0;
            } else {
                result.set_data(*key, Scalar(-value.0));
            }
        }
        result
    }
}


// Neg for Section<Scalar>
impl Neg for Section<Scalar> {
    type Output = Section<Scalar>;

    fn neg(self) -> Self::Output {
        let result = self.clone();
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 = -value.0;
        }
        result
    }
}


// Div for Section<Scalar>
impl Div<f64> for Section<Scalar> {
    type Output = Section<Scalar>;

    fn div(self, rhs: f64) -> Self::Output {
        let result = self.clone();
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 /= rhs;
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    // Helper function to create a MeshEntity for testing
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id) // Adjust according to the MeshEntity variant in your implementation
    }

    #[test]
    fn test_vector3_add_assign() {
        let mut v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([0.5, 0.5, 0.5]);
        v1 += v2;

        assert_eq!(v1.0, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_vector3_mul() {
        let v = Vector3([1.0, 2.0, 3.0]);
        let scaled = v * 2.0;

        assert_eq!(scaled.0, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor3x3_add_assign() {
        let mut t1 = Tensor3x3([[1.0; 3]; 3]);
        let t2 = Tensor3x3([[0.5; 3]; 3]);
        t1 += t2;

        assert_eq!(t1.0, [[1.5; 3]; 3]);
    }

    #[test]
    fn test_tensor3x3_mul() {
        let t = Tensor3x3([[1.0; 3]; 3]);
        let scaled = t * 2.0;

        assert_eq!(scaled.0, [[2.0; 3]; 3]);
    }

    #[test]
    fn test_section_set_and_restrict_data() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        let value = Scalar(3.14);

        section.set_data(entity, value);
        let retrieved = section.restrict(&entity);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, 3.14);
    }

    #[test]
    fn test_section_parallel_update() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=10).map(create_test_mesh_entity).collect();

        for (i, entity) in entities.iter().enumerate() {
            section.set_data(*entity, Scalar(i as f64));
        }

        section.parallel_update(|value| {
            value.0 *= 2.0;
        });

        for (i, entity) in entities.iter().enumerate() {
            assert_eq!(section.restrict(entity).unwrap().0, (i as f64) * 2.0);
        }
    }

    #[test]
    fn test_section_update_with_derivative() {
        let section: Section<Scalar> = Section::new();
        let derivative: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);

        section.set_data(entity, Scalar(1.0));
        derivative.set_data(entity, Scalar(0.5));

        section.update_with_derivative(&derivative, 2.0);

        assert_eq!(section.restrict(&entity).unwrap().0, 2.0);
    }

    #[test]
    fn test_section_entities() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=5).map(create_test_mesh_entity).collect();

        for entity in &entities {
            section.set_data(*entity, Scalar(1.0));
        }

        let retrieved_entities = section.entities();
        assert_eq!(retrieved_entities.len(), entities.len());
    }

    #[test]
    fn test_section_clear() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        section.clear();

        assert!(section.restrict(&entity).is_none());
    }

    #[test]
    fn test_section_scale() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(2.0));

        section.scale(3.0);

        assert_eq!(section.restrict(&entity).unwrap().0, 6.0);
    }

    // Debugging utilities for better output on failure
    fn debug_section_data<T>(section: &Section<T>)
    where
        T: std::fmt::Debug,
    {
        println!("Section data:");
        for entry in section.data.iter() {
            println!("{:?} -> {:?}", entry.key(), entry.value());
        }
    }

    #[test]
    fn test_debugging_output() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        debug_section_data(&section);
    }
}
```

---

`src/domain/entity_fill.rs`

```rust
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashSet;

impl Sieve {
    /// Infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.
    /// For 2D meshes, this method generates edges by connecting vertices of a cell.
    /// These edges are then associated with the corresponding vertices in the sieve.
    pub fn fill_missing_entities(&self) {
        let mut edge_set: FxHashSet<(MeshEntity, MeshEntity)> = FxHashSet::default();
        let mut next_edge_id = 0;
        let mut arrows_to_add: Vec<(MeshEntity, MeshEntity)> = Vec::new();

        // Collect cells and their associated vertices to avoid modifying the map during iteration
        let cell_vertices: Vec<(MeshEntity, Vec<MeshEntity>)> = self.adjacency.iter()
            .filter_map(|entry| {
                let cell = entry.key();
                if let MeshEntity::Cell(_) = cell {
                    let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                    Some((cell.clone(), vertices))
                } else {
                    None
                }
            }).collect();

        for (cell, vertices) in cell_vertices {
            if vertices.len() < 3 {
                eprintln!("Skipping cell with fewer than 3 vertices: {:?}", cell);
                continue;
            }

            eprintln!("Processing cell: {:?}", cell);

            for i in 0..vertices.len() {
                let (v1, v2) = (
                    vertices[i].clone(),
                    vertices[(i + 1) % vertices.len()].clone(),
                );
                let edge_key = if v1 < v2 {
                    (v1.clone(), v2.clone())
                } else {
                    (v2.clone(), v1.clone())
                };

                if edge_set.contains(&edge_key) {
                    eprintln!("Edge {:?} already processed, skipping.", edge_key);
                    continue;
                }
                edge_set.insert(edge_key.clone());

                let edge = MeshEntity::Edge(next_edge_id);
                next_edge_id += 1;

                // Collect arrows to add after iteration
                arrows_to_add.push((v1.clone(), edge.clone()));
                arrows_to_add.push((v2.clone(), edge.clone()));
                arrows_to_add.push((edge.clone(), v1.clone()));
                arrows_to_add.push((edge.clone(), v2.clone()));

                eprintln!(
                    "Created edge {:?} between {:?} and {:?}",
                    edge, v1, v2
                );
            }
        }

        // Now add all arrows to self.adjacency outside the iteration
        for (from, to) in arrows_to_add {
            self.add_arrow(from, to);
        }
    }
}


#[cfg(test)]
mod tests {
    use std::time::Instant;
    use dashmap::DashMap;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    fn run_with_timeout<F: FnOnce()>(test_name: &str, func: F) {
        let start = Instant::now();
        func();
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "{} test timed out!", test_name);
    }

    #[test]
    fn test_fill_missing_entities_with_empty_sieve() {
        run_with_timeout("test_fill_missing_entities_with_empty_sieve", || {
            let sieve = Sieve::new();
            sieve.fill_missing_entities();

            assert!(
                sieve.adjacency.is_empty(),
                "No edges should be created for an empty sieve"
            );
        });
    }

    #[test]
    fn test_fill_missing_entities_for_single_cell() {
        run_with_timeout("test_fill_missing_entities_for_single_cell", || {
            let sieve = Sieve::new();
            let cell = MeshEntity::Cell(1);
            let vertices = vec![
                MeshEntity::Vertex(1),
                MeshEntity::Vertex(2),
                MeshEntity::Vertex(3),
            ];

            sieve
                .adjacency
                .entry(cell.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices {
                sieve.add_arrow(cell.clone(), vertex.clone());
            }

            sieve.fill_missing_entities();

            let expected_edges = vec![
                (MeshEntity::Vertex(1), MeshEntity::Vertex(2)),
                (MeshEntity::Vertex(2), MeshEntity::Vertex(3)),
                (MeshEntity::Vertex(3), MeshEntity::Vertex(1)),
            ];

            for (v1, v2) in expected_edges {
                let edge_exists = sieve.adjacency.get(&v1).map_or(false, |adj| {
                    adj.iter().any(|entry| {
                        if let MeshEntity::Edge(_) = entry.key() {
                            sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                                edge_adj.contains_key(&v2)
                            })
                        } else {
                            false
                        }
                    })
                });
                assert!(edge_exists, "Edge ({:?}, {:?}) should exist", v1, v2);
            }
        });
    }

    #[test]
    fn test_fill_missing_entities_no_duplicate_edges() {
        run_with_timeout("test_fill_missing_entities_no_duplicate_edges", || {
            let sieve = Sieve::new();
            let cell1 = MeshEntity::Cell(1);
            let cell2 = MeshEntity::Cell(2);
            let shared_vertices = vec![MeshEntity::Vertex(1), MeshEntity::Vertex(2)];

            let vertices1 = vec![
                shared_vertices[0],
                shared_vertices[1],
                MeshEntity::Vertex(3),
            ];

            let vertices2 = vec![
                shared_vertices[0],
                shared_vertices[1],
                MeshEntity::Vertex(4),
            ];

            sieve
                .adjacency
                .entry(cell1.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices1 {
                sieve.add_arrow(cell1.clone(), vertex.clone());
            }

            sieve
                .adjacency
                .entry(cell2.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices2 {
                sieve.add_arrow(cell2.clone(), vertex.clone());
            }

            sieve.fill_missing_entities();

            let shared_edge = (shared_vertices[0], shared_vertices[1]);
            let edge_count = sieve.adjacency.get(&shared_edge.0).map_or(0, |adj| {
                adj.iter()
                    .filter(|entry| {
                        if let MeshEntity::Edge(_) = entry.key() {
                            sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                                edge_adj.contains_key(&shared_edge.1)
                            })
                        } else {
                            false
                        }
                    })
                    .count()
            });

            assert_eq!(edge_count, 1, "Shared edge should not be duplicated");
        });
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

#[cfg(test)]
mod tests {
    use dashmap::DashMap;

    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    #[test]
    fn test_stratify_empty_sieve() {
        // Test stratification on an empty Sieve
        let sieve = Sieve::new();
        let strata = sieve.stratify();

        // Ensure no strata are created for an empty sieve
        assert!(strata.is_empty());
    }

    #[test]
    fn test_stratify_single_entity_per_dimension() {
        // Create a Sieve and add one entity per dimension
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));
        sieve.add_arrow(MeshEntity::Edge(1), MeshEntity::Face(1));
        sieve.add_arrow(MeshEntity::Face(1), MeshEntity::Cell(1));

        // Add the entities directly to ensure they're present
        sieve.adjacency.entry(MeshEntity::Vertex(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Edge(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Face(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Cell(1)).or_insert_with(DashMap::new);

        let strata = sieve.stratify();

        // Verify that each dimension contains exactly one entity
        assert_eq!(strata.get(&0).unwrap().len(), 1); // Stratum for vertices
        assert_eq!(strata.get(&1).unwrap().len(), 1); // Stratum for edges
        assert_eq!(strata.get(&2).unwrap().len(), 1); // Stratum for faces
        assert_eq!(strata.get(&3).unwrap().len(), 1); // Stratum for cells

        // Verify the correct entities are in each stratum
        assert_eq!(strata.get(&0).unwrap()[0], MeshEntity::Vertex(1));
        assert_eq!(strata.get(&1).unwrap()[0], MeshEntity::Edge(1));
        assert_eq!(strata.get(&2).unwrap()[0], MeshEntity::Face(1));
        assert_eq!(strata.get(&3).unwrap()[0], MeshEntity::Cell(1));
    }

    #[test]
    fn test_stratify_multiple_entities_per_dimension() {
        // Create a Sieve with multiple entities in each dimension
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));
        sieve.add_arrow(MeshEntity::Vertex(2), MeshEntity::Edge(2));
        sieve.add_arrow(MeshEntity::Edge(1), MeshEntity::Face(1));
        sieve.add_arrow(MeshEntity::Edge(2), MeshEntity::Face(2));
        sieve.add_arrow(MeshEntity::Face(1), MeshEntity::Cell(1));
        sieve.add_arrow(MeshEntity::Face(2), MeshEntity::Cell(2));

        // Add the entities directly to ensure they're present
        sieve.adjacency.entry(MeshEntity::Vertex(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Vertex(2)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Edge(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Edge(2)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Face(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Face(2)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Cell(1)).or_insert_with(DashMap::new);
        sieve.adjacency.entry(MeshEntity::Cell(2)).or_insert_with(DashMap::new);

        let strata = sieve.stratify();

        // Verify that each dimension contains the correct number of entities
        assert_eq!(strata.get(&0).unwrap().len(), 2); // Two vertices
        assert_eq!(strata.get(&1).unwrap().len(), 2); // Two edges
        assert_eq!(strata.get(&2).unwrap().len(), 2); // Two faces
        assert_eq!(strata.get(&3).unwrap().len(), 2); // Two cells

        // Verify the correct entities are in each stratum
        assert!(strata.get(&0).unwrap().contains(&MeshEntity::Vertex(1)));
        assert!(strata.get(&0).unwrap().contains(&MeshEntity::Vertex(2)));
        assert!(strata.get(&1).unwrap().contains(&MeshEntity::Edge(1)));
        assert!(strata.get(&1).unwrap().contains(&MeshEntity::Edge(2)));
        assert!(strata.get(&2).unwrap().contains(&MeshEntity::Face(1)));
        assert!(strata.get(&2).unwrap().contains(&MeshEntity::Face(2)));
        assert!(strata.get(&3).unwrap().contains(&MeshEntity::Cell(1)));
        assert!(strata.get(&3).unwrap().contains(&MeshEntity::Cell(2)));
    }

    #[test]
    fn test_stratify_overlapping_entities() {
        // Create a Sieve with overlapping entities across dimensions
        let sieve = Sieve::new();
        sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));
        sieve.add_arrow(MeshEntity::Edge(1), MeshEntity::Vertex(1)); // Circular reference
        sieve.add_arrow(MeshEntity::Face(1), MeshEntity::Edge(1));

        let strata = sieve.stratify();

        // Verify that circular references are handled correctly
        assert_eq!(strata.get(&0).unwrap().len(), 1); // One vertex
        assert_eq!(strata.get(&1).unwrap().len(), 1); // One edge
        assert_eq!(strata.get(&2).unwrap().len(), 1); // One face

        // Verify the correct entities are in each stratum
        assert!(strata.get(&0).unwrap().contains(&MeshEntity::Vertex(1)));
        assert!(strata.get(&1).unwrap().contains(&MeshEntity::Edge(1)));
        assert!(strata.get(&2).unwrap().contains(&MeshEntity::Face(1)));
    }

    #[test]
    fn test_stratify_large_mesh() {
        // Create a large Sieve with many entities
        let sieve = Sieve::new();
        for i in 0..100 {
            sieve.add_arrow(MeshEntity::Vertex(i), MeshEntity::Edge(i));
            sieve.add_arrow(MeshEntity::Edge(i), MeshEntity::Face(i));
            sieve.add_arrow(MeshEntity::Face(i), MeshEntity::Cell(i));

            // Add the entities directly to ensure they're present
            sieve.adjacency.entry(MeshEntity::Vertex(i)).or_insert_with(DashMap::new);
            sieve.adjacency.entry(MeshEntity::Edge(i)).or_insert_with(DashMap::new);
            sieve.adjacency.entry(MeshEntity::Face(i)).or_insert_with(DashMap::new);
            sieve.adjacency.entry(MeshEntity::Cell(i)).or_insert_with(DashMap::new);
        }

        let strata = sieve.stratify();

        // Verify that each dimension contains the correct number of entities
        assert_eq!(strata.get(&0).unwrap().len(), 100); // 100 vertices
        assert_eq!(strata.get(&1).unwrap().len(), 100); // 100 edges
        assert_eq!(strata.get(&2).unwrap().len(), 100); // 100 faces
        assert_eq!(strata.get(&3).unwrap().len(), 100); // 100 cells
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
pub mod geometry_validation;
pub mod boundary_validation;

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

I want to develop a interface to make building simple domains easier in Hydra. I have now provided the current code for the `Domain` module. Please provide a detailed outline of the implementation for the interface, which will tie together the construction of a domain with a relatively simple API. For now, assume that we will develop this adapter in the file `src/interface_adapters/domain_adapter.rs`.