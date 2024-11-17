### Summary of the Problem

#### **Hydra Context and Task**
Hydra is a sophisticated computational fluid dynamics (CFD) framework designed for geophysical fluid dynamics simulations. A core task in Hydra is to manage and manipulate mesh entities (vertices, edges, faces, and cells) using a structured data model. The `Sieve` structure organizes relationships between these entities, ensuring consistency and enabling complex simulations.

The **specific task** for this part of Hydra involves:
1. **Filling Missing Entities**: Automatically inferring and creating edges (for 2D meshes) or faces (for 3D meshes) based on existing cells and vertices.
2. **Verifying Integrity**: Ensuring inferred entities are correctly added and associated with the mesh.
3. **Avoiding Duplicates**: Ensuring shared edges or faces across cells are not duplicated.

The method being tested is `fill_missing_entities`, which iterates over cells, infers missing edges, and adds these edges to the adjacency map while maintaining relationships with the appropriate vertices.

---

#### **Error in Tests**
The failure of `test_fill_missing_entities_for_single_cell` suggests that **inferred edges are not being correctly added or associated with their vertices in the adjacency map**. The test attempts to verify the existence of edges (e.g., between vertices `(1, 2)`), but the assertion fails because:
- The adjacency map does not contain the expected edges, or
- The edges are not correctly associated with the vertices.

#### **Root Causes of Failure**
1. **Edge Deduplication**: 
   - In `fill_missing_entities`, deduplication of edges using `DashMap` may not be functioning correctly.
   - This could result in missing edges or improper associations.

2. **Edge Association**:
   - The adjacency relationships between vertices and edges may not be established properly. For example, vertex `1` should reference edge `(1, 2)` in the adjacency map.

3. **Test Verification Logic**:
   - The logic in the test to validate edge presence in the adjacency map may not accurately match how edges are stored.

---

#### **Background of `fill_missing_entities`**
1. **Input**:
   - A `Sieve` structure with cells and their associated vertices already defined.
   - Example: Cell `(1)` is connected to vertices `(1, 2, 3)`.

2. **Processing**:
   - Loop over cells.
   - Infer edges by pairing adjacent vertices in a loop (e.g., `(1, 2)`, `(2, 3)`, `(3, 1)` for a triangular cell).
   - Add these edges to the adjacency map, ensuring:
     - Edges are associated with their constituent vertices.
     - Duplicate edges (e.g., shared edges across cells) are avoided.

3. **Expected Output**:
   - An adjacency map with inferred edges correctly connected to vertices.
   - For example, vertex `1` should reference edge `(1, 2)`.

---

#### **Test Objectives**
The `test_fill_missing_entities_for_single_cell` aims to validate the following:
1. **Edges Are Created**: Verify that edges `(1, 2)`, `(2, 3)`, and `(3, 1)` are inferred and added.
2. **Edges Are Associated**: Check that these edges are correctly connected to their vertices in the adjacency map.

---

### Next Steps
1. **Debugging `fill_missing_entities`**:
   - Ensure edges are correctly deduplicated and associated with vertices.
   - Verify that edges are uniquely created and connected to both vertices.

2. **Refining Test Logic**:
   - Adjust test validation to match how edges and vertices are stored in the adjacency map.

3. **Rerun and Validate Tests**:
   - Ensure all edge-related tests pass, including tests for multiple cells and shared edges.

This task is central to maintaining the integrity of Hydra’s mesh data structure, which underpins all CFD computations. By resolving these issues, we ensure robust and reliable handling of mesh entities.

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

`src/domain/entity_fill.rs`

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
        let edge_set: DashMap<(MeshEntity, MeshEntity), MeshEntity> = DashMap::new();
        let mut next_edge_id = 0;

        // Loop through each cell and infer its edges
        self.adjacency.iter().for_each(|entry| {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                if vertices.len() < 3 {
                    // Skip cells with fewer than 3 vertices
                    return;
                }

                // Create edges by connecting vertices
                for i in 0..vertices.len() {
                    let v1 = vertices[i].clone();
                    let v2 = vertices[(i + 1) % vertices.len()].clone();
                    let edge_key = if v1 < v2 { (v1.clone(), v2.clone()) } else { (v2.clone(), v1.clone()) };

                    // Add the edge if it doesn't exist
                    edge_set.entry(edge_key).or_insert_with(|| {
                        let edge = MeshEntity::Edge(next_edge_id);
                        next_edge_id += 1;
                        self.add_arrow(v1.clone(), edge.clone());
                        self.add_arrow(v2.clone(), edge.clone());
                        edge
                    });
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    #[test]
    fn test_fill_missing_entities_with_empty_sieve() {
        let sieve = Sieve::new();
        sieve.fill_missing_entities();

        assert!(sieve.adjacency.is_empty(), "No edges should be created for an empty sieve");
    }

    #[test]
    fn test_fill_missing_entities_for_single_cell() {
        let sieve = Sieve::new();
        let cell = MeshEntity::Cell(1);
        let vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
        ];

        sieve.adjacency.entry(cell.clone()).or_insert_with(DashMap::new);
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
    }

    #[test]
    fn test_fill_missing_entities_for_multiple_cells() {
        let sieve = Sieve::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let vertices1 = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
        ];
        let vertices2 = vec![
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
            MeshEntity::Vertex(4),
        ];

        sieve.adjacency.entry(cell1.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices1 {
            sieve.add_arrow(cell1.clone(), vertex.clone());
        }

        sieve.adjacency.entry(cell2.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices2 {
            sieve.add_arrow(cell2.clone(), vertex.clone());
        }

        sieve.fill_missing_entities();

        let expected_edges = vec![
            (MeshEntity::Vertex(1), MeshEntity::Vertex(2)),
            (MeshEntity::Vertex(2), MeshEntity::Vertex(3)),
            (MeshEntity::Vertex(3), MeshEntity::Vertex(1)),
            (MeshEntity::Vertex(3), MeshEntity::Vertex(4)),
            (MeshEntity::Vertex(2), MeshEntity::Vertex(4)),
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
    }

    #[test]
    fn test_fill_missing_entities_no_duplicate_edges() {
        let sieve = Sieve::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let shared_vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
        ];

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

        sieve.adjacency.entry(cell1.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices1 {
            sieve.add_arrow(cell1.clone(), vertex.clone());
        }

        sieve.adjacency.entry(cell2.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices2 {
            sieve.add_arrow(cell2.clone(), vertex.clone());
        }

        sieve.fill_missing_entities();

        let shared_edge = (shared_vertices[0], shared_vertices[1]);
        let edge_count = sieve.adjacency.get(&shared_edge.0).map_or(0, |adj| {
            adj.iter().filter(|entry| {
                if let MeshEntity::Edge(_) = entry.key() {
                    sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                        edge_adj.contains_key(&shared_edge.1)
                    })
                } else {
                    false
                }
            }).count()
        });

        assert_eq!(edge_count, 1, "Shared edge should not be duplicated");
    }
}

```

