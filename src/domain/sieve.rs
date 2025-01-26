use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::mesh_entity::MeshEntity;
use log;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity` elements.
/// 
/// The `Sieve` uses an adjacency map to represent directed relations between entities in the mesh.
/// This structure enables querying relationships like cones, closures, and stars, making it a
/// versatile tool for managing mesh topology.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// Thread-safe adjacency map.
    /// - **Key**: A `MeshEntity` representing the source entity in the relationship.
    /// - **Value**: A `DashMap` of `MeshEntity` objects that are related to the key entity.
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new `Sieve` instance with an empty adjacency map.
    ///
    /// # Returns
    /// - A new `Sieve` with no relationships.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.
    ///
    /// # Parameters
    /// - `from`: The source entity.
    /// - `to`: The target entity related to the source.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).
    ///
    /// This operation is referred to as retrieving the **cone** of the entity.
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the cone is retrieved.
    ///
    /// # Returns
    /// - `Ok(Vec<MeshEntity>)`: Entities in the cone.
    /// - `Err(String)`: If the entity does not exist or has no relationships.
    pub fn cone(&self, point: &MeshEntity) -> Result<Vec<MeshEntity>, String> {
        self.adjacency
            .get(point)
            .map(|cone| cone.iter().map(|entry| *entry.key()).collect())
            .ok_or_else(|| format!("Entity {:?} not found in the adjacency map.", point))
    }

    /// Computes the closure of a given `MeshEntity`.
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the closure is computed.
    ///
    /// # Returns
    /// - `Result<DashMap<MeshEntity, ()>, String>`: Entities in the closure or an error message.
    pub fn closure(&self, point: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());
        let mut depth = 0;

        while !stack.is_empty() {
            depth += 1;

            // Safeguard against excessive depth (potential cyclic relationships)
            if depth > 1000 {
                return Err(format!(
                    "Exceeded maximum recursion depth while computing closure for {:?}.",
                    point
                ));
            }

            let keys: Vec<MeshEntity> = stack.iter().map(|entry| *entry.key()).collect();
            for p in keys {
                if result.insert(p.clone(), ()).is_none() {
                    if let Ok(cones) = self.cone(&p) {
                        for q in cones {
                            stack.insert(q, ());
                        }
                    }
                }
                stack.remove(&p);
            }
        }

        Ok(result)
    }


    /// Computes the star of a given `MeshEntity`.
    ///
    /// The star includes:
    /// - The entity itself.
    /// - All entities that directly cover it (supports).
    /// - All entities that the entity directly points to (cone).
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the star is computed.
    ///
    /// # Returns
    /// - `Result<DashMap<MeshEntity, ()>, String>`: The star of the entity, or an error message.
    pub fn star(&self, point: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        let result = DashMap::new();
        result.insert(point.clone(), ());

        // Include supports (entities pointing to `point`)
        match self.support(point) {
            Ok(supports) => {
                for support in supports {
                    result.insert(support, ());
                }
            }
            Err(err) => {
                log::warn!("Error retrieving supports for {:?}: {}", point, err);
            }
        }

        // Include cone (entities that `point` points to)
        match self.cone(point) {
            Ok(cones) => {
                for cone_entity in cones {
                    result.insert(cone_entity, ());
                }
            }
            Err(err) => {
                log::warn!("Error retrieving cone for {:?}: {}", point, err);
            }
        }

        Ok(result)
    }

    /// Retrieves all entities that support the given entity (`point`).
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which supports are retrieved.
    ///
    /// # Returns
    /// - `Ok(Vec<MeshEntity>)`: Supporting entities.
    /// - `Err(String)`: If no supporting entities are found.
    pub fn support(&self, point: &MeshEntity) -> Result<Vec<MeshEntity>, String> {
        let supports: Vec<MeshEntity> = self
            .adjacency
            .iter()
            .filter_map(|entry| {
                let from = entry.key();
                if entry.value().contains_key(point) {
                    Some(*from)
                } else {
                    None
                }
            })
            .collect();

        if supports.is_empty() {
            Err(format!("No supports found for entity {:?}.", point))
        } else {
            Ok(supports)
        }
    }


    /// Computes the meet operation for two entities, `p` and `q`.
    ///
    /// The meet is defined as the intersection of their closures.
    ///
    /// # Parameters
    /// - `p`: The first `MeshEntity`.
    /// - `q`: The second `MeshEntity`.
    ///
    /// # Returns
    /// - `Result<DashMap<MeshEntity, ()>, String>`: The intersection of the closures, or an error message.
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        let closure_p = self.closure(p)?;
        let closure_q = self.closure(q)?;
        let result = DashMap::new();

        closure_p.iter().for_each(|entry| {
            let key = entry.key();
            if closure_q.contains_key(key) {
                result.insert(key.clone(), ());
            }
        });

        Ok(result)
    }

    /// Computes the join operation for two entities, `p` and `q`.
    ///
    /// The join is defined as the union of their stars.
    ///
    /// # Parameters
    /// - `p`: The first `MeshEntity`.
    /// - `q`: The second `MeshEntity`.
    ///
    /// # Returns
    /// - `Result<DashMap<MeshEntity, ()>, String>`: The union of the stars, or an error message.
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        let star_p = self.star(p)?;
        let star_q = self.star(q)?;
        let result = DashMap::new();

        star_p.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });
        star_q.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });

        Ok(result)
    }

    /// Applies a given function in parallel to all adjacency map entries.
    ///
    /// # Parameters
    /// - `func`: A closure that operates on each key-value pair in the adjacency map.
    ///   The function is called with a tuple containing:
    ///   - A reference to a `MeshEntity` key.
    ///   - A `Vec<MeshEntity>` of entities related to the key.
    ///
    /// # Errors
    /// - Logs an error if the adjacency map is empty.
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, Vec<MeshEntity>)) + Sync + Send,
    {
        if self.adjacency.is_empty() {
            log::warn!("The adjacency map is empty. No relationships to iterate.");
            return;
        }

        let entries: Vec<_> = self
            .adjacency
            .iter()
            .map(|entry| {
                let key = *entry.key();
                let values: Vec<MeshEntity> = entry.value().iter().map(|e| *e.key()).collect();
                (key, values)
            })
            .collect();

        entries.par_iter().for_each(|entry| {
            func((&entry.0, entry.1.clone()));
        });
    }

    /// Converts the internal adjacency map into a standard `HashMap`.
    ///
    /// The resulting map contains each `MeshEntity` as a key and its related entities as a `Vec<MeshEntity>`.
    ///
    /// # Returns
    /// - An `FxHashMap` containing the adjacency relationships.
    pub fn to_adjacency_map(&self) -> FxHashMap<MeshEntity, Vec<MeshEntity>> {
        let mut adjacency_map: FxHashMap<MeshEntity, Vec<MeshEntity>> = FxHashMap::default();

        // Convert the thread-safe DashMap to FxHashMap
        for entry in self.adjacency.iter() {
            let key = *entry.key();
            let values: Vec<MeshEntity> = entry.value().iter().map(|v| *v.key()).collect();
            adjacency_map.insert(key, values);
        }

        adjacency_map
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
        let closure_result = sieve.closure(&vertex).unwrap();
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
        let support_result = sieve.support(&edge).unwrap();

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

        let star_result = sieve.star(&face).unwrap();

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

        let meet_result = sieve.meet(&vertex1, &vertex2).unwrap();

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

        let join_result = sieve.join(&vertex1, &vertex2).unwrap();

        assert!(join_result.contains_key(&vertex1), "Join result should contain vertex1");
        assert!(join_result.contains_key(&vertex2), "Join result should contain vertex2");
        assert_eq!(join_result.len(), 2);
    }
}

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test adding multiple arrows and querying the cone for complex entity relationships.
    fn test_complex_cone() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let face1 = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge1);
        sieve.add_arrow(edge1, face1);

        let cone_vertex1 = sieve.cone(&vertex1).expect("Failed to compute cone for vertex1");
        let cone_vertex2 = sieve.cone(&vertex2).expect("Failed to compute cone for vertex2");
        let cone_edge1 = sieve.cone(&edge1).expect("Failed to compute cone for edge1");

        assert_eq!(cone_vertex1, vec![edge1]);
        assert_eq!(cone_vertex2, vec![edge1]);
        assert_eq!(cone_edge1, vec![face1]);
    }

    #[test]
    /// Test closure computation for deeply nested relationships.
    fn test_closure_complex() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let edge2 = MeshEntity::Edge(2);
        let face1 = MeshEntity::Face(1);
        let cell1 = MeshEntity::Cell(1);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge2);
        sieve.add_arrow(edge1, face1);
        sieve.add_arrow(edge2, face1);
        sieve.add_arrow(face1, cell1);

        let closure_vertex1 = sieve
            .closure(&vertex1)
            .expect("Failed to compute closure for vertex1");

        assert!(closure_vertex1.contains_key(&vertex1));
        assert!(closure_vertex1.contains_key(&edge1));
        assert!(closure_vertex1.contains_key(&face1));
        assert!(closure_vertex1.contains_key(&cell1));
        assert!(!closure_vertex1.contains_key(&edge2));
    }

    #[test]
    /// Test star computation for entities with multiple supporting relationships.
    fn test_star_multiple_supports() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let face1 = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge1);
        sieve.add_arrow(edge1, face1);

        let star_face1 = sieve.star(&face1).expect("Failed to compute star for face1");

        assert!(star_face1.contains_key(&face1));
        assert!(star_face1.contains_key(&edge1));
        assert!(!star_face1.contains_key(&vertex1));
        assert!(!star_face1.contains_key(&vertex2));
    }

    #[test]
    /// Test meet operation for entities with intersecting closures.
    fn test_meet_intersecting_closures() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let face1 = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge1);
        sieve.add_arrow(edge1, face1);

        let meet_result = sieve
            .meet(&vertex1, &vertex2)
            .expect("Failed to compute meet for vertex1 and vertex2");

        assert!(meet_result.contains_key(&edge1));
        assert!(meet_result.contains_key(&face1));
        assert!(!meet_result.contains_key(&vertex1));
        assert!(!meet_result.contains_key(&vertex2));
    }

    #[test]
    /// Test join operation for entities with disjoint relationships.
    fn test_join_disjoint_relationships() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let edge2 = MeshEntity::Edge(2);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge2);

        let join_result = sieve
            .join(&vertex1, &vertex2)
            .expect("Failed to compute join for vertex1 and vertex2");

        assert!(join_result.contains_key(&vertex1));
        assert!(join_result.contains_key(&vertex2));
        assert!(join_result.contains_key(&edge1));
        assert!(join_result.contains_key(&edge2));
    }

    #[test]
    /// Test adjacency map conversion with a complex relationship graph.
    fn test_to_adjacency_map_complex() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge1 = MeshEntity::Edge(1);
        let face1 = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge1);
        sieve.add_arrow(vertex2, edge1);
        sieve.add_arrow(edge1, face1);

        let adjacency_map = sieve.to_adjacency_map();
        assert_eq!(adjacency_map.get(&vertex1).unwrap(), &vec![edge1]);
        assert_eq!(adjacency_map.get(&vertex2).unwrap(), &vec![edge1]);
        assert_eq!(adjacency_map.get(&edge1).unwrap(), &vec![face1]);
        assert!(adjacency_map.get(&face1).is_none());
    }

    #[test]
    /// Test parallel iteration over adjacency map for performance scenarios.
    fn test_parallel_iteration() {
        let sieve = Sieve::new();
        for i in 1..=100 {
            sieve.add_arrow(MeshEntity::Vertex(i), MeshEntity::Edge(i));
        }

        use std::sync::Mutex;

        // Wrap the results in a Mutex to allow thread-safe access.
        let results = Mutex::new(Vec::new());

        sieve.par_for_each_adjacent(|(entity, related)| {
            if let MeshEntity::Vertex(id) = entity {
                let mut results_lock = results.lock().unwrap();
                results_lock.push((*id, related.len()));
            }
        });

        let results = results.into_inner().unwrap(); // Extract the results from the Mutex.
        assert_eq!(results.len(), 100);
        for (_id, len) in results {
            assert_eq!(len, 1);
        }
    }

    #[test]
    fn test_cone_error_handling() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);

        let result = sieve.cone(&vertex);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Entity Vertex(1) not found in the adjacency map."
        );
    }

    #[test]
    fn test_closure_error_handling() {
        let sieve = Sieve::new();
        let mut current_entity = MeshEntity::Vertex(1);
    
        // Create a deep chain of relationships to exceed depth limit
        for i in 2..=2000 {
            let next_entity = MeshEntity::Vertex(i);
            sieve.add_arrow(current_entity, next_entity);
            current_entity = next_entity;
        }
    
        let result = sieve.closure(&MeshEntity::Vertex(1));
    
        // Ensure the method detects excessive recursion depth
        assert!(result.is_err(), "Expected an error due to excessive recursion depth.");
        assert_eq!(
            result.unwrap_err(),
            "Exceeded maximum recursion depth while computing closure for Vertex(1)."
        );
    }
    

    #[test]
    fn test_self_referential_arrow() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);

        sieve.add_arrow(vertex, vertex);

        let closure_result = sieve.closure(&vertex).unwrap();
        assert!(closure_result.contains_key(&vertex));
        assert_eq!(closure_result.len(), 1);

        let cone_result = sieve.cone(&vertex).unwrap();
        assert!(cone_result.contains(&vertex));
        assert_eq!(cone_result.len(), 1);
    }

    #[test]
    fn test_parallel_vs_sequential_iteration() {
        let sieve = Sieve::new();
        for i in 1..=100 {
            sieve.add_arrow(MeshEntity::Vertex(i), MeshEntity::Edge(i));
        }
    
        let mut sequential_results = Vec::new();
        sieve.adjacency.iter().for_each(|entry| {
            sequential_results.push((*entry.key(), entry.value().len()));
        });
    
        use std::sync::Mutex;
        let parallel_results = Mutex::new(Vec::new());
        sieve.par_for_each_adjacent(|(entity, related)| {
            let mut results = parallel_results.lock().unwrap();
            results.push((entity.clone(), related.len()));
        });
    
        let mut parallel_results = parallel_results.into_inner().unwrap();
    
        // Sort results by entity ID for consistent comparison
        sequential_results.sort_by(|a, b| a.0.get_id().cmp(&b.0.get_id()));
        parallel_results.sort_by(|a, b| a.0.get_id().cmp(&b.0.get_id()));
    
        assert_eq!(
            sequential_results, parallel_results,
            "Sequential and parallel results should match after sorting."
        );
    }
    

    #[test]
    fn test_dynamic_relationship_updates() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge1 = MeshEntity::Edge(1);
        let edge2 = MeshEntity::Edge(2);

        sieve.add_arrow(vertex, edge1);

        let initial_cone = sieve.cone(&vertex).unwrap();
        assert!(initial_cone.contains(&edge1));
        assert!(!initial_cone.contains(&edge2));

        sieve.add_arrow(vertex, edge2);
        let updated_cone = sieve.cone(&vertex).unwrap();
        assert!(updated_cone.contains(&edge1));
        assert!(updated_cone.contains(&edge2));
        assert_eq!(updated_cone.len(), 2);
    }
}
