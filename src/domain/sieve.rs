use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
        
        // Include supports (entities pointing to `point`)
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        
        // Include cone (entities that `point` points to)
        if let Some(cones) = self.cone(point) {
            for cone_entity in cones {
                result.insert(cone_entity, ());
            }
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

    /// Converts the internal adjacency map into a standard `HashMap` where each key
    /// is a `MeshEntity` and the value is a vector of related `MeshEntity` objects.
    /// 
    /// This function is useful for processing the adjacency relationships in contexts
    /// where thread-safety is not required.
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

        let cone_vertex1 = sieve.cone(&vertex1).unwrap();
        let cone_vertex2 = sieve.cone(&vertex2).unwrap();
        let cone_edge1 = sieve.cone(&edge1).unwrap();

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

        let closure_vertex1 = sieve.closure(&vertex1);
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

        let star_face1 = sieve.star(&face1);
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

        let meet_result = sieve.meet(&vertex1, &vertex2);
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

        let join_result = sieve.join(&vertex1, &vertex2);
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

    

}
