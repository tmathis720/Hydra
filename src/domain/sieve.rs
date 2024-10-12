use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use crossbeam::thread;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity`  
/// elements, organized in an adjacency map.  
///
/// The adjacency map tracks directed relations between entities in the mesh.  
/// It supports operations such as adding relationships, querying direct  
/// relations (cones), and computing closure and star sets for entities.  
/// 
/// Example usage:
/// 
///    let sieve = Sieve::new();  
///    let vertex = MeshEntity::Vertex(1);  
///    let edge = MeshEntity::Edge(2);  
///    sieve.add_arrow(vertex, edge);  
///    let cone = sieve.cone(&vertex).unwrap();  
/// 
#[derive(Clone, Debug)]
pub struct Sieve {
    /// A thread-safe adjacency map where each key is a `MeshEntity`,  
    /// and the value is a set of `MeshEntity` objects related to the key.  
    pub adjacency: Arc<RwLock<FxHashMap<MeshEntity, FxHashSet<MeshEntity>>>>,
}

impl Sieve {
    /// Creates a new empty `Sieve` instance with an empty adjacency map.  
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    assert!(sieve.adjacency.read().unwrap().is_empty());  
    /// 
    pub fn new() -> Self {
        Sieve {
            adjacency: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.  
    /// The relationship is stored in the adjacency map from the `from` entity  
    /// to the `to` entity.  
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    sieve.add_arrow(vertex, edge);  
    ///
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        let mut adjacency = self.adjacency.write().unwrap();
        adjacency.entry(from).or_insert_with(FxHashSet::default).insert(to);
    }

    /// Retrieves all entities directly related to the given entity (`point`).  
    /// This operation is referred to as retrieving the cone of the entity.  
    /// Returns `None` if there are no related entities.  
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    sieve.add_arrow(vertex, MeshEntity::Edge(1));  
    ///    let cone = sieve.cone(&vertex);  
    /// 
    pub fn cone(&self, point: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        self.adjacency.read().unwrap().get(point).cloned()
    }

    /// Computes the transitive closure for a given `MeshEntity`.  
    /// The closure includes the entity itself and all entities reachable  
    /// through arrows from the entity.
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    sieve.add_arrow(vertex, MeshEntity::Edge(1));  
    ///    let closure_set = sieve.closure(&vertex);  
    ///
    pub fn closure(&self, point: &MeshEntity) -> FxHashSet<MeshEntity> {
        let mut result = FxHashSet::default();
        let mut stack = vec![point.clone()];
        while let Some(p) = stack.pop() {
            if let Some(cones) = self.cone(&p) {
                for q in cones {
                    if result.insert(q.clone()) {
                        stack.push(q.clone());
                    }
                }
            }
        }
        result
    }

    /// Computes the star of a given `MeshEntity`.  
    /// The star includes all entities directly related to the entity  
    /// (cone and support sets).
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(1);  
    ///    sieve.add_arrow(vertex, edge);  
    ///    let star_set = sieve.star(&vertex);  
    ///
    pub fn star(&self, point: &MeshEntity) -> FxHashSet<MeshEntity> {
        let mut result = FxHashSet::default();
        let mut stack = vec![point.clone()];

        while let Some(p) = stack.pop() {
            if result.insert(p.clone()) {
                if let Some(cones) = self.cone(&p) {
                    for q in cones {
                        stack.push(q.clone());
                    }
                }
                let supports = self.support(&p);
                for q in supports {
                    stack.push(q.clone());
                }
            }
        }
        result
    }

    /// Retrieves all entities that support the given entity (`point`).  
    /// These are the entities that have an arrow pointing to `point`.
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(1);  
    ///    sieve.add_arrow(vertex, edge);  
    ///    let support_set = sieve.support(&edge);  
    ///
    pub fn support(&self, point: &MeshEntity) -> FxHashSet<MeshEntity> {
        let adjacency = self.adjacency.read().unwrap();
        adjacency
            .iter()
            .filter_map(|(from, to_set)| if to_set.contains(point) { Some(from.clone()) } else { None })
            .collect()
    }

    /// Computes the meet operation for two entities, `p` and `q`.  
    /// This is the intersection of their closures (minimal separator).
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex1 = MeshEntity::Vertex(1);  
    ///    let vertex2 = MeshEntity::Vertex(2);  
    ///    let edge = MeshEntity::Edge(1);  
    ///    sieve.add_arrow(vertex1, edge);  
    ///    sieve.add_arrow(vertex2, edge);  
    ///    let meet_set = sieve.meet(&vertex1, &vertex2);  
    ///
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> FxHashSet<MeshEntity> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        closure_p.intersection(&closure_q).cloned().collect()
    }

    /// Computes the join operation for two entities, `p` and `q`.  
    /// This is the union of their stars (minimal separator).
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    let vertex1 = MeshEntity::Vertex(1);  
    ///    let vertex2 = MeshEntity::Vertex(2);  
    ///    let edge = MeshEntity::Edge(1);  
    ///    sieve.add_arrow(vertex1, edge);  
    ///    sieve.add_arrow(vertex2, edge);  
    ///    let join_set = sieve.join(&vertex1, &vertex2);  
    ///
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> FxHashSet<MeshEntity> {
        let star_p = self.star(p);
        let star_q = self.star(q);

        let join_result: FxHashSet<MeshEntity> = star_p.union(&star_q).cloned().collect();
        join_result
    }

    /// Applies a given function in parallel to all adjacency map entries.  
    /// This function is executed concurrently over each entity and its  
    /// corresponding set of related entities.
    ///
    /// Example usage:
    /// 
    ///    sieve.par_for_each_adjacent(|(entity, adj_set)| {  
    ///        println!("{:?} -> {:?}", entity, adj_set);  
    ///    });  
    ///
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, &FxHashSet<MeshEntity>)) + Sync + Send,
    {
        let adjacency = self.adjacency.read().unwrap();
        adjacency.par_iter().for_each(func);
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
    /// all transitive relationships.  
    fn test_closure() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        sieve.add_arrow(vertex, edge);
        let closure_result = sieve.closure(&vertex);
        assert!(closure_result.contains(&edge));
    }

    #[test]
    /// Test that verifies the support of an entity includes the  
    /// correct supporting entities.  
    fn test_support() {
        let mut sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let support_result = sieve.support(&edge);

        assert!(support_result.contains(&vertex));
    }

    #[test]
    /// Test that verifies the star of an entity includes both cone and  
    /// support sets of the entity.  
    fn test_star() {
        let mut sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(vertex, edge);
        sieve.add_arrow(edge, face);

        let star_result = sieve.star(&face);

        assert!(star_result.contains(&edge));
        assert!(star_result.contains(&vertex));
    }

    #[test]
    /// Test that verifies the meet operation between two entities returns  
    /// the correct minimal separator (intersection of closures).  
    fn test_meet() {
        let mut sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge);
        sieve.add_arrow(vertex2, edge);
        sieve.add_arrow(edge, face);

        let meet_result = sieve.meet(&vertex1, &vertex2);

        assert!(meet_result.contains(&edge));
    }

    #[test]
    /// Test that verifies the join operation between two entities returns  
    /// the correct minimal separator (union of stars).  
    fn test_join() {
        let mut sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(vertex1, edge);
        sieve.add_arrow(vertex2, edge);
        sieve.add_arrow(edge, face);

        let join_result = sieve.join(&vertex1, &vertex2);

        assert!(join_result.contains(&vertex1), "Join result should contain vertex1");
        assert!(join_result.contains(&vertex2), "Join result should contain vertex2");
        assert!(join_result.contains(&edge), "Join result should contain the edge");
        assert!(join_result.contains(&face), "Join result should contain the face");
    }
}
