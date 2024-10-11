use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use crossbeam::thread;

#[derive(Clone)]
pub struct Sieve {
    pub adjacency: Arc<RwLock<FxHashMap<MeshEntity, FxHashSet<MeshEntity>>>>,
}

impl Sieve {
    pub fn new() -> Self {
        Sieve {
            adjacency: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        let mut adjacency = self.adjacency.write().unwrap();
        adjacency
            .entry(from)
            .or_insert_with(FxHashSet::default)
            .insert(to);
    }

    pub fn cone(&self, point: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        let adjacency = self.adjacency.read().unwrap();
        adjacency.get(point).cloned()
    }

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

    pub fn support(&self, point: &MeshEntity) -> FxHashSet<MeshEntity> {
        let adjacency = self.adjacency.read().unwrap();
        adjacency
            .iter()
            .filter_map(|(from, to_set)| if to_set.contains(point) { Some(from.clone()) } else { None })
            .collect()
    }

    // Meet operation: Minimal separator of closure(p) and closure(q)
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> FxHashSet<MeshEntity> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        closure_p.intersection(&closure_q).cloned().collect()
    }

    // Join operation: Minimal separator of star(p) and star(q)
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> FxHashSet<MeshEntity> {

        let star_p = self.star(p);  // Get all entities related to p
        let star_q = self.star(q);  // Get all entities related to q

        // Return the union of both stars (the minimal separator)
        let join_result: FxHashSet<MeshEntity> = star_p.union(&star_q).cloned().collect();
        join_result
    }

    // Parallel iteration over adjacency entries
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, &FxHashSet<MeshEntity>)) + Sync + Send,
    {
        let adjacency = self.adjacency.read().unwrap();
        adjacency.par_iter().for_each(|entry| {
            func(entry);
        });
    }

    
}

// Unit tests for the Sieve structure and its operations

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_add_arrow_and_cone() {
        let mut sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let cone_result = sieve.cone(&vertex).unwrap();

        assert!(cone_result.contains(&edge));
    }

    #[test]
    fn test_closure() {
        let mut sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(vertex, edge);
        sieve.add_arrow(edge, face);

        let closure_result = sieve.closure(&vertex);

        assert!(closure_result.contains(&edge));
        assert!(closure_result.contains(&face));
    }

    #[test]
    fn test_support() {
        let mut sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let support_result = sieve.support(&edge);

        assert!(support_result.contains(&vertex));
    }

    #[test]
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
