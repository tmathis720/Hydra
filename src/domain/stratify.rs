use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use std::collections::HashMap;

/// Organizes the mesh entities into strata based on their dimension.
/// E.g., vertices are in stratum 0, edges in stratum 1, and cells in stratum 2.
impl Sieve {
    pub fn stratify(&self) -> HashMap<usize, Vec<MeshEntity>> {
        let mut strata: HashMap<usize, Vec<MeshEntity>> = HashMap::new();
        for (entity, _) in &self.adjacency {
            let dimension = match entity {
                MeshEntity::Vertex(_) => 0,
                MeshEntity::Edge(_) => 1,
                MeshEntity::Face(_) => 2,
                MeshEntity::Cell(_) => 3,
            };
            strata.entry(dimension).or_insert_with(Vec::new).push(*entity);
        }
        strata
    }
}
