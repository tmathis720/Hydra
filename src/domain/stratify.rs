use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashMap;

/// Organizes the mesh entities into strata based on their dimension.
/// E.g., vertices are in stratum 0, edges in stratum 1, and cells in stratum 2.
impl Sieve {
    pub fn stratify(&self) -> FxHashMap<usize, Vec<MeshEntity>> {
        let mut strata: FxHashMap<usize, Vec<MeshEntity>> = FxHashMap::default();
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
