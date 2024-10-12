use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashMap;

/// Implements a stratification method for the `Sieve` structure.  
/// Stratification organizes the mesh entities into different strata based on  
/// their dimensions:  
/// - Stratum 0: Vertices  
/// - Stratum 1: Edges  
/// - Stratum 2: Faces  
/// - Stratum 3: Cells  
///
/// This method categorizes each `MeshEntity` into its corresponding stratum and  
/// returns a map where the keys are the dimension (stratum) and the values  
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
    pub fn stratify(&self) -> FxHashMap<usize, Vec<MeshEntity>> {
        let mut strata: FxHashMap<usize, Vec<MeshEntity>> = FxHashMap::default();
        
        // Acquire a read lock to access the adjacency data.
        let adjacency = self.adjacency.read().unwrap();
        
        // Iterate over the adjacency map to classify entities by their dimension.
        for (entity, _) in adjacency.iter() {
            // Determine the dimension of the current entity.
            let dimension = match entity {
                MeshEntity::Vertex(_) => 0,  // Stratum 0 for vertices
                MeshEntity::Edge(_) => 1,    // Stratum 1 for edges
                MeshEntity::Face(_) => 2,    // Stratum 2 for faces
                MeshEntity::Cell(_) => 3,    // Stratum 3 for cells
            };
            
            // Add the entity to the appropriate stratum in the map.
            strata.entry(dimension).or_insert_with(Vec::new).push(*entity);
        }
        
        strata
    }
}
