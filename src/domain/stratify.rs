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
