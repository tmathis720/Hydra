use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashSet;

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
        let mut edge_set: FxHashSet<(MeshEntity, MeshEntity)> = FxHashSet::default();

        // Acquire a read lock to access the adjacency data.
        let adjacency = self.adjacency.read().unwrap();

        // Loop through each cell and infer its edges (for 2D meshes)
        for (cell, vertices) in adjacency.iter() {
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = vertices.iter().collect();
                // Connect each vertex with its neighboring vertex to form edges.
                for i in 0..vertices.len() {
                    let v1 = vertices[i];
                    let v2 = vertices[(i + 1) % vertices.len()];
                    let edge = if v1 < v2 { (*v1, *v2) } else { (*v2, *v1) };
                    edge_set.insert(edge);
                }
            }
        }

        // Add the deduced edges to the sieve.
        let adjacency = self.adjacency.write().unwrap();
        for (v1, v2) in edge_set {
            // Generate a unique ID for the new edge.
            let edge = MeshEntity::Edge(adjacency.len());  
            self.add_arrow(v1, edge);
            self.add_arrow(v2, edge);
        }
    }
}
