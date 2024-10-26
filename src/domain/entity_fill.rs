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
        // Use DashMap instead of FxHashSet for concurrent access.
        let edge_set: DashMap<(MeshEntity, MeshEntity), ()> = DashMap::new();

        // Loop through each cell and infer its edges (for 2D meshes)
        self.adjacency.iter().for_each(|entry| {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                // Connect each vertex with its neighboring vertex to form edges.
                for i in 0..vertices.len() {
                    let v1 = vertices[i].clone();
                    let v2 = vertices[(i + 1) % vertices.len()].clone();
                    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    edge_set.insert(edge, ());
                }
            }
        });

        // Add the deduced edges to the sieve.
        let edge_count = self.adjacency.len();
        edge_set.into_iter().enumerate().for_each(|(index, ((v1, v2), _))| {
            // Generate a unique ID for the new edge.
            let edge = MeshEntity::Edge(edge_count + index);
            self.add_arrow(v1, edge.clone());
            self.add_arrow(v2, edge);
        });
    }
}
