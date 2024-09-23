use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use std::collections::HashSet;

impl Sieve {
    /// Given cells and vertices, infer and add edges (in 2D) or faces (in 3D).
    /// For 2D: Cells will generate edges, and edges will connect vertices.
    pub fn fill_missing_entities(&mut self) {
        let mut edge_set: HashSet<(MeshEntity, MeshEntity)> = HashSet::new();

        // Loop through each cell and infer its edges (for 2D meshes)
        for (cell, vertices) in &self.adjacency {
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = vertices.iter().collect();
                for i in 0..vertices.len() {
                    let v1 = vertices[i];
                    let v2 = vertices[(i + 1) % vertices.len()];
                    let edge = if v1 < v2 { (*v1, *v2) } else { (*v2, *v1) };
                    edge_set.insert(edge);
                }
            }
        }

        // Add the deduced edges to the sieve
        for (v1, v2) in edge_set {
            let edge = MeshEntity::Edge(self.adjacency.len());  // Generate unique ID for new edge
            self.add_arrow(v1, edge);
            self.add_arrow(v2, edge);
        }
    }
}
