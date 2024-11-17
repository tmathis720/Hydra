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
        let edge_set: DashMap<(MeshEntity, MeshEntity), MeshEntity> = DashMap::new();
        let mut next_edge_id = 0;

        // Loop through each cell and infer its edges
        self.adjacency.iter().for_each(|entry| {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                if vertices.len() < 3 {
                    // Skip cells with fewer than 3 vertices
                    return;
                }

                // Create edges by connecting vertices
                for i in 0..vertices.len() {
                    let v1 = vertices[i].clone();
                    let v2 = vertices[(i + 1) % vertices.len()].clone();
                    let edge_key = if v1 < v2 { (v1.clone(), v2.clone()) } else { (v2.clone(), v1.clone()) };

                    // Add the edge if it doesn't exist
                    edge_set.entry(edge_key).or_insert_with(|| {
                        let edge = MeshEntity::Edge(next_edge_id);
                        next_edge_id += 1;
                        self.add_arrow(v1.clone(), edge.clone());
                        self.add_arrow(v2.clone(), edge.clone());
                        edge
                    });
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    #[test]
    fn test_fill_missing_entities_with_empty_sieve() {
        let sieve = Sieve::new();
        sieve.fill_missing_entities();

        assert!(sieve.adjacency.is_empty(), "No edges should be created for an empty sieve");
    }

    #[test]
    fn test_fill_missing_entities_for_single_cell() {
        let sieve = Sieve::new();
        let cell = MeshEntity::Cell(1);
        let vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
        ];

        sieve.adjacency.entry(cell.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices {
            sieve.add_arrow(cell.clone(), vertex.clone());
        }

        sieve.fill_missing_entities();

        let expected_edges = vec![
            (MeshEntity::Vertex(1), MeshEntity::Vertex(2)),
            (MeshEntity::Vertex(2), MeshEntity::Vertex(3)),
            (MeshEntity::Vertex(3), MeshEntity::Vertex(1)),
        ];

        for (v1, v2) in expected_edges {
            let edge_exists = sieve.adjacency.get(&v1).map_or(false, |adj| {
                adj.iter().any(|entry| {
                    if let MeshEntity::Edge(_) = entry.key() {
                        sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                            edge_adj.contains_key(&v2)
                        })
                    } else {
                        false
                    }
                })
            });
            assert!(edge_exists, "Edge ({:?}, {:?}) should exist", v1, v2);
        }
    }

    #[test]
    fn test_fill_missing_entities_for_multiple_cells() {
        let sieve = Sieve::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let vertices1 = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
        ];
        let vertices2 = vec![
            MeshEntity::Vertex(2),
            MeshEntity::Vertex(3),
            MeshEntity::Vertex(4),
        ];

        sieve.adjacency.entry(cell1.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices1 {
            sieve.add_arrow(cell1.clone(), vertex.clone());
        }

        sieve.adjacency.entry(cell2.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices2 {
            sieve.add_arrow(cell2.clone(), vertex.clone());
        }

        sieve.fill_missing_entities();

        let expected_edges = vec![
            (MeshEntity::Vertex(1), MeshEntity::Vertex(2)),
            (MeshEntity::Vertex(2), MeshEntity::Vertex(3)),
            (MeshEntity::Vertex(3), MeshEntity::Vertex(1)),
            (MeshEntity::Vertex(3), MeshEntity::Vertex(4)),
            (MeshEntity::Vertex(2), MeshEntity::Vertex(4)),
        ];

        for (v1, v2) in expected_edges {
            let edge_exists = sieve.adjacency.get(&v1).map_or(false, |adj| {
                adj.iter().any(|entry| {
                    if let MeshEntity::Edge(_) = entry.key() {
                        sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                            edge_adj.contains_key(&v2)
                        })
                    } else {
                        false
                    }
                })
            });
            assert!(edge_exists, "Edge ({:?}, {:?}) should exist", v1, v2);
        }
    }

    #[test]
    fn test_fill_missing_entities_no_duplicate_edges() {
        let sieve = Sieve::new();
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let shared_vertices = vec![
            MeshEntity::Vertex(1),
            MeshEntity::Vertex(2),
        ];

        let vertices1 = vec![
            shared_vertices[0],
            shared_vertices[1],
            MeshEntity::Vertex(3),
        ];

        let vertices2 = vec![
            shared_vertices[0],
            shared_vertices[1],
            MeshEntity::Vertex(4),
        ];

        sieve.adjacency.entry(cell1.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices1 {
            sieve.add_arrow(cell1.clone(), vertex.clone());
        }

        sieve.adjacency.entry(cell2.clone()).or_insert_with(DashMap::new);
        for vertex in &vertices2 {
            sieve.add_arrow(cell2.clone(), vertex.clone());
        }

        sieve.fill_missing_entities();

        let shared_edge = (shared_vertices[0], shared_vertices[1]);
        let edge_count = sieve.adjacency.get(&shared_edge.0).map_or(0, |adj| {
            adj.iter().filter(|entry| {
                if let MeshEntity::Edge(_) = entry.key() {
                    sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                        edge_adj.contains_key(&shared_edge.1)
                    })
                } else {
                    false
                }
            }).count()
        });

        assert_eq!(edge_count, 1, "Shared edge should not be duplicated");
    }
}

