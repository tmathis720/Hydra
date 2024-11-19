use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashSet;

impl Sieve {
    /// Infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.
    /// For 2D meshes, this method generates edges by connecting vertices of a cell.
    /// These edges are then associated with the corresponding vertices in the sieve.
    pub fn fill_missing_entities(&self) {
        let mut edge_set: FxHashSet<(MeshEntity, MeshEntity)> = FxHashSet::default();
        let mut next_edge_id = 0;

        for entry in self.adjacency.iter() {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                if vertices.len() < 3 {
                    eprintln!("Skipping cell with fewer than 3 vertices: {:?}", cell);
                    continue;
                }

                eprintln!("Processing cell: {:?}", cell);

                for i in 0..vertices.len() {
                    let (v1, v2) = (vertices[i].clone(), vertices[(i + 1) % vertices.len()].clone());
                    let edge_key = if v1 < v2 { (v1.clone(), v2.clone()) } else { (v2.clone(), v1.clone()) };

                    if edge_set.contains(&edge_key) {
                        eprintln!("Edge {:?} already processed, skipping.", edge_key);
                        continue;
                    }
                    edge_set.insert(edge_key.clone());

                    let edge = MeshEntity::Edge(next_edge_id);
                    next_edge_id += 1;

                    // Add arrows from vertices to edge
                    self.add_arrow(v1.clone(), edge.clone());
                    self.add_arrow(v2.clone(), edge.clone());

                    // Add arrows from edge to vertices (bidirectional)
                    self.add_arrow(edge.clone(), v1.clone());
                    self.add_arrow(edge.clone(), v2.clone());

                    eprintln!("Created edge {:?} between {:?} and {:?}", edge, v1, v2);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use dashmap::DashMap;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    fn run_with_timeout<F: FnOnce()>(test_name: &str, func: F) {
        let start = Instant::now();
        func();
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "{} test timed out!", test_name);
    }

    #[test]
    fn test_fill_missing_entities_with_empty_sieve() {
        run_with_timeout("test_fill_missing_entities_with_empty_sieve", || {
            let sieve = Sieve::new();
            sieve.fill_missing_entities();

            assert!(sieve.adjacency.is_empty(), "No edges should be created for an empty sieve");
        });
    }

    #[test]
    fn test_fill_missing_entities_for_single_cell() {
        run_with_timeout("test_fill_missing_entities_for_single_cell", || {
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
        });
    }

    #[test]
    fn test_fill_missing_entities_no_duplicate_edges() {
        run_with_timeout("test_fill_missing_entities_no_duplicate_edges", || {
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
        });
    }
}
