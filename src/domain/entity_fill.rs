use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::FxHashSet;

impl Sieve {
    /// Infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.
    ///
    /// For 2D meshes, this method generates edges by connecting consecutive vertices of a cell.
    /// These edges are then added to the sieve's adjacency map, along with their relationships
    /// to the corresponding vertices and back-references from vertices to edges.
    ///
    /// - **Input**: Existing cells and vertices in the sieve.
    /// - **Output**: Updated sieve with inferred edges and relationships.
    pub fn fill_missing_entities(&self) {
        let mut edge_set: FxHashSet<(MeshEntity, MeshEntity)> = FxHashSet::default(); // To track processed edges
        let mut next_edge_id = 0; // Counter for assigning unique IDs to new edges
        let mut arrows_to_add: Vec<(MeshEntity, MeshEntity)> = Vec::new(); // Temporary storage for relationships to add

        // Collect cells and their associated vertices to avoid modifying the map during iteration
        let cell_vertices: Vec<(MeshEntity, Vec<MeshEntity>)> = self.adjacency.iter()
            .filter_map(|entry| {
                let cell = entry.key();
                if let MeshEntity::Cell(_) = cell {
                    let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                    Some((cell.clone(), vertices))
                } else {
                    None
                }
            }).collect();

        for (cell, vertices) in cell_vertices {
            // Skip cells with fewer than three vertices (not valid for forming edges)
            if vertices.len() < 3 {
                eprintln!("Skipping cell with fewer than 3 vertices: {:?}", cell);
                continue;
            }

            eprintln!("Processing cell: {:?}", cell);

            // Iterate over pairs of consecutive vertices, wrapping around for the last edge
            for i in 0..vertices.len() {
                let (v1, v2) = (
                    vertices[i].clone(),
                    vertices[(i + 1) % vertices.len()].clone(),
                );

                // Ensure edges are consistently defined as (min, max) to avoid duplication
                let edge_key = if v1 < v2 {
                    (v1.clone(), v2.clone())
                } else {
                    (v2.clone(), v1.clone())
                };

                // Skip if the edge has already been processed
                if edge_set.contains(&edge_key) {
                    eprintln!("Edge {:?} already processed, skipping.", edge_key);
                    continue;
                }

                // Mark the edge as processed
                edge_set.insert(edge_key.clone());

                // Create a new edge entity
                let edge = MeshEntity::Edge(next_edge_id);
                next_edge_id += 1;

                // Collect arrows representing relationships between vertices and the new edge
                arrows_to_add.push((v1.clone(), edge.clone()));
                arrows_to_add.push((v2.clone(), edge.clone()));
                arrows_to_add.push((edge.clone(), v1.clone()));
                arrows_to_add.push((edge.clone(), v2.clone()));

                eprintln!(
                    "Created edge {:?} between {:?} and {:?}",
                    edge, v1, v2
                );
            }
        }

        // Add all collected arrows to the sieve's adjacency map outside the iteration
        for (from, to) in arrows_to_add {
            self.add_arrow(from, to);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use dashmap::DashMap;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::sieve::Sieve;

    /// Runs a test function with a timeout, ensuring it completes within 5 seconds.
    fn run_with_timeout<F: FnOnce()>(test_name: &str, func: F) {
        let start = Instant::now();
        func();
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "{} test timed out!", test_name);
    }

    /// Verifies that `fill_missing_entities` does not add edges to an empty sieve.
    #[test]
    fn test_fill_missing_entities_with_empty_sieve() {
        run_with_timeout("test_fill_missing_entities_with_empty_sieve", || {
            let sieve = Sieve::new();
            sieve.fill_missing_entities();

            // Check that no new edges or relationships were added
            assert!(
                sieve.adjacency.is_empty(),
                "No edges should be created for an empty sieve"
            );
        });
    }

    /// Tests that `fill_missing_entities` generates edges for a single cell with three vertices.
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

            // Add the cell and its vertices to the sieve
            sieve
                .adjacency
                .entry(cell.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices {
                sieve.add_arrow(cell.clone(), vertex.clone());
            }

            // Fill in the missing edges
            sieve.fill_missing_entities();

            // Verify that the expected edges were created
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

    /// Ensures that `fill_missing_entities` does not duplicate edges shared by multiple cells.
    #[test]
    fn test_fill_missing_entities_no_duplicate_edges() {
        run_with_timeout("test_fill_missing_entities_no_duplicate_edges", || {
            let sieve = Sieve::new();
            let cell1 = MeshEntity::Cell(1);
            let cell2 = MeshEntity::Cell(2);
            let shared_vertices = vec![MeshEntity::Vertex(1), MeshEntity::Vertex(2)];

            // Define vertices for two cells, sharing some vertices
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

            // Add the cells and their vertices to the sieve
            sieve
                .adjacency
                .entry(cell1.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices1 {
                sieve.add_arrow(cell1.clone(), vertex.clone());
            }

            sieve
                .adjacency
                .entry(cell2.clone())
                .or_insert_with(DashMap::new);
            for vertex in &vertices2 {
                sieve.add_arrow(cell2.clone(), vertex.clone());
            }

            // Fill in the missing edges
            sieve.fill_missing_entities();

            // Verify that the shared edge is not duplicated
            let shared_edge = (shared_vertices[0], shared_vertices[1]);
            let edge_count = sieve.adjacency.get(&shared_edge.0).map_or(0, |adj| {
                adj.iter()
                    .filter(|entry| {
                        if let MeshEntity::Edge(_) = entry.key() {
                            sieve.adjacency.get(entry.key()).map_or(false, |edge_adj| {
                                edge_adj.contains_key(&shared_edge.1)
                            })
                        } else {
                            false
                        }
                    })
                    .count()
            });

            assert_eq!(edge_count, 1, "Shared edge should not be duplicated");
        });
    }
}
