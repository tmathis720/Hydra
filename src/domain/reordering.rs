use std::collections::{HashMap, HashSet, VecDeque};
use crate::domain::mesh_entity::MeshEntity;

/// Reorders mesh entities using the Cuthill-McKee algorithm.
/// This improves memory locality and is useful for solver optimization.
pub fn cuthill_mckee(entities: &[MeshEntity], adjacency: &HashMap<MeshEntity, Vec<MeshEntity>>) -> Vec<MeshEntity> {
    let mut visited: HashSet<MeshEntity> = HashSet::new();
    let mut queue: VecDeque<MeshEntity> = VecDeque::new();
    let mut ordered: Vec<MeshEntity> = Vec::new();

    // Start by adding the lowest degree vertex
    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);  // Dereference start to get the MeshEntity value
        visited.insert(*start);   // Insert into the visited set
    }

    // Breadth-first search for reordering
    while let Some(entity) = queue.pop_front() {
        ordered.push(entity);

        // Get the neighbors of the current entity
        if let Some(neighbors) = adjacency.get(&entity) {
            // Filter out neighbors that have already been visited
            let mut sorted_neighbors: Vec<_> = neighbors.iter()
                .filter(|&&n| !visited.contains(&n))  // Double dereference to handle &MeshEntity correctly
                .cloned()  // Clone to avoid borrowing issues
                .collect();

            // Sort neighbors by their degree (number of adjacent entities)
            sorted_neighbors.sort_by_key(|n| adjacency.get(n).map_or(0, |neighbors| neighbors.len()));

            // Add the sorted neighbors to the queue and mark them as visited
            for neighbor in sorted_neighbors {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    ordered
}
