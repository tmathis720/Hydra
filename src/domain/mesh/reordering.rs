use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use dashmap::DashMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{collections::VecDeque, sync::Arc};
use rayon::prelude::*;

/// Reorders mesh entities using the Cuthill-McKee algorithm.
///
/// The algorithm reduces the bandwidth of sparse matrices by reordering entities
/// to improve memory locality. Starting from the node with the smallest degree,
/// neighbors are visited in increasing order of their degree.
///
/// # Arguments
/// * `entities` - A slice of all mesh entities to reorder.
/// * `adjacency` - A map defining adjacency relationships between mesh entities.
///
/// # Returns
/// * `Vec<MeshEntity>` - A vector of mesh entities reordered by the Cuthill-McKee algorithm.
pub fn cuthill_mckee(
    entities: &[MeshEntity],
    adjacency: &FxHashMap<MeshEntity, Vec<MeshEntity>>,
) -> Vec<MeshEntity> {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::new();

    // Find the starting entity with the smallest degree (number of neighbors).
    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);
        visited.insert(*start);
    }

    // Perform Cuthill-McKee reordering using breadth-first traversal.
    while let Some(entity) = queue.pop_front() {
        ordered.push(entity);

        if let Some(neighbors) = adjacency.get(&entity) {
            let mut sorted_neighbors: Vec<_> = neighbors
                .iter()
                .filter(|&&n| !visited.contains(&n)) // Only process unvisited neighbors.
                .cloned()
                .collect();

            // Sort neighbors by increasing degree.
            sorted_neighbors.sort_by_key(|n| adjacency.get(n).map_or(0, |neighbors| neighbors.len()));

            // Enqueue the sorted neighbors.
            for neighbor in sorted_neighbors {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    ordered
}

impl Mesh {
    /// Applies a new ordering to the mesh entities based on the provided list of new IDs.
    ///
    /// This function updates the mesh's internal entity identifiers to match the specified
    /// ordering and updates the sieve structure to maintain consistency.
    ///
    /// # Arguments
    /// * `new_order` - A slice of new IDs corresponding to the desired order of entities.
    pub fn apply_reordering(&mut self, new_order: &[usize]) {
        // Map old entities to new entities based on the new order.
        let entities: Vec<_> = self.entities.read().unwrap().iter().cloned().collect();
        let mut id_mapping: FxHashMap<MeshEntity, MeshEntity> = FxHashMap::default();

        for (new_id, entity) in new_order.iter().zip(entities.iter()) {
            let new_entity = entity.with_id(*new_id);
            id_mapping.insert(*entity, new_entity);
        }

        // Update the entities set with the new IDs.
        let mut entities_write = self.entities.write().unwrap();
        entities_write.clear();
        for new_entity in id_mapping.values() {
            entities_write.insert(*new_entity);
        }

        // Update the sieve with the new IDs.
        let new_adjacency = DashMap::new();
        for entry in self.sieve.adjacency.iter() {
            let old_from = *entry.key();
            let new_from = *id_mapping.get(&old_from).unwrap_or(&old_from);

            let new_cone = DashMap::new();
            for cone_entry in entry.value().iter() {
                let old_to = *cone_entry.key();
                let new_to = *id_mapping.get(&old_to).unwrap_or(&old_to);
                new_cone.insert(new_to, ());
            }
            new_adjacency.insert(new_from, new_cone);
        }

        let mut sieve = Arc::clone(&self.sieve);
        Arc::make_mut(&mut sieve).adjacency = new_adjacency;
        self.sieve = sieve;
    }

    /// Computes the Reverse Cuthill-McKee (RCM) ordering of the mesh entities.
    ///
    /// The RCM algorithm minimizes the bandwidth of sparse matrices by reordering
    /// entities in reverse order of their Cuthill-McKee ordering.
    ///
    /// # Arguments
    /// * `start_node` - The starting entity for the RCM algorithm.
    ///
    /// # Returns
    /// * `Vec<MeshEntity>` - A vector of entities reordered by the RCM algorithm.
    pub fn rcm_ordering(&self, start_node: MeshEntity) -> Vec<MeshEntity> {
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        let mut ordering = Vec::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        // Perform breadth-first traversal and order nodes by degree.
        while let Some(node) = queue.pop_front() {
            ordering.push(node);
            if let Some(neighbors) = self.sieve.cone(&node) {
                let mut sorted_neighbors: Vec<_> = neighbors
                    .into_iter()
                    .filter(|n| !visited.contains(n)) // Only unvisited neighbors.
                    .collect();

                // Sort neighbors by degree.
                sorted_neighbors.sort_by_key(|n| self.sieve.cone(n).map_or(0, |set| set.len()));

                // Enqueue the sorted neighbors.
                for neighbor in sorted_neighbors {
                    queue.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        ordering.reverse(); // Reverse the ordering to obtain RCM.
        ordering
    }

    /// Reorders elements in the mesh using Morton order (Z-order curve).
    ///
    /// Morton order improves memory access patterns by organizing elements in a
    /// space-filling curve. This method operates on 2D elements specified by their
    /// x and y coordinates.
    ///
    /// # Arguments
    /// * `elements` - A mutable slice of 2D elements represented as `(x, y)` coordinate pairs.
    pub fn reorder_by_morton_order(&mut self, elements: &mut [(u32, u32)]) {
        elements.par_sort_by_key(|&(x, y)| Self::morton_order_2d(x, y));
    }

    /// Computes the Morton order (Z-order curve) for a 2D point `(x, y)`.
    ///
    /// Morton order is calculated by interleaving the bits of the x and y coordinates
    /// to produce a single 64-bit value. This encoding ensures spatial locality.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate of the point.
    /// * `y` - The y-coordinate of the point.
    ///
    /// # Returns
    /// * `u64` - The Morton order value for the point.
    pub fn morton_order_2d(x: u32, y: u32) -> u64 {
        // Helper function to interleave the bits of a 32-bit integer.
        fn part1by1(n: u32) -> u64 {
            let mut n = n as u64;
            n = (n | (n << 16)) & 0x0000_0000_ffff_0000;
            n = (n | (n << 8)) & 0x0000_ff00_00ff_0000;
            n = (n | (n << 4)) & 0x00f0_00f0_00f0_00f0;
            n = (n | (n << 2)) & 0x0c30_0c30_0c30_0c30;
            n = (n | (n << 1)) & 0x2222_2222_2222_2222;
            n
        }

        // Interleave x and y bits to compute Morton order.
        part1by1(x) | (part1by1(y) << 1)
    }
}
